import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import OpticalMaskDataset, load_split
from .models import UNet3D
from .paths import DEFAULT_AEF_DIR, DEFAULT_NORM, DEFAULT_OPTICAL_OUTPUT, DEFAULT_S2_DIR, DEFAULT_SPLIT
from .training import load_history, run_epoch, save_checkpoint


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parser_for():
    defaults = dict(epochs=80, batch=12, lr=1e-3, weight_decay=1e-2, output=DEFAULT_OPTICAL_OUTPUT)
    parser = argparse.ArgumentParser(description="Train the five-region optical segmentation model.")
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--aef-dir", type=Path, default=DEFAULT_AEF_DIR)
    parser.add_argument("--s2-dir", type=Path, default=DEFAULT_S2_DIR)
    parser.add_argument("--norm-json", type=Path, default=DEFAULT_NORM)
    parser.add_argument("--output-dir", type=Path, default=defaults["output"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--align-to-mask-storage", action="store_true", help="Match the joint AEF reference-mask orientation.")
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", action="store_true", help="Enable FP16 AMP on CUDA (recommended for Optical).")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    return parser


def train():
    args = parser_for().parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    seed_everything(args.seed)
    splits = load_split(args.split_json, args.aef_dir, args.s2_dir)
    train_events = splits["train"][: args.limit_train] if args.limit_train else splits["train"]
    val_events = splits["val"][: args.limit_val] if args.limit_val else splits["val"]

    train_ds = OpticalMaskDataset(train_events, args.norm_json, augment=True, align_to_mask_storage=args.align_to_mask_storage)
    val_ds = OpticalMaskDataset(val_events, args.norm_json, augment=False, align_to_mask_storage=args.align_to_mask_storage)
    probe, _ = train_ds[0]
    model_cfg = {"in_channels": probe.shape[1], "num_classes": 1, "img_res": 128, "dropout": 0.0}
    model = UNet3D(**model_cfg).to(device)
    pos_weight, dice_weight = 5.0, 0.5

    loaders = {
        "train": DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True),
        "val": DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if args.amp and device.type == "cuda" else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = args.output_dir / "best.pth", args.output_dir / "last.pth"
    history_path = args.output_dir / "history.json"
    start_epoch, best_iou, bad_epochs, history = 1, -1.0, 0, []
    if last_path.exists() and not args.force_train:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_iou = checkpoint.get("best_val_iou", -1.0)
        bad_epochs = checkpoint.get("bad_epochs", 0)
        history = load_history(history_path, checkpoint["epoch"])
        if start_epoch > args.epochs:
            print(f"Training already completed at epoch {checkpoint['epoch']}.")
            return
        print(f"Resuming at epoch {start_epoch}/{args.epochs}")

    train_cfg = vars(args).copy()
    train_cfg = {key: str(value) if isinstance(value, Path) else value for key, value in train_cfg.items()}
    print(f"Model: optical | Device: {device} | Train: {len(train_ds)} | Val: {len(val_ds)}")
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], optimizer, device, pos_weight, dice_weight, scaler)
        with torch.inference_mode():
            val_metrics = run_epoch(model, loaders["val"], None, device, pos_weight, dice_weight)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        improved = val_metrics["iou"] > best_iou
        best_iou = max(best_iou, val_metrics["iou"])
        bad_epochs = 0 if improved else bad_epochs + 1
        save_checkpoint(
            last_path, model, optimizer, epoch, best_iou, model_cfg, train_cfg,
            scaler=scaler, bad_epochs=bad_epochs,
        )
        if improved:
            save_checkpoint(
                best_path, model, optimizer, epoch, best_iou, model_cfg, train_cfg,
                scaler=scaler, bad_epochs=bad_epochs,
            )
        print(f"Epoch {epoch:03d}/{args.epochs}: val F1={val_metrics['f1']:.4f}, IoU={val_metrics['iou']:.4f}")
        if bad_epochs >= args.patience:
            print(f"Early stopping after {args.patience} epochs without IoU improvement.")
            break
