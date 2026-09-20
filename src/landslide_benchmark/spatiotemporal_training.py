import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import load_split
from .metrics import add_counts, confusion_counts, detailed_metrics, empty_counts
from .paths import (
    DEFAULT_AEF_DIR,
    DEFAULT_S2_DIR,
    DEFAULT_SPATIOTEMPORAL_OUTPUT,
    DEFAULT_SPLIT,
    LEGACY_AEF_CHECKPOINT,
)
from .spatiotemporal_data import AEFSpatioTemporalDataset, TEMPORAL_CLASSES, load_or_compute_statistics
from .spatiotemporal_model import AEFSpatioTemporalNet, JointLoss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def temporal_metrics(confusion: np.ndarray) -> dict:
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    tp = np.diag(confusion).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    observed = support > 0
    total = support.sum()
    return {
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "balanced_accuracy": float(recall[observed].mean()) if observed.any() else 0.0,
        "macro_f1_observed": float(f1[observed].mean()) if observed.any() else 0.0,
        "valid_samples": int(total),
        "observed_classes": np.flatnonzero(observed).tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def set_warmup_lrs(optimizer, epoch: int, warmup_epochs: int, target_lrs, min_lrs) -> None:
    fraction = min(epoch / max(warmup_epochs, 1), 1.0)
    for group, target, minimum in zip(optimizer.param_groups, target_lrs, min_lrs):
        group["lr"] = minimum + fraction * (target - minimum)


def set_spatial_trainable(model: AEFSpatioTemporalNet, trainable: bool) -> None:
    for module in (model.stem, model.body, model.head):
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, grad_clip=1.0):
    training = optimizer is not None
    model.train(training)
    totals = Counter()
    spatial_counts = empty_counts()
    month_confusion = np.zeros((TEMPORAL_CLASSES, TEMPORAL_CLASSES), dtype=np.int64)
    progress = tqdm(loader, desc="train" if training else "val", leave=False)

    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        months = batch["month"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None
        with torch.set_grad_enabled(training), torch.amp.autocast(device.type, enabled=use_amp):
            outputs = model(images)
            losses = criterion(outputs, masks, months)
        if training:
            if scaler is None:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            else:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

        batch_size = images.shape[0]
        totals["samples"] += batch_size
        for name in ("total", "spatial", "bce", "dice", "temporal"):
            totals[name] += float(losses[name].detach()) * batch_size
        predictions = torch.sigmoid(outputs["segmentation_logits"].detach()) >= 0.5
        add_counts(spatial_counts, confusion_counts(predictions, masks))
        truth = months.detach().cpu().numpy()
        predicted_month = outputs["temporal_logits"].argmax(1).detach().cpu().numpy()
        np.add.at(month_confusion, (truth, predicted_month), 1)
        progress.set_postfix(loss=totals["total"] / max(totals["samples"], 1))

    samples = max(totals["samples"], 1)
    spatial = detailed_metrics(spatial_counts)
    pixels = sum(spatial_counts.values())
    spatial_diagnostics = {
        "predicted_positive_ratio": (spatial_counts["tp"] + spatial_counts["fp"]) / max(pixels, 1),
        "target_positive_ratio": (spatial_counts["tp"] + spatial_counts["fn"]) / max(pixels, 1),
    }
    return {
        "loss": totals["total"] / samples,
        "spatial_loss": totals["spatial"] / samples,
        "bce_loss": totals["bce"] / samples,
        "dice_loss": totals["dice"] / samples,
        "temporal_loss": totals["temporal"] / samples,
        "segmentation": spatial,
        "segmentation_diagnostics": spatial_diagnostics,
        "temporal": temporal_metrics(month_confusion),
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, model_config, train_config, statistics, best):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "model_config": model_config,
            "train_config": train_config,
            "training_statistics": statistics,
            "best_scores": best,
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the joint AEF segmentation and event-month model.")
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--aef-dir", type=Path, default=DEFAULT_AEF_DIR)
    parser.add_argument("--s2-dir", type=Path, default=DEFAULT_S2_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SPATIOTEMPORAL_OUTPUT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5, help="Spatial/shared learning rate.")
    parser.add_argument("--temporal-lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--temporal-min-lr", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--temporal-dropout", type=float, default=0.15)
    parser.add_argument("--temporal-shared-gradient-scale", type=float, default=0.10)
    parser.add_argument("--freeze-spatial-epochs", type=int, default=5)
    parser.add_argument("--temporal-weight", type=float, default=0.25)
    parser.add_argument("--bce-weight", type=float, default=0.70)
    parser.add_argument("--dice-weight", type=float, default=0.30)
    parser.add_argument("--circular-smoothing", type=float, default=0.10)
    parser.add_argument("--joint-spatial-weight", type=float, default=0.70)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spatial-init", type=Path, default=LEGACY_AEF_CHECKPOINT)
    parser.add_argument("--no-spatial-init", action="store_true", help="Train the spatial path from scratch.")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit-train", type=int, help="Smoke-test only; changes the fitted statistics.")
    parser.add_argument("--limit-val", type=int, help="Smoke-test only.")
    parser.add_argument("--force-train", action="store_true", help="Ignore last.pth and start over.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.joint_spatial_weight <= 1.0:
        raise ValueError("--joint-spatial-weight must be in [0, 1].")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    seed_everything(args.seed)

    splits = load_split(args.split_json, args.aef_dir, args.s2_dir)
    train_events = splits["train"][: args.limit_train] if args.limit_train else splits["train"]
    val_events = splits["val"][: args.limit_val] if args.limit_val else splits["val"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    statistics = load_or_compute_statistics(args.output_dir / "training_stats.json", train_events)
    train_dataset = AEFSpatioTemporalDataset(train_events, augment=args.augment)
    val_dataset = AEFSpatioTemporalDataset(val_events, augment=False)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory and device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model_config = {
        "in_channels": 64,
        "hidden_channels": args.hidden_channels,
        "residual_scale": 0.1,
        "temporal_classes": TEMPORAL_CLASSES,
        "temporal_dropout": args.temporal_dropout,
        "temporal_shared_gradient_scale": args.temporal_shared_gradient_scale,
    }
    model = AEFSpatioTemporalNet(**model_config).to(device)
    criterion = JointLoss(
        statistics["temporal_class_weights"],
        temporal_weight=args.temporal_weight,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        circular_smoothing=args.circular_smoothing,
    ).to(device)

    last_path = args.output_dir / "last.pth"
    resume_available = last_path.is_file() and not args.force_train
    spatial_warm_started = False
    if not resume_available and not args.no_spatial_init:
        if args.spatial_init.is_file():
            spatial_checkpoint = torch.load(args.spatial_init, map_location="cpu", weights_only=False)
            spatial_state = spatial_checkpoint.get("model_state_dict", spatial_checkpoint)
            model.load_spatial_state(spatial_state)
            spatial_warm_started = True
            print(f"Initialized spatial path from: {args.spatial_init}")
        else:
            print(f"Spatial initialization checkpoint not found; training from scratch: {args.spatial_init}")

    spatial_parameters = list(model.stem.parameters()) + list(model.body.parameters()) + list(model.head.parameters())
    temporal_parameters = list(model.temporal_branch.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": spatial_parameters, "lr": args.min_lr, "name": "spatial_shared"},
            {"params": temporal_parameters, "lr": args.temporal_min_lr, "name": "temporal"},
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        threshold=3e-4,
        threshold_mode="abs",
        min_lr=[args.min_lr, args.temporal_min_lr],
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    train_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}

    start_epoch, bad_epochs = 1, 0
    best = {"joint": -1.0, "spatial_iou": -1.0, "epoch_joint": 0, "epoch_spatial": 0}
    history_path = args.output_dir / "history.json"
    history = []
    if last_path.is_file() and not args.force_train:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("format_version") != 2:
            raise ValueError("last.pth uses the obsolete v1 architecture; use the spatiotemporal_v2 output directory.")
        if checkpoint["training_statistics"]["split_fingerprint"] != statistics["split_fingerprint"]:
            raise ValueError("last.pth belongs to a different training split; use another output directory.")
        model.load_state_dict(checkpoint["model_state_dict"])
        spatial_warm_started = bool(checkpoint.get("train_config", {}).get("spatial_warm_started", False))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best.update(checkpoint.get("best_scores", {}))
        if history_path.is_file():
            history = [row for row in json.loads(history_path.read_text(encoding="utf-8")) if row["epoch"] < start_epoch]
        bad_epochs = max(0, start_epoch - 1 - int(best["epoch_spatial"]))
        print(f"Resuming from epoch {start_epoch}.")

    train_config["spatial_warm_started"] = spatial_warm_started

    if start_epoch > args.epochs:
        print(f"Training already completed at epoch {start_epoch - 1}; increase --epochs to continue.")
        return

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device} | AMP: {use_amp} | Parameters: {parameters:,}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"Temporal class counts (Jan-Dec, no-event): {statistics['month_counts']}")
    print(f"No-event samples with a positive mask: {statistics['no_event_samples_with_positive_mask']}")
    print(f"AEF preprocessing: {statistics['aef_preprocessing']} (no normalization)")
    print(f"Positive pixel ratio: {statistics['positive_ratio']:.6f} | spatial BCE pos_weight: disabled")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            spatial_frozen = spatial_warm_started and epoch <= args.freeze_spatial_epochs
            set_spatial_trainable(model, not spatial_frozen)
            if epoch <= args.warmup_epochs:
                set_warmup_lrs(
                    optimizer,
                    epoch,
                    args.warmup_epochs,
                    [args.lr, args.temporal_lr],
                    [args.min_lr, args.temporal_min_lr],
                )
            current_lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
            train_result = run_epoch(model, train_loader, criterion, device, optimizer, scaler, args.grad_clip)
            with torch.inference_mode():
                val_result = run_epoch(model, val_loader, criterion, device)
            if epoch > args.warmup_epochs:
                scheduler.step(val_result["spatial_loss"])
            spatial_iou = val_result["segmentation"]["landslide"]["iou"]
            spatial_metrics = val_result["segmentation"]["landslide"]
            predicted_positive_ratio = val_result["segmentation_diagnostics"]["predicted_positive_ratio"]
            month_f1 = val_result["temporal"]["macro_f1_observed"]
            joint = args.joint_spatial_weight * spatial_iou + (1.0 - args.joint_spatial_weight) * month_f1
            row = {
                "epoch": epoch,
                "learning_rates": current_lrs,
                "joint_score": joint,
                "train": train_result,
                "val": val_result,
            }
            history.append(row)
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            improved_joint = joint > best["joint"]
            improved_spatial = spatial_iou > best["spatial_iou"]
            if improved_joint:
                best.update(joint=joint, epoch_joint=epoch)
            if improved_spatial:
                best.update(spatial_iou=spatial_iou, epoch_spatial=epoch)
                bad_epochs = 0
            else:
                bad_epochs += 1
            save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, model_config, train_config, statistics, best)
            if improved_joint:
                save_checkpoint(args.output_dir / "best.pth", model, optimizer, scheduler, scaler, epoch, model_config, train_config, statistics, best)
            if improved_spatial:
                save_checkpoint(args.output_dir / "best_spatial.pth", model, optimizer, scheduler, scaler, epoch, model_config, train_config, statistics, best)
            print(
                f"Epoch {epoch:03d}/{args.epochs}: spatial_lr={current_lrs['spatial_shared']:.2e} "
                f"temporal_lr={current_lrs['temporal']:.2e} loss={val_result['loss']:.4f} "
                f"P/R/F1/IoU={spatial_metrics['precision']:.4f}/{spatial_metrics['recall']:.4f}/"
                f"{spatial_metrics['f1']:.4f}/{spatial_iou:.4f} pred_pos={predicted_positive_ratio:.4f} "
                f"month_acc={val_result['temporal']['accuracy']:.4f} "
                f"month_macro_F1={month_f1:.4f} joint={joint:.4f} spatial_frozen={spatial_frozen}"
            )
            if bad_epochs >= args.patience:
                print(f"Early stopping: spatial IoU did not improve for {args.patience} epochs.")
                break
    except KeyboardInterrupt:
        print("\nStopped safely. last.pth contains the most recently completed epoch.")
        raise


if __name__ == "__main__":
    main()
