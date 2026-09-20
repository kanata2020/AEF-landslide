import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from landslide_benchmark.data import OpticalMaskDataset, load_split
from landslide_benchmark.metrics import add_counts, confusion_counts, detailed_metrics, empty_counts
from landslide_benchmark.models import UNet3D
from landslide_benchmark.paths import (
    DEFAULT_AEF_DIR,
    DEFAULT_NORM,
    DEFAULT_OPTICAL_OUTPUT,
    DEFAULT_S2_DIR,
    DEFAULT_SPLIT,
    LEGACY_OPTICAL_CHECKPOINT,
    PACKAGE_ROOT,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Optical on the held-out split.")
    parser.add_argument("--model", choices=("optical",), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--aef-dir", type=Path, default=DEFAULT_AEF_DIR)
    parser.add_argument("--s2-dir", type=Path, default=DEFAULT_S2_DIR)
    parser.add_argument("--norm-json", type=Path, default=DEFAULT_NORM)
    parser.add_argument("--split", default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def default_checkpoint():
    new_path = DEFAULT_OPTICAL_OUTPUT / "best.pth"
    legacy = LEGACY_OPTICAL_CHECKPOINT
    return new_path if new_path.is_file() else legacy


def load_model(checkpoint, device):
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    state = {key[4:] if key.startswith("net.") else key: value for key, value in state.items()}
    config = checkpoint.get("config", checkpoint.get("model_cfg", {}))
    model = UNet3D(**checkpoint.get("model_cfg", config))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def metric_row(name, values, samples=None):
    prefix = f"{name:<18}" + (f" {samples:>7d}" if samples is not None else "")
    return prefix + "".join(f" {values[key]:>10.6f}" for key in ("precision", "recall", "f1", "iou"))


def print_report(report):
    overall = report["overall"]
    print("\n1. Overall (Landslide Class) Metrics")
    print(metric_row("Landslide", overall["landslide"]))
    print("\n2. Class-wise Metrics (Mask Level)")
    print(metric_row("Landslide", overall["landslide"]))
    print(metric_row("Non-Landslide", overall["non_landslide"]))
    print(f"Accuracy: {overall['accuracy']:.6f}")
    print("\n3. Macro Average Metrics")
    print(metric_row("Macro", overall["macro"]))
    print("\n4. Per-region Metrics (Landslide Class)")
    print(f"{'Region':<18} {'Samples':>7} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10}")
    for region, values in report["per_region"].items():
        print(metric_row(region, values["landslide"], values["samples"]))


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint or default_checkpoint()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    events = load_split(args.split_json, args.aef_dir, args.s2_dir)[args.split]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = load_model(checkpoint, device)
    dataset = OpticalMaskDataset(events, args.norm_json,
                                 align_to_mask_storage=checkpoint.get("train_cfg", {}).get("align_to_mask_storage", False))
    batch_size = args.batch_size or 4
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=args.num_workers)

    overall, regions, samples = empty_counts(), defaultdict(empty_counts), defaultdict(int)
    offset = 0
    print(f"Model: {args.model} | Device: {device} | Events: {len(events)} | Checkpoint: {checkpoint_path}")
    with torch.inference_mode():
        for batch_index, (inputs, targets) in enumerate(loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            logits = output["segmentation"]
            predictions = torch.sigmoid(logits) >= args.threshold
            for index in range(inputs.shape[0]):
                event = events[offset + index]
                counts = confusion_counts(predictions[index], targets[index])
                add_counts(overall, counts)
                add_counts(regions[event["region"]], counts)
                samples[event["region"]] += 1
            offset += inputs.shape[0]
            if batch_index % 10 == 0 or offset == len(events):
                print(f"Processed {offset}/{len(events)}", flush=True)

    report = {
        "model": args.model,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "split_json": str(args.split_json.resolve()),
        "threshold": args.threshold,
        "samples": len(events),
        "overall": detailed_metrics(overall),
        "per_region": {
            region: {"samples": samples[region], **detailed_metrics(regions[region])}
            for region in sorted(regions)
        },
    }
    print_report(report)
    output = args.output or PACKAGE_ROOT / "outputs" / "reports" / f"{args.model}_{args.split}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
