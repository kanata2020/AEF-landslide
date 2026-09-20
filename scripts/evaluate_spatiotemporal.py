import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from landslide_benchmark.data import load_split
from landslide_benchmark.metrics import add_counts, confusion_counts, detailed_metrics, empty_counts
from landslide_benchmark.paths import (
    DEFAULT_AEF_DIR,
    DEFAULT_S2_DIR,
    DEFAULT_SPATIOTEMPORAL_OUTPUT,
    DEFAULT_SPLIT,
    PACKAGE_ROOT,
)
from landslide_benchmark.spatiotemporal_data import (
    AEFSpatioTemporalDataset,
    MONTH_NAMES,
    NO_EVENT_CLASS_INDEX,
    TEMPORAL_CLASSES,
    split_fingerprint,
)
from landslide_benchmark.spatiotemporal_model import AEFSpatioTemporalNet
from landslide_benchmark.spatiotemporal_training import temporal_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate both outputs of the joint AEF model.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_SPATIOTEMPORAL_OUTPUT / "best.pth")
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--aef-dir", type=Path, default=DEFAULT_AEF_DIR)
    parser.add_argument("--s2-dir", type=Path, default=DEFAULT_S2_DIR)
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "outputs" / "reports" / "spatiotemporal_val_metrics.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--limit", type=int, help="Smoke-test only.")
    parser.add_argument("--limit-train", type=int, help="Match a checkpoint trained with --limit-train.")
    return parser.parse_args()


def month_report(confusion):
    base = temporal_metrics(confusion)
    support = confusion.sum(1)
    predicted = confusion.sum(0)
    rows = []
    for index, name in enumerate(MONTH_NAMES):
        tp = int(confusion[index, index])
        precision = tp / predicted[index] if predicted[index] else 0.0
        recall = tp / support[index] if support[index] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"month": name, "support": int(support[index]), "precision": precision, "recall": recall, "f1": f1})
    base["class_wise"] = rows
    return base


def metric_row(name, values, samples=None):
    prefix = f"{name:<18}" + (f" {samples:>7d}" if samples is not None else "")
    return prefix + "".join(f" {values[key]:>10.6f}" for key in ("precision", "recall", "f1", "iou"))


def print_spatial_report(report):
    overall = report["segmentation"]
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
        print(metric_row(region, values["segmentation"]["landslide"], values["samples"]))


def print_temporal_report(temporal):
    print("\n5. Temporal Classification Metrics")
    print(f"Accuracy:                 {temporal['accuracy']:.6f}")
    print(f"Balanced Accuracy:        {temporal['balanced_accuracy']:.6f}")
    print(f"Macro F1 (observed):      {temporal['macro_f1_observed']:.6f}")
    print(f"Top-3 Accuracy:           {temporal['top3_accuracy']:.6f}")
    print(f"Circular MAE (months):    {temporal['circular_mae_months']:.6f}")
    print(f"Global-mode Baseline:     {temporal['global_majority_baseline_accuracy']:.6f}")
    print(f"Region-mode Baseline:     {temporal['region_mode_baseline_accuracy']:.6f}")


def main():
    args = parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    statistics = checkpoint["training_statistics"]
    splits = load_split(args.split_json, args.aef_dir, args.s2_dir)
    fingerprint_events = splits["train"][: args.limit_train] if args.limit_train else splits["train"]
    if split_fingerprint(fingerprint_events) != statistics["split_fingerprint"]:
        raise ValueError("Checkpoint training statistics belong to a different training split.")
    events = splits["val"][: args.limit] if args.limit else splits["val"]
    dataset = AEFSpatioTemporalDataset(events)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = AEFSpatioTemporalNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    overall_spatial = empty_counts()
    region_spatial = defaultdict(empty_counts)
    region_samples = defaultdict(int)
    overall_month = np.zeros((TEMPORAL_CLASSES, TEMPORAL_CLASSES), dtype=np.int64)
    region_month = defaultdict(lambda: np.zeros((TEMPORAL_CLASSES, TEMPORAL_CLASSES), dtype=np.int64))
    top3_correct = circular_distance = temporal_samples = circular_samples = 0
    no_event_samples_with_positive_mask = 0
    baseline_global_correct = baseline_region_correct = 0
    global_mode = int(np.argmax(statistics["month_counts"]))
    region_modes = {
        region: int(np.argmax(counts)) for region, counts in statistics.get("region_month_counts", {}).items()
    }

    with torch.inference_mode():
        for batch in tqdm(loader, desc="evaluate"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            months = batch["month"].to(device, non_blocking=True)
            outputs = model(images)
            predictions = torch.sigmoid(outputs["segmentation_logits"]) >= args.threshold
            add_counts(overall_spatial, confusion_counts(predictions, masks))
            month_predictions = outputs["temporal_logits"].argmax(1)
            top3 = outputs["temporal_logits"].topk(3, dim=1).indices

            for index, region in enumerate(batch["region"]):
                add_counts(region_spatial[region], confusion_counts(predictions[index], masks[index]))
                region_samples[region] += 1
                truth = int(months[index])
                prediction = int(month_predictions[index])
                overall_month[truth, prediction] += 1
                region_month[region][truth, prediction] += 1
                top3_correct += int((top3[index] == truth).any())
                temporal_samples += 1
                if truth < NO_EVENT_CLASS_INDEX and prediction < NO_EVENT_CLASS_INDEX:
                    difference = abs(truth - prediction)
                    circular_distance += min(difference, 12 - difference)
                    circular_samples += 1
                if truth == NO_EVENT_CLASS_INDEX and bool(masks[index].any()):
                    no_event_samples_with_positive_mask += 1
                baseline_global_correct += int(global_mode == truth)
                baseline_region_correct += int(region_modes.get(region, global_mode) == truth)

    temporal = month_report(overall_month)
    temporal.update(
        top3_accuracy=top3_correct / max(temporal_samples, 1),
        circular_mae_months=circular_distance / max(circular_samples, 1),
        circular_mae_samples=circular_samples,
        global_majority_baseline_accuracy=baseline_global_correct / max(temporal_samples, 1),
        region_mode_baseline_accuracy=baseline_region_correct / max(temporal_samples, 1),
        no_event_samples_with_positive_mask=no_event_samples_with_positive_mask,
    )
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": "val",
        "events": len(events),
        "threshold": args.threshold,
        "segmentation": detailed_metrics(overall_spatial),
        "temporal": temporal,
        "per_region": {
            region: {
                "samples": region_samples[region],
                "segmentation": detailed_metrics(region_spatial[region]),
                "temporal": month_report(region_month[region]),
            }
            for region in sorted(region_spatial)
        },
        "methodological_note": (
            "Missing or invalid dates are assigned to class 13 (no-event). Calendar-month coverage depends on the "
            "selected split, and month may be confounded with region; compare accuracy with the region-mode baseline "
            "and inspect no_event_samples_with_positive_mask for label conflicts."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Checkpoint: {args.checkpoint} (epoch {checkpoint['epoch']})")
    print(f"Events: {len(events)} | temporal samples: {temporal_samples}")
    print_spatial_report(report)
    print_temporal_report(temporal)
    print(f"No-event samples with a positive mask: {no_event_samples_with_positive_mask}")
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
