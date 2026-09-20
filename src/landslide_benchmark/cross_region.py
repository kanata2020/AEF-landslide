"""Four-fold leave-one-region-out experiment with source-only model selection."""

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from .data import OpticalMaskDataset, event_key, validate_split, resolve_s2_path
from .metrics import add_counts, confusion_counts, detailed_metrics, empty_counts
from .models import UNet3D
from .paths import DEFAULT_AEF_DIR, DEFAULT_S2_DIR, DEFAULT_SPLIT, PACKAGE_ROOT
from .spatiotemporal_data import (
    AEFSpatioTemporalDataset, TEMPORAL_CLASSES, _crop, _pad,
    read_mask_and_month, split_fingerprint,
)
from .spatiotemporal_model import AEFSpatioTemporalNet
from .spatiotemporal_training import temporal_metrics


REGIONS = ("chimanimani", "hiroshima", "hokkaido", "dominicamaria")
MODELS = ("optical", "aef")  # aef always denotes the joint model here.
LABELS = dict(zip(REGIONS, ("Chimanimani", "Hiroshima", "Hokkaido", "Dominica")))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def fixed_json(path, payload):
    """Refuse to mix an existing experiment with new data or training settings."""
    if path.exists():
        if read_json(path) != payload:
            raise ValueError(f"Experiment configuration changed: {path}. Use a new --output-dir.")
    else:
        write_json(path, payload)


def training_identity(config):
    """Resource settings may change on resume; all other settings must match."""
    commands = []
    for name, arguments in config["commands"]:
        retained = []
        index = 0
        while index < len(arguments):
            if arguments[index] in ("--batch-size", "--num-workers"):
                index += 2
            else:
                retained.append(arguments[index])
                index += 1
        commands.append([name, retained])
    return {**config, "commands": commands}


def compatible_training_config(left, right):
    return training_identity(left) == training_identity(right)


def check_training_config(path, config):
    if path.exists():
        if not compatible_training_config(read_json(path), config):
            raise ValueError(f"Experiment configuration changed: {path}. Use a new --output-dir.")
    else:
        write_json(path, config)


def record_training_invocation(output, config):
    """Keep the original run config and an audit trail of resource changes."""
    original = read_json(output / "run_config.json")
    path = output / "runtime_config_history.json"
    history = read_json(path) if path.exists() else [{"recorded_at": None, "config": original}]
    if history[-1]["config"] != config:
        history.append({"recorded_at": datetime.now(timezone.utc).isoformat(), "config": config})
        write_json(path, history)
        print("Resuming with updated batch size / workers; recorded in runtime_config_history.json.")


def collect_events(split_path, aef_dir, s2_dir):
    raw = read_json(split_path)
    grouped = {region: [] for region in REGIONS}
    seen = set()
    for partition in ("train", "val"):
        for item in raw[partition]:
            region, identifier = event_key(item)
            if region not in grouped:
                continue
            if (region, identifier) in seen:
                raise ValueError(f"Duplicate event in input inventory: {region}/{identifier}")
            seen.add((region, identifier))
            s2 = resolve_s2_path(item, s2_dir)
            grouped[region].append({
                "region": region, "event_id": identifier,
                "AEF": str((aef_dir / f"{region}_AEF_{identifier}.tif").resolve()),
                "S2": str(s2.resolve()),
            })
    for region, events in grouped.items():
        if len(events) < 2:
            raise ValueError(f"At least two events required for {region}; found {len(events)}.")
    return grouped


def make_folds(grouped, seed=42, val_fraction=0.3, max_events_per_region=None):
    if not 0 < val_fraction < 1:
        raise ValueError("--val-fraction must be between zero and one.")
    partitions = {}
    for region in REGIONS:
        events = sorted(grouped[region], key=event_key)
        random.Random(f"{seed}:{region}").shuffle(events)
        if max_events_per_region is not None:
            events = events[:max_events_per_region]
        if len(events) < 2:
            raise ValueError(f"At least two events required for {region}.")
        count = min(len(events) - 1, max(1, round(len(events) * val_fraction)))
        partitions[region] = {"train": events[count:], "val": events[:count]}
    folds = {}
    for target in REGIONS:
        train = [e for r in REGIONS if r != target for e in partitions[r]["train"]]
        val = [e for r in REGIONS if r != target for e in partitions[r]["val"]]
        test = partitions[target]["train"] + partitions[target]["val"]
        validate_split({"train": train, "val": val}, check_files=False)
        sets = [{event_key(e) for e in part} for part in (train, val, test)]
        if sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("Target-region leakage detected.")
        folds[target] = {"train": train, "val": val, "test": test}
    return folds


def prepare(args):
    grouped = collect_events(args.split_json, args.aef_dir, args.s2_dir)
    folds = make_folds(grouped, args.seed, args.val_fraction, args.max_events_per_region)
    manifest = {
        "protocol": "leave-one-region-out-v1", "regions": list(REGIONS),
        "seed": args.seed, "val_fraction": args.val_fraction,
        "smoke_test": args.max_events_per_region is not None,
        "max_events_per_region": args.max_events_per_region,
        "folds": folds,
    }
    # File existence checks use only selected regions; Italy is outside this experiment.
    selected = [e for fold in folds.values() for e in fold["test"]]
    validate_split({"train": selected, "val": []})
    fixed_json(args.output_dir / "manifest.json", manifest)
    for region, fold in folds.items():
        directory = args.output_dir / region
        fixed_json(directory / "source_split.json", {k: fold[k] for k in ("train", "val")})
        fixed_json(directory / "target_test.json", {"test": fold["test"]})
        print(f"{region}: source train={len(fold['train'])}, source val={len(fold['val'])}, test={len(fold['test'])}")
    return manifest


def fit_optical_norm(events, path):
    """Streaming population moments from source-training observations only."""
    fingerprint = split_fingerprint(events)
    if path.exists():
        cached = read_json(path)
        if cached.get("split_fingerprint") != fingerprint:
            raise ValueError(f"Normalization belongs to another training set: {path}")
        return
    moments = {}
    expected = None
    for index, event in enumerate(events, 1):
        with xr.open_dataset(event["S2"]) as dataset:
            bands = [name for name in dataset.data_vars if name not in OpticalMaskDataset.EXCLUDED]
            names = bands + (["DEM"] if "DEM" in dataset else [])
            if expected is None:
                expected = names
            if names != expected:
                raise ValueError(f"Inconsistent optical band order: {event['S2']}")
            for name in names:
                values = dataset[name]
                if name == "DEM" and "time" in values.dims:
                    values = values.isel(time=0)
                values = np.asarray(values.values, dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite optical values in {event['S2']} ({name}).")
                count = values.size
                mean = float(values.mean())
                m2 = float(np.square(values - mean).sum())
                # DEM is repeated once per acquisition in OpticalMaskDataset.
                if name == "DEM":
                    repeats = dataset.sizes.get("time", 1)
                    count, m2 = count * repeats, m2 * repeats
                old_n, old_mean, old_m2 = moments.get(name, (0, 0.0, 0.0))
                delta, total = mean - old_mean, old_n + count
                moments[name] = (total, old_mean + delta * count / total,
                                 old_m2 + m2 + delta * delta * old_n * count / total)
        if index % 100 == 0 or index == len(events):
            print(f"Source-only optical statistics: {index}/{len(events)}", flush=True)
    write_json(path, {
        "split_fingerprint": fingerprint,
        "s2": {
            "mean": {name: moments[name][1] for name in expected},
            "std": {name: max(float(np.sqrt(moments[name][2] / moments[name][0])), 1e-6) for name in expected},
        },
    })


def run_script(name, arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PACKAGE_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    command = [sys.executable, str(PACKAGE_ROOT / "scripts" / name), *map(str, arguments)]
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def train_fold(args, manifest, region, model):
    directory = args.output_dir / region
    fold = manifest["folds"][region]
    output = directory / model
    common = ["--split-json", directory / "source_split.json", "--aef-dir", args.aef_dir,
              "--s2-dir", args.s2_dir, "--seed", args.seed, "--device", args.device,
              "--num-workers", args.num_workers]
    if model == "optical":
        fit_optical_norm(fold["train"], directory / "optical_norm.json")
        commands = [("train_optical.py", [*common, "--norm-json", directory / "optical_norm.json",
                    "--output-dir", output, "--epochs", args.optical_epochs,
                    "--align-to-mask-storage",
                    "--batch-size", args.optical_batch_size, "--patience", args.patience,
                    *(["--amp"] if args.amp else [])])]
        if len(fold["train"]) < args.optical_batch_size:
            raise ValueError("Optical training has fewer events than --optical-batch-size; reduce it.")
    else:
        joint = [*common, "--batch-size", args.aef_batch_size, "--patience", args.patience,
                 "--amp" if args.amp else "--no-amp"]
        pretrain = directory / "source_pretrain"
        # Use the joint architecture for source-only spatial warm-up. No standalone model.
        commands = [
            ("train_spatiotemporal.py", [*joint, "--output-dir", pretrain,
             "--epochs", args.pretrain_epochs, "--no-spatial-init", "--temporal-weight", 0]),
            ("train_spatiotemporal.py", [*joint, "--output-dir", output,
             "--epochs", args.aef_epochs, "--spatial-init", pretrain / "best_spatial.pth"]),
        ]
    config = {"manifest_digest": digest(manifest),
              "commands": [[name, list(map(str, command))] for name, command in commands]}
    check_training_config(output / "run_config.json", config)
    if (output / "training_complete.json").exists():
        if not compatible_training_config(read_json(output / "training_complete.json"), config):
            raise ValueError(f"Invalid training completion record: {output}")
        print(f"Already trained: {region}/{model}")
        return
    record_training_invocation(output, config)
    for index, (name, command) in enumerate(commands):
        stage_marker = output / f"stage_{index}_complete.json"
        if stage_marker.exists():
            if not compatible_training_config(read_json(stage_marker), config):
                raise ValueError(f"Invalid stage completion record: {stage_marker}")
            continue
        if "--spatial-init" in command:
            initialization = Path(command[command.index("--spatial-init") + 1])
            if not initialization.is_file():
                raise FileNotFoundError(f"Source-only spatial initialization is missing: {initialization}")
        run_script(name, command)
        write_json(stage_marker, config)
    write_json(output / "training_complete.json", config)


class OpticalTestDataset(Dataset):
    def __init__(self, events, norm):
        self.events = events
        self.optical = OpticalMaskDataset(events, norm, align_to_mask_storage=True)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        image, optical_mask = self.optical[index]
        reference, month = read_mask_and_month(self.events[index]["S2"])
        reference = torch.from_numpy(_pad(_crop(reference, 128), 128))
        if not torch.equal(optical_mask.bool(), reference[0].bool()):
            raise ValueError("Optical and AEF reference masks differ in shape or orientation.")
        return {"image": image, "mask": reference, "month": month}


def evaluate_fold(args, manifest, region, model_name):
    directory = args.output_dir / region
    output = directory / model_name
    config = read_json(output / "run_config.json")
    if (config["manifest_digest"] != digest(manifest)
            or not compatible_training_config(read_json(output / "training_complete.json"), config)):
        raise ValueError("Training is incomplete or belongs to another experiment.")
    filename = "best_spatial.pth" if model_name == "aef" else "best.pth"
    path = output / filename
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    fold = manifest["folds"][region]
    if model_name == "aef":
        if checkpoint["training_statistics"]["split_fingerprint"] != split_fingerprint(fold["train"]):
            raise ValueError("AEF checkpoint was trained on another split.")
        network = AEFSpatioTemporalNet(**checkpoint["model_config"])
        dataset = AEFSpatioTemporalDataset(fold["test"])
        batch_size = args.aef_batch_size
    else:
        norm = directory / "optical_norm.json"
        if read_json(norm)["split_fingerprint"] != split_fingerprint(fold["train"]):
            raise ValueError("Optical normalization was fitted on another split.")
        network = UNet3D(**checkpoint["model_cfg"])
        dataset = OpticalTestDataset(fold["test"], norm)
        batch_size = args.eval_optical_batch_size
    train_config = checkpoint["train_config" if model_name == "aef" else "train_cfg"]
    if Path(train_config["split_json"]).resolve() != (directory / "source_split.json").resolve():
        raise ValueError("Checkpoint selection used another source split.")
    network.load_state_dict(checkpoint["model_state_dict"])
    network.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    counts = empty_counts()
    months = np.zeros((TEMPORAL_CLASSES, TEMPORAL_CLASSES), dtype=np.int64)
    with torch.inference_mode():
        for batch in loader:
            predictions = network(batch["image"].to(device))
            key = "segmentation_logits" if model_name == "aef" else "segmentation"
            mask = torch.sigmoid(predictions[key]).cpu() >= 0.5
            if mask.shape != batch["mask"].shape:
                raise ValueError("Prediction and reference mask shapes differ.")
            add_counts(counts, confusion_counts(mask, batch["mask"]))
            if model_name == "aef":
                predicted = predictions["temporal_logits"].argmax(1).cpu().numpy()
                np.add.at(months, (batch["month"].numpy(), predicted), 1)
    report = {
        "manifest_digest": digest(manifest), "held_out_region": region, "model": model_name,
        "checkpoint": str(path), "checkpoint_epoch": checkpoint["epoch"],
        "samples": len(dataset), "threshold": 0.5, "split": "target_test",
        "test_fingerprint": split_fingerprint(fold["test"]),
        "segmentation": detailed_metrics(counts),
        "temporal": temporal_metrics(months) if model_name == "aef" else None,
    }
    write_json(output / "test_metrics.json", report)
    print(f"Test {region}/{model_name}: {report['segmentation']['landslide']}", flush=True)


def summarize(output_dir, manifest, require_complete=True):
    rows, missing = [], []
    for region in REGIONS:
        for model in MODELS:
            path = output_dir / region / model / "test_metrics.json"
            if not path.exists():
                missing.append(f"{region}/{model}")
                continue
            report = read_json(path)
            if (report["manifest_digest"] != digest(manifest)
                    or report["held_out_region"] != region or report["model"] != model
                    or report["test_fingerprint"] != split_fingerprint(manifest["folds"][region]["test"])):
                raise ValueError(f"Report does not match this experiment: {path}")
            rows.append({"region": LABELS[region], "model": "AEF-based" if model == "aef" else "3D-UNet",
                         **report["segmentation"]["landslide"],
                         "month_accuracy": report["temporal"]["accuracy"] if model == "aef" else None})
    if missing:
        message = "No four-fold mean yet. Missing: " + ", ".join(missing)
        if require_complete:
            raise ValueError(message)
        print(message)
        return
    for model in ("3D-UNet", "AEF-based"):
        selected = [row for row in rows if row["model"] == model]
        rows.append({"region": "Mean", "model": model,
                     **{metric: sum(row[metric] for row in selected) / len(REGIONS)
                        for metric in ("precision", "recall", "f1", "iou")},
                     "month_accuracy": sum(row["month_accuracy"] for row in selected) / len(REGIONS)
                     if model == "AEF-based" else None})
    write_json(output_dir / "summary.json", {"smoke_test": manifest["smoke_test"], "rows": rows,
               "averaging": "unweighted mean of four regional metrics; month accuracy includes no-event"})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    latex = []
    for row in rows:
        metrics = ["--" if row[key] is None else f"{row[key]:.4f}"
                   for key in ("precision", "recall", "f1", "iou", "month_accuracy")]
        latex.append(" & ".join([row["region"], row["model"], *metrics]) + r" \\")
    (output_dir / "table_rows.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    print(f"Four-fold results saved: {output_dir / 'summary.csv'}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "evaluate", "summarize", "all"), default="all")
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--aef-dir", type=Path, default=DEFAULT_AEF_DIR)
    parser.add_argument("--s2-dir", type=Path, default=DEFAULT_S2_DIR)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "outputs" / "cross_region")
    parser.add_argument("--folds", nargs="+", choices=REGIONS, default=list(REGIONS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.3)
    parser.add_argument("--pretrain-epochs", type=int, default=300)
    parser.add_argument("--aef-epochs", type=int, default=300)
    parser.add_argument("--optical-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--aef-batch-size", type=int, default=16)
    parser.add_argument("--optical-batch-size", type=int, default=12)
    parser.add_argument("--eval-optical-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-events-per-region", type=int, help="Smoke test only; use a separate output directory.")
    return parser


def main():
    args = build_parser().parse_args()
    for name in ("pretrain_epochs", "aef_epochs", "optical_epochs", "patience", "aef_batch_size",
                 "optical_batch_size", "eval_optical_batch_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive.")
    args.output_dir = args.output_dir.resolve()
    args.aef_dir, args.s2_dir = args.aef_dir.resolve(), args.s2_dir.resolve()
    if args.stage in ("prepare", "all"):
        manifest = prepare(args)
    else:
        manifest = read_json(args.output_dir / "manifest.json")
        if (args.seed != manifest["seed"] or args.val_fraction != manifest["val_fraction"]
                or args.max_events_per_region != manifest["max_events_per_region"]):
            raise ValueError("Use the seed, val-fraction and smoke-test limit recorded in manifest.json.")
    # Verify immutable files before allowing training or target evaluation.
    for region, fold in manifest["folds"].items():
        if read_json(args.output_dir / region / "source_split.json") != {k: fold[k] for k in ("train", "val")}:
            raise ValueError(f"Source split was modified: {region}")
        if read_json(args.output_dir / region / "target_test.json") != {"test": fold["test"]}:
            raise ValueError(f"Target test split was modified: {region}")
    for region in args.folds:
        for model in args.models:
            if args.stage in ("train", "all"):
                train_fold(args, manifest, region, model)
            if args.stage in ("evaluate", "all"):
                evaluate_fold(args, manifest, region, model)
    if args.stage in ("summarize", "all"):
        summarize(args.output_dir, manifest, require_complete=args.stage == "summarize")
