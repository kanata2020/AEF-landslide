import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio
import torch
import xarray as xr
from torch.utils.data import Dataset

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "No event",
]
NO_EVENT_CLASS_INDEX = 12
TEMPORAL_CLASSES = 13


def parse_event_month(value) -> int:
    """Return a zero-based month, or class 12 when the date is absent/invalid."""
    if value is None:
        return NO_EVENT_CLASS_INDEX
    raw = str(value).strip()
    if not raw or raw.lower() in {"none", "nan", "nat"}:
        return NO_EVENT_CLASS_INDEX
    dates = []
    for token in raw.split(","):
        normalized = token.strip().replace("/", "-")
        if not normalized:
            continue
        try:
            dates.append(datetime.fromisoformat(normalized[:10]))
        except ValueError:
            continue
    return min(dates).month - 1 if dates else NO_EVENT_CLASS_INDEX


def read_mask_and_month(path: str):
    with xr.open_dataset(path) as dataset:
        mask = dataset["MASK"].values.astype(np.float32)
        month = parse_event_month(dataset.attrs.get("event_date"))
    if mask.ndim == 3:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"Unexpected MASK shape in {path}: {mask.shape}")
    return (mask > 0).astype(np.float32)[None], month


def _crop(array: np.ndarray, size: int) -> np.ndarray:
    _, height, width = array.shape
    if height > size:
        top = (height - size) // 2
        array = array[:, top : top + size, :]
    if width > size:
        left = (width - size) // 2
        array = array[:, :, left : left + size]
    return array


def _pad(array: np.ndarray, size: int) -> np.ndarray:
    _, height, width = array.shape
    pad_h, pad_w = max(0, size - height), max(0, size - width)
    if pad_h or pad_w:
        array = np.pad(
            array,
            ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
            mode="constant",
        )
    return array.astype(np.float32, copy=False)


def read_aef_raw(path: str, size: int = 128) -> np.ndarray:
    with rasterio.open(path) as source:
        image = source.read().astype(np.float32)
    image = np.rot90(image, k=-1, axes=(1, 2)).copy()
    return _crop(image, size)


def split_fingerprint(events: List[Dict]) -> str:
    keys = sorted(f"{event['region']}:{event['event_id']}" for event in events)
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def compute_training_statistics(events: List[Dict], size: int = 128) -> Dict:
    """Compute target statistics from training events only."""
    positive_pixels = total_pixels = 0
    month_counts = np.zeros(TEMPORAL_CLASSES, dtype=np.int64)
    region_month_counts: Dict[str, np.ndarray] = {}
    no_event_samples_with_positive_mask = 0

    for index, event in enumerate(events, start=1):
        mask, month = read_mask_and_month(event["S2"])
        mask = _crop(mask, size)
        positive_pixels += int((mask > 0).sum())
        total_pixels += int(mask.size)
        month_counts[month] += 1
        region = str(event["region"])
        region_month_counts.setdefault(region, np.zeros(TEMPORAL_CLASSES, dtype=np.int64))[month] += 1
        if month == NO_EVENT_CLASS_INDEX and np.any(mask > 0):
            no_event_samples_with_positive_mask += 1
        if index % 250 == 0 or index == len(events):
            print(f"Statistics: {index}/{len(events)} events", flush=True)

    if not events:
        raise ValueError("Cannot compute statistics from an empty training set.")

    observed = month_counts > 0
    temporal_weights = np.zeros(TEMPORAL_CLASSES, dtype=np.float64)
    if observed.any():
        temporal_weights[observed] = np.sqrt(month_counts[observed].max() / month_counts[observed])
        temporal_weights[observed] /= temporal_weights[observed].mean()

    return {
        "split_fingerprint": split_fingerprint(events),
        "schema_version": 2,
        "num_events": len(events),
        "aef_preprocessing": "raw_finite_values_nan_inf_to_zero",
        "positive_pixels": positive_pixels,
        "total_pixels": total_pixels,
        "positive_ratio": positive_pixels / max(total_pixels, 1),
        "month_counts": month_counts.tolist(),
        "region_month_counts": {name: counts.tolist() for name, counts in region_month_counts.items()},
        "temporal_class_weights": temporal_weights.tolist(),
        "no_event_samples": int(month_counts[NO_EVENT_CLASS_INDEX]),
        "no_event_samples_with_positive_mask": no_event_samples_with_positive_mask,
    }


def load_or_compute_statistics(path: Path, events: List[Dict], size: int = 128) -> Dict:
    fingerprint = split_fingerprint(events)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("split_fingerprint") == fingerprint and payload.get("schema_version") == 2:
            return payload
        print("Cached statistics use a different training split; recomputing.")
    payload = compute_training_statistics(events, size)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


class AEFSpatioTemporalDataset(Dataset):
    def __init__(
        self,
        events: List[Dict],
        size: int = 128,
        augment: bool = False,
    ):
        self.events = events
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> Dict:
        event = self.events[index]
        image = read_aef_raw(event["AEF"], self.size)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = _pad(image, self.size)

        mask, month = read_mask_and_month(event["S2"])
        mask = _pad(_crop(mask, self.size), self.size)
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)
        if self.augment:
            if torch.rand(()) < 0.5:
                image_tensor, mask_tensor = image_tensor.flip(-1), mask_tensor.flip(-1)
            if torch.rand(()) < 0.5:
                image_tensor, mask_tensor = image_tensor.flip(-2), mask_tensor.flip(-2)
            k = int(torch.randint(0, 4, ()).item())
            if k:
                image_tensor = torch.rot90(image_tensor, k, (-2, -1))
                mask_tensor = torch.rot90(mask_tensor, k, (-2, -1))
        return {
            "image": image_tensor.contiguous(),
            "mask": mask_tensor.contiguous(),
            "month": torch.tensor(month, dtype=torch.long),
            "region": str(event["region"]),
            "event_id": str(event["event_id"]),
        }
