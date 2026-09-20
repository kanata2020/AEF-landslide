import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset


def event_key(event: Dict) -> tuple[str, str]:
    return str(event["region"]), str(event["event_id"])


def resolve_s2_path(item: Dict, s2_dir: Path) -> Path:
    value = item.get("S2", item.get("s2"))
    if not value:
        raise KeyError(f"Missing S2 path for {event_key(item)}")
    path = Path(value)
    if path.is_absolute() and path.is_file():
        return path
    if not path.is_absolute():
        candidate = s2_dir / path
        if candidate.is_file():
            return candidate
    region, identifier = event_key(item)
    return s2_dir / f"{region}_s2_{identifier}.nc"


def load_split(split_path: Path, aef_dir: Path, s2_dir: Optional[Path] = None) -> Dict[str, List[Dict]]:
    with open(split_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if "train" not in raw or "val" not in raw:
        raise ValueError("Split JSON must contain 'train' and 'val' lists.")

    result = {}
    for split_name in ("train", "val"):
        events = []
        for item in raw[split_name]:
            region, event_id = event_key(item)
            s2_path = resolve_s2_path(item, s2_dir if s2_dir is not None else split_path.parent)
            events.append(
                {
                    "region": region,
                    "event_id": event_id,
                    "AEF": str(aef_dir / f"{region}_AEF_{event_id}.tif"),
                    "S2": str(s2_path),
                }
            )
        result[split_name] = events
    validate_split(result)
    return result


def validate_split(splits: Dict[str, List[Dict]], check_files: bool = True) -> None:
    keys = {name: [event_key(event) for event in events] for name, events in splits.items()}
    for name, values in keys.items():
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate events in '{name}' split.")
    overlap = set(keys["train"]) & set(keys["val"])
    if overlap:
        raise ValueError(f"Train/val leakage detected: {sorted(overlap)[:5]}")
    if check_files:
        missing = [
            f"{name} {event_key(event)} {field}: {event[field]}"
            for name, events in splits.items()
            for event in events
            for field in ("AEF", "S2")
            if not Path(event[field]).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} data files. First entries:\n" + "\n".join(missing[:20])
            )


class NormalizeS2:
    def __init__(self, path: Path):
        with open(path, "r", encoding="utf-8") as handle:
            stats = json.load(handle)["s2"]
        self.band_names = [name for name in stats["mean"] if name != "DEM"]
        self.means = stats["mean"]
        self.stds = stats["std"]

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        names = self.band_names + (["DEM"] if image.shape[1] > len(self.band_names) else [])
        means = torch.tensor([self.means[name] for name in names]).view(1, -1, 1, 1)
        stds = torch.tensor([self.stds[name] for name in names]).view(1, -1, 1, 1)
        return (image - means) / stds


class OpticalMaskDataset(Dataset):
    EXCLUDED = {"MASK", "DEM", "SCL", "spatial_ref"}

    def __init__(self, events: List[Dict], norm_path: Path, augment: bool = False, align_to_mask_storage: bool = False):
        self.events = events
        self.normalize = NormalizeS2(norm_path)
        self.augment = augment
        self.align_to_mask_storage = align_to_mask_storage

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int):
        event = self.events[index]
        with xr.open_dataset(event["S2"]) as dataset:
            if "time" in dataset.coords:
                dataset = dataset.sortby("time")
            image_data = dataset.drop_vars(self.EXCLUDED, errors="ignore")
            bands = [image_data[name] for name in image_data.data_vars]
            image = xr.concat(bands, dim="bands").transpose("time", "bands", "y", "x").values
            if "DEM" in dataset:
                dem = dataset["DEM"]
                if "time" in dem.dims:
                    dem = dem.isel(time=0)
                dem = dem.transpose("y", "x").values.astype(np.float32)
                dem = np.broadcast_to(dem[None, None], (image.shape[0], 1, *dem.shape))
                image = np.concatenate([image, dem], axis=1)
            mask = dataset["MASK"]
            if "time" in mask.dims:
                mask = mask.isel(time=0)
            transpose_to_storage = self.align_to_mask_storage and mask.dims == ("x", "y")
            mask = (mask.transpose("y", "x").values > 0).astype(np.int64)
        image = self.normalize(torch.from_numpy(image.astype(np.float32)))
        target = torch.from_numpy(mask)
        if transpose_to_storage:
            image, target = image.transpose(-2, -1), target.transpose(-2, -1)
        if self.augment:
            if torch.rand(1).item() < 0.5:
                image, target = image.flip(-1), target.flip(-1)
            if torch.rand(1).item() < 0.5:
                image, target = image.flip(-2), target.flip(-2)
            if torch.rand(1).item() < 0.5:
                k = int(torch.randint(1, 4, (1,)).item())
                image, target = torch.rot90(image, k, (-2, -1)), torch.rot90(target, k, (-2, -1))
        return image, target
