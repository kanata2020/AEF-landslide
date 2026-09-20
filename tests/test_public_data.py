import hashlib
import json
import tempfile
import unittest
from pathlib import Path, PureWindowsPath

from landslide_benchmark.paths import PACKAGE_ROOT, DEFAULT_SPLIT
from landslide_benchmark.data import load_split, resolve_s2_path
from landslide_benchmark.cross_region import collect_events, make_folds


class PublicDataTests(unittest.TestCase):
    def test_bundled_splits_and_provenance(self):
        directory = PACKAGE_ROOT / "splits"
        original = json.loads((directory / "dataset_split_resolved.json").read_text())
        current = json.loads(DEFAULT_SPLIT.read_text())
        self.assertEqual([len(original[k]) for k in ("train", "val")], [2095, 898])
        self.assertEqual([len(current[k]) for k in ("train", "val")], [1687, 743])
        for part in ("train", "val"):
            self.assertEqual(current[part], [e for e in original[part] if e["region"] != "italy"])
            for e in original[part]:
                self.assertEqual(e["s2"], PureWindowsPath(e["s2"]).name)
                self.assertFalse(PureWindowsPath(e["s2"]).is_absolute())
        train = {(e["region"], e["event_id"]) for e in current["train"]}
        val = {(e["region"], e["event_id"]) for e in current["val"]}
        self.assertFalse(train & val)
        provenance = json.loads((directory / "provenance.json").read_text())
        for name, info in provenance["files"].items():
            self.assertEqual(info["sha256"], hashlib.sha256((directory / name).read_bytes()).hexdigest())

    def test_relocated_data_and_both_loaders(self):
        with tempfile.TemporaryDirectory(prefix="public data ") as tmp:
            directory = Path(tmp)
            aef, s2 = directory / "aef", directory / "s2"
            aef.mkdir()
            s2.mkdir()
            raw = {"train": [], "val": []}
            for region in ("chimanimani", "hiroshima", "hokkaido", "dominicamaria"):
                for index, part in enumerate(("train", "val")):
                    filename = f"{region}_s2_{index}.nc"
                    (s2 / filename).touch()
                    (aef / f"{region}_AEF_{index}.tif").touch()
                    raw[part].append({"region": region, "event_id": str(index), "s2": filename})
            split = directory / "split.json"
            split.write_text(json.dumps(raw))
            loaded = load_split(split, aef, s2)
            for part in loaded.values():
                for event in part:
                    self.assertEqual(Path(event["S2"]).parent, s2)
            grouped = collect_events(split, aef, s2)
            self.assertEqual(len(make_folds(grouped)), 4)
            legacy = {**raw["train"][0], "s2": r"Z:\old-machine\chimanimani_s2_0.nc"}
            self.assertEqual(resolve_s2_path(legacy, s2), s2 / "chimanimani_s2_0.nc")


if __name__ == "__main__":
    unittest.main()
