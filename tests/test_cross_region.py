import json
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from landslide_benchmark.cross_region import (
    REGIONS, MODELS, build_parser, digest, fit_optical_norm, fixed_json,
    make_folds, read_json, summarize, train_fold, write_json, OpticalTestDataset,
    check_training_config, compatible_training_config, record_training_invocation,
)
from landslide_benchmark.spatiotemporal_data import split_fingerprint


class CrossRegionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.grouped = {
            region: [{"region": region, "event_id": str(i), "AEF": f"{region}_{i}.tif",
                      "S2": f"{region}_{i}.nc"} for i in range(10 + index)]
            for index, region in enumerate(REGIONS)
        }

    def test_disjoint_complete_reproducible_folds(self):
        folds = make_folds(self.grouped)
        self.assertEqual(folds, make_folds({r: list(reversed(es)) for r, es in self.grouped.items()}))
        all_keys = {(r, e["event_id"]) for r, es in self.grouped.items() for e in es}
        for region, fold in folds.items():
            keys = [{(e["region"], e["event_id"]) for e in fold[k]} for k in ("train", "val", "test")]
            self.assertFalse(keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
            self.assertEqual(set.union(*keys), all_keys)
            self.assertEqual({e["region"] for e in fold["test"]}, {region})
            for name in ("train", "val"):
                self.assertEqual({e["region"] for e in fold[name]}, set(REGIONS) - {region})

    def test_duplicate_and_small_source_rejected(self):
        self.grouped[REGIONS[0]].append(self.grouped[REGIONS[0]][0])
        with self.assertRaises(ValueError):
            make_folds(self.grouped)
        with self.assertRaises(ValueError):
            make_folds(self.grouped, max_events_per_region=1)

    def test_immutable_configuration(self):
        path = self.root / "config.json"
        fixed_json(path, {"seed": 42})
        fixed_json(path, {"seed": 42})
        with self.assertRaises(ValueError):
            fixed_json(path, {"seed": 43})

    def test_resource_changes_preserve_original_and_record_history(self):
        original = {"manifest_digest": "fixed", "commands": [["train_optical.py",
                    ["--batch-size", "8", "--num-workers", "8", "--epochs", "80"]]]}
        changed = copy.deepcopy(original)
        changed["commands"][0][1][1] = "4"
        changed["commands"][0][1][3] = "0"
        path = self.root / "run_config.json"
        check_training_config(path, original)
        check_training_config(path, changed)
        record_training_invocation(self.root, changed)
        record_training_invocation(self.root, changed)
        self.assertEqual(read_json(path), original)
        history = read_json(self.root / "runtime_config_history.json")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["config"], original)
        self.assertEqual(history[1]["config"], changed)
        self.assertTrue(compatible_training_config(original, changed))
        for prohibited in ("epochs", "manifest"):
            invalid = copy.deepcopy(changed)
            if prohibited == "epochs":
                invalid["commands"][0][1][-1] = "100"
            else:
                invalid["manifest_digest"] = "different"
            with self.assertRaises(ValueError):
                check_training_config(path, invalid)

    def test_norm_only_uses_supplied_training_events(self):
        events = []
        for index, value in enumerate((1.0, 3.0, 10000.0)):
            path = self.root / f"{index}.nc"
            xr.Dataset({"B02": (("time", "y", "x"), np.full((2, 2, 2), value)),
                        "DEM": (("y", "x"), np.full((2, 2), value * 2)),
                        "MASK": (("y", "x"), np.zeros((2, 2)))}).to_netcdf(path)
            events.append({"region": "source" if index < 2 else "target", "event_id": str(index), "S2": str(path)})
        path = self.root / "norm.json"
        fit_optical_norm(events[:2], path)
        result = read_json(path)
        self.assertEqual(result["s2"]["mean"], {"B02": 2.0, "DEM": 4.0})
        self.assertEqual(result["s2"]["std"], {"B02": 1.0, "DEM": 2.0})
        with self.assertRaises(ValueError):
            fit_optical_norm(events, path)

    def test_source_pretraining_and_checkpoint_selection(self):
        args = build_parser().parse_args([])
        args.output_dir = self.root
        manifest = {"folds": make_folds(self.grouped)}
        def finish_pretraining(name, command):
            if "--no-spatial-init" in command:
                path = self.root / REGIONS[0] / "source_pretrain" / "best_spatial.pth"
                path.parent.mkdir(parents=True)
                path.touch()
        with patch("landslide_benchmark.cross_region.run_script", side_effect=finish_pretraining) as run:
            train_fold(args, manifest, REGIONS[0], "aef")
        self.assertEqual(run.call_count, 2)
        pretraining, joint = [call.args[1] for call in run.call_args_list]
        self.assertIn("--no-spatial-init", pretraining)
        self.assertEqual(pretraining[pretraining.index("--temporal-weight") + 1], 0)
        self.assertEqual(joint[joint.index("--spatial-init") + 1],
                         self.root / REGIONS[0] / "source_pretrain" / "best_spatial.pth")
        self.assertNotIn("target_test.json", " ".join(map(str, pretraining + joint)))

    def test_resume_skips_completed_pretraining(self):
        args = build_parser().parse_args([])
        args.output_dir = self.root
        manifest = {"folds": make_folds(self.grouped)}
        def interrupted_joint(name, command):
            if "--no-spatial-init" in command:
                path = self.root / REGIONS[0] / "source_pretrain" / "best_spatial.pth"
                path.parent.mkdir(parents=True)
                path.touch()
            else:
                raise RuntimeError("interrupted joint training")
        with patch("landslide_benchmark.cross_region.run_script", side_effect=interrupted_joint):
            with self.assertRaises(RuntimeError):
                train_fold(args, manifest, REGIONS[0], "aef")
        args.aef_batch_size = 8
        args.num_workers = 4
        with patch("landslide_benchmark.cross_region.run_script") as run:
            train_fold(args, manifest, REGIONS[0], "aef")
        self.assertEqual(run.call_count, 1)
        self.assertIn("--spatial-init", run.call_args.args[1])
        completed = read_json(self.root / REGIONS[0] / "aef" / "training_complete.json")
        original = read_json(self.root / REGIONS[0] / "aef" / "run_config.json")
        self.assertTrue(compatible_training_config(original, completed))
        args.aef_batch_size = 4
        with patch("landslide_benchmark.cross_region.run_script") as run:
            train_fold(args, manifest, REGIONS[0], "aef")
        run.assert_not_called()

    def test_optical_image_and_reference_share_storage_orientation(self):
        mask = np.zeros((128, 128), dtype=np.float32)
        mask[2, 7] = 1
        path = self.root / "event.nc"
        xr.Dataset({"B02": (("time", "x", "y"), mask[None]),
                    "MASK": (("x", "y"), mask)}).to_netcdf(path)
        norm = self.root / "norm.json"
        write_json(norm, {"s2": {"mean": {"B02": 0}, "std": {"B02": 1}}})
        sample = OpticalTestDataset([{"S2": str(path)}], norm)[0]
        self.assertEqual(sample["mask"][0, 2, 7].item(), 1)
        self.assertEqual(sample["image"][0, 0, 2, 7].item(), 1)
        self.assertEqual(sample["image"][0, 0, 7, 2].item(), 0)

    def test_failed_training_never_marks_complete(self):
        args = build_parser().parse_args([])
        args.output_dir = self.root
        with patch("landslide_benchmark.cross_region.run_script", side_effect=RuntimeError("interrupted")):
            with self.assertRaises(RuntimeError):
                train_fold(args, {"folds": make_folds(self.grouped)}, REGIONS[0], "aef")
        self.assertFalse((self.root / REGIONS[0] / "aef" / "training_complete.json").exists())

    def test_equal_region_mean_and_missing_fold_rejected(self):
        manifest = {"folds": make_folds(self.grouped), "smoke_test": False}
        for index, region in enumerate(REGIONS):
            for model in MODELS:
                report = {"manifest_digest": digest(manifest), "held_out_region": region, "model": model,
                          "test_fingerprint": split_fingerprint(manifest["folds"][region]["test"]),
                          "segmentation": {"landslide": {k: index / 4 for k in ("precision", "recall", "f1", "iou")}},
                          "temporal": {"accuracy": index / 4} if model == "aef" else None}
                write_json(self.root / region / model / "test_metrics.json", report)
        summarize(self.root, manifest)
        rows = read_json(self.root / "summary.json")["rows"]
        self.assertEqual(rows[-1]["f1"], 0.375)
        self.assertEqual(rows[-1]["month_accuracy"], 0.375)
        self.assertIsNone(rows[-2]["month_accuracy"])
        (self.root / REGIONS[0] / "aef" / "test_metrics.json").unlink()
        with self.assertRaises(ValueError):
            summarize(self.root, manifest)


if __name__ == "__main__":
    unittest.main()
