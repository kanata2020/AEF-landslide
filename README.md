# AEF Landslide

[English](README.md) | [简体中文](README.zh-CN.md)

Research code for **Spatiotemporal Landslide Detection Using Multimodal AlphaEarth Foundation Model Embeddings**. The repository contains an AEF joint segmentation/occurrence-month model and a Sentinel-2/DEM 3D-UNet baseline, including four-fold leave-one-region-out evaluation.

## Installation

Use Python 3.10 or newer. Run commands from this repository's root (the directory containing `pyproject.toml`). A dedicated environment avoids importing another local copy of `landslide_benchmark`.

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` on PowerShell, or `source .venv/bin/activate` on Linux/macOS. Install a [PyTorch build](https://pytorch.org/get-started/locally/) appropriate for your CUDA environment, then install this repository:

```bash
python -m pip install -e .
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

All examples use single-line commands and forward slashes, which work in PowerShell and common Unix shells. Use `--device cpu` when CUDA is unavailable.

## Data and bundled splits

Obtain the original NetCDF samples from [Sen12Landslides](https://github.com/PaulH97/Sen12Landslides) and prepare the matching annual, 64-channel [AlphaEarth Foundations embeddings](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) as event GeoTIFFs. Imagery and trained weights are not included. This repository consumes prepared patches; it does not download or export embeddings.

Default layout:

```text
<repository>/
  splits/
    dataset_split_resolved.json       # original five-region membership
    dataset_split_four_regions.json   # default four-region membership
  data/
    aef/<region>_AEF_<event_id>.tif
    s2/<region>_s2_<event_id>.nc
    optical_norm.json                 # generated from training samples
  scripts/
  src/landslide_benchmark/
  tests/
  outputs/
```

AEF inputs must match the event year and sample extent. The existing loader rotates AEF arrays clockwise by 90 degrees, then center-crops/pads to 128 x 128 and replaces NaN/Inf with zero. Prepare GeoTIFFs in the expected orientation and check their alignment with the NetCDF masks. Optical samples contain the temporal Sentinel-2 bands and DEM; event-month targets come from the NetCDF `event_date` attribute.

Both split files are included in Git. `dataset_split_resolved.json` preserves all original event IDs, ordering, and train/validation assignments; only machine-specific absolute S2 paths were replaced with filenames resolved under `--s2-dir`. `dataset_split_four_regions.json` removes Italy without reshuffling any remaining event. It is the default for the current four-region paper.

| Region key | Original train | Original validation |
|---|---:|---:|
| chimanimani | 443 | 190 |
| hiroshima | 621 | 243 |
| hokkaido | 200 | 90 |
| dominicamaria | 423 | 220 |
| italy | 408 | 155 |
| Original total | 2095 | 898 |
| Default four-region total | 1687 | 743 |

Dominica is named `dominicamaria` in filenames and command-line options. For the historical five-region experiment, explicitly pass `--split-json splits/dataset_split_resolved.json` consistently to preparation, training, and evaluation, and use separate output/normalization files.

Data can remain outside this repository. Pass `--aef-dir /path/to/aef --s2-dir /path/to/s2` to the relevant commands. Paths containing spaces must be quoted. No original parent-project checkout is required. The imported inventory and its four-region derivative are documented in [splits/provenance.json](splits/provenance.json).

## Joint AEF model: fixed split

The model predicts a pixel mask and 13 temporal classes (January--December plus `no-event`). It uses a 96-channel shared stem, five full-resolution residual-SE blocks, and a temporal branch whose gradients into the stem are scaled by 0.1. The loss is `0.7 * BCE + 0.3 * Dice + 0.25 * temporal_loss`; temporal supervision uses training-only class weights and circular label smoothing.

For spatial pretraining followed by joint training, use the same training split in both stages:

```bash
python scripts/train_spatiotemporal.py --device cuda --no-spatial-init --temporal-weight 0 --output-dir outputs/spatial_pretrain
python scripts/train_spatiotemporal.py --device cuda --spatial-init outputs/spatial_pretrain/best_spatial.pth
```

Pretraining uses the joint architecture with zero temporal-loss weight; there is no separate standalone AEF model. The joint stage loads only the shared/spatial weights, freezes them for the first five epochs, and initializes the temporal branch separately. An explicitly supplied missing checkpoint raises an error. Omitting `--spatial-init` starts from random weights, with no initial spatial freeze.

Defaults: 300 epochs, batch size 16, spatial/shared learning rate `3e-5`, temporal learning rate `1e-4`, eight warm-up epochs, CUDA AMP, and early stopping after 60 epochs without validation spatial-IoU improvement. Run `--help` for all options.

```bash
python scripts/evaluate_spatiotemporal.py --device cuda --checkpoint outputs/spatiotemporal_v2/best_spatial.pth --batch-size 16 --num-workers 0
```

`best_spatial.pth` is selected by validation spatial IoU. `best.pth` is selected by `0.7 * IoU + 0.3 * temporal macro-F1` and is the evaluation script's default when `--checkpoint` is omitted. `last.pth` supports resuming from the next epoch; an interrupted partial epoch is rerun. Fixed-split metrics are written to `outputs/reports/spatiotemporal_val_metrics.json`. These are validation results, not a separate held-out-region test.

## Optical baseline: fixed split

Fit normalization only on the selected training partition before training:

```bash
python scripts/prepare_optical_norm.py
python scripts/train_optical.py --device cuda --amp --batch-size 4 --num-workers 0
python scripts/evaluate.py --model optical --device cuda --batch-size 2
```

Normalization defaults to `data/optical_norm.json`; use `--output` when preparing and `--norm-json` when training/evaluating to override it. The normalization cache rejects a different training-event fingerprint. Training defaults remain 80 epochs, batch size 12, learning rate `1e-3`, and validation-IoU checkpoint selection. The smaller batch size above is a memory-conscious example. Evaluation uses `outputs/optical/best.pth`, or the checkpoint supplied with `--checkpoint`; it never falls back to another project's weights.

## Four-fold cross-regional transferability

Each fold holds out one entire region for testing. Each of the other three regions is split approximately 70/30 into source training and source validation. The candidate inventory combines the input train/validation lists before constructing these new folds. Italy is excluded even when the original five-region inventory is supplied.

| Held-out region | Source train | Source validation | Target test |
|---|---:|---:|---:|
| Chimanimani | 1258 | 539 | 633 |
| Hiroshima | 1096 | 470 | 864 |
| Hokkaido | 1498 | 642 | 290 |
| Dominica | 1251 | 536 | 643 |

Both models use the same fold assignments. Normalization and temporal class weights are fitted only on source training samples. Each fold performs its own AEF spatial pretraining. Checkpoint selection and early stopping use source validation; target-region samples are used only for final inference, with threshold 0.5 and no fine-tuning. Optical images and masks are aligned to the joint model's mask storage order in this experiment, with a reference-mask consistency check.

Prepare splits, then run all four folds sequentially (not concurrently):

```bash
python scripts/cross_region.py --stage prepare
python scripts/cross_region.py --device cuda --aef-batch-size 8 --optical-batch-size 4 --eval-optical-batch-size 2 --num-workers 0
```

Defaults are 300 spatial-pretraining epochs, 300 joint-training epochs, and 80 optical epochs per fold, with patience 60. The command above explicitly reduces batch sizes; it does not change these epoch limits. Run one fold with `--folds chimanimani` (other choices: `hiroshima`, `hokkaido`, `dominicamaria`), or one model with `--models optical` / `--models aef`. Here `aef` always means the joint model.

To resume, rerun the command with the same output directory. Completed stages are skipped. Batch size and `--num-workers` may be changed; changes are recorded in `runtime_config_history.json`. Changing batch size changes subsequent optimization and should be included in experiment records. Other training settings and the data split must remain consistent, or use a new output directory.

If training was run separately with `--stage train`, evaluate and aggregate with:

```bash
python scripts/cross_region.py --stage evaluate --device cuda --aef-batch-size 8 --eval-optical-batch-size 2 --num-workers 0
python scripts/cross_region.py --stage summarize
```

Use the same custom `--output-dir`, data paths, seed, and validation fraction across stages. `--stage all` (the default) already trains, evaluates, and aggregates.

```text
outputs/cross_region/manifest.json
outputs/cross_region/<region>/source_split.json
outputs/cross_region/<region>/target_test.json
outputs/cross_region/<region>/optical_norm.json
outputs/cross_region/<region>/source_pretrain/best_spatial.pth
outputs/cross_region/<region>/optical/best.pth
outputs/cross_region/<region>/aef/best_spatial.pth
outputs/cross_region/<region>/<model>/test_metrics.json
outputs/cross_region/summary.json
outputs/cross_region/summary.csv
outputs/cross_region/table_rows.tex
```

Reports contain landslide-class Precision, Recall, F1, IoU, and AEF's 13-class month accuracy (including `no-event`). Summary values are fractions in [0, 1]. The final mean is an unweighted arithmetic mean across four regions, not a pooled pixel/sample score. All eight model/region reports are required. LaTeX rows are exported without editing the paper.

## Small-run checks and troubleshooting

Run one small fold to verify data and training connections; these outputs are not paper results:

```bash
python scripts/cross_region.py --device cuda --folds chimanimani --max-events-per-region 2 --pretrain-epochs 1 --aef-epochs 1 --optical-epochs 1 --aef-batch-size 2 --optical-batch-size 2 --eval-optical-batch-size 1 --num-workers 0 --output-dir outputs/cross_region_smoke
python -m unittest discover -s tests -v
```

For CUDA out-of-memory errors, reduce the relevant batch size. For Windows shared-memory error 1455, start with `--num-workers 0` and check system memory/pagefile capacity. Quoted external data paths work without copying imagery. Run `python -c "import landslide_benchmark; print(landslide_benchmark.__file__)"` to confirm that the active environment imports this checkout.

## Interpretation and reproducibility

Missing or invalid dates are assigned to `no-event`; this encoding is not proof that no landslide occurred. Inspect `no_event_samples_with_positive_mask` in joint-model reports. Calendar months can be strongly associated with region, and annual embeddings do not by themselves establish precise event timing. The fixed-split evaluator reports temporal baselines; the held-out-region experiment measures a different generalization setting. No script manually corrects source event dates.

Record the split file, seed, training settings (including resource-change history), data version, and selected checkpoint. Keep images, weights, and generated outputs outside Git; the bundled `splits/` files remain tracked. Use the upstream data sources for imagery and their applicable terms. This repository does not bundle final paper metrics or pretrained weights.
