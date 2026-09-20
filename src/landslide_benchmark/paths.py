from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PACKAGE_ROOT.parent

DEFAULT_SPLIT = (
    REPOSITORY_ROOT
    / "Sen12Landslides-main"
    / "outputs"
    / "landdetect_3dcnn"
    / "splits"
    / "dataset_split_resolved.json"
)
DEFAULT_AEF_DIR = REPOSITORY_ROOT / "AEFdata" / "AEF_Embedding_data"
DEFAULT_S2_DIR = REPOSITORY_ROOT / "Sen12Landslides" / "Sen12Landslides" / "s2_data"
DEFAULT_NORM = (
    REPOSITORY_ROOT
    / "Sen12Landslides-main"
    / "tasks"
    / "S12LS-LD"
    / "raw"
    / "s2"
    / "norm.json"
)
DEFAULT_OPTICAL_OUTPUT = PACKAGE_ROOT / "outputs" / "optical"
DEFAULT_SPATIOTEMPORAL_OUTPUT = PACKAGE_ROOT / "outputs" / "spatiotemporal_v2"
LEGACY_AEF_CHECKPOINT = REPOSITORY_ROOT / "LandDetect" / "model" / "aef_s2_3dcnn_split_stable" / "best.pth"
LEGACY_OPTICAL_CHECKPOINT = (
    REPOSITORY_ROOT
    / "Sen12Landslides-main"
    / "outputs"
    / "landdetect_3dcnn"
    / "checkpoints"
    / "best_unet3d.ckpt"
)
