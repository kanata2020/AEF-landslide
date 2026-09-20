from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = PACKAGE_ROOT / "splits" / "dataset_split_four_regions.json"
DEFAULT_AEF_DIR = PACKAGE_ROOT / "data" / "aef"
DEFAULT_S2_DIR = PACKAGE_ROOT / "data" / "s2"
DEFAULT_NORM = PACKAGE_ROOT / "data" / "optical_norm.json"
DEFAULT_OPTICAL_OUTPUT = PACKAGE_ROOT / "outputs" / "optical"
DEFAULT_SPATIOTEMPORAL_OUTPUT = PACKAGE_ROOT / "outputs" / "spatiotemporal_v2"
