from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DATA_DIR = PROJECT_ROOT / "data/tmot_dataset/images/train"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Model
PRETRAINED_WEIGHTS = "rf-detr_thermal_ema.pth"
USED_DEVICE = "cuda"