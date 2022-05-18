"""
Global config file for vax project
"""

# Project root directory
ROOT = "/sdf/home/s/skivelso/sanket/vax"

RAW_TOMO_DIR = f"{ROOT}/data/raw" 

# Path to csv with annotation information
ANNOTATION_CSV = f"{ROOT}/data/for_processing/csv" 

# Path to extracted image slices
SLICES_TO_LABEL = f"{ROOT}/data/for_processing/tomo_slices" 

# Path to mitochondria masks
VAX_ANNOTATIONS = f"{ROOT}/data/for_processing/vax_masks"

TRAIN_TOMO_DIR = f"{ROOT}/data/train_tomos" 

LOG_DIR = f"{ROOT}/src/train/logs"

PREDICTIONS_DIR = f"{ROOT}/data/predictions"