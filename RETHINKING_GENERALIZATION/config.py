import os
from pathlib import Path

# --- BASE PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# --- REGISTERED DATASETS ---
# Register known paths. If new datasets are added, they can be appended to this list, or 
# simply placed in raw/ or processed/ folders where the automatic fallback functions will find them.
IMAGE_DATASETS = {
    "cifar10": RAW_DATA_DIR / "cifar10_images",
    "cifar10_images_randomized": PROCESSED_DATA_DIR / "cifar10_images_randomized",
    "cifar10_images_randomized_%10": PROCESSED_DATA_DIR / "cifar10_images_randomized_%10",
    "cifar10_pixel_shuffle": PROCESSED_DATA_DIR / "cifar10_pixel_shuffle",
    
    "mnist": RAW_DATA_DIR / "mnist_images",
    "mnist_images_randomized": PROCESSED_DATA_DIR / "mnist_images_randomized",
    "mnist_images_randomized_%10": PROCESSED_DATA_DIR / "mnist_images_randomized_%10",
    "mnist_pixel_shuffle": PROCESSED_DATA_DIR / "mnist_pixel_shuffle",
}

LABEL_DATASETS = {
    "cifar10": RAW_DATA_DIR / "cifar10_labels.csv",
    "cifar10_labels_randomized": PROCESSED_DATA_DIR / "cifar10_labels_randomized.csv",
    
    "mnist": RAW_DATA_DIR / "mnist_labels.csv",
    "mnist_labels_randomized": PROCESSED_DATA_DIR / "mnist_labels_randomized.csv",
}

# --- GETTERS WITH AUTOMATIC FALLBACK ---

def get_image_dir(name):
    if name in IMAGE_DATASETS:
        return IMAGE_DATASETS[name]
    
    processed_path = PROCESSED_DATA_DIR / name
    if processed_path.exists():
        return processed_path
    
    raw_path = RAW_DATA_DIR / name
    if raw_path.exists():
        return raw_path
        
    raise ValueError(f"Image dataset '{name}' not found locally or in registry. Check paths under: {DATA_DIR}")

def get_label_csv(name):
    if name in LABEL_DATASETS:
        return LABEL_DATASETS[name]
        
    csv_name = name if name.endswith(".csv") else f"{name}.csv"
    
    processed_path = PROCESSED_DATA_DIR / csv_name
    if processed_path.exists():
        return processed_path
    
    raw_path = RAW_DATA_DIR / csv_name
    if raw_path.exists():
        return raw_path
        
    raise ValueError(f"Label dataset '{name}' not found locally or in registry. Check paths under: {DATA_DIR}")

# --- SAVE CATEGORY LOGIC ---

def get_save_category(image_dataset, label_dataset):
    """
    Determines where the model should be saved under the 'models/' structure.
    Adjust this mapping as new experiments are added.
    """
    img_lower = image_dataset.lower()
    lbl_lower = label_dataset.lower()
    
    if "labels_randomized" in lbl_lower:
        return "random_labels"
    elif "images_randomized" in img_lower:
        return "random_pixels"
    elif "pixel_shuffle" in img_lower:
        return "shuffled_pixels"
    else:
        return "normal"
