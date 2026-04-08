import os
import numpy as np
from PIL import Image
from pathlib import Path

def process_dataset_pixel_shuffle(dataset_name, file_extension="*.bmp", seed=42):
    """
    Shuffles the pixels of images in a dataset using a fixed permutation for all images.
    This corresponds to the 'shuffled pixels' experiment in 'Understanding Deep Learning 
    Requires Rethinking Generalization'.
    """
    # Base directory (project root)
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_dir = base_dir / "data" / "raw" / f"{dataset_name}_images"
    processed_dir = base_dir / "data" / "processed" / f"{dataset_name}_pixel_shuffle"
    
    if not raw_dir.exists():
        print(f"Warning: Raw {dataset_name} images not found at {raw_dir}. Skipping...")
        return

    processed_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(raw_dir.glob(file_extension))
    if not files:
        print(f"Warning: No {file_extension} files found in {raw_dir}. Skipping...")
        return
        
    print(f"Processing {len(files)} images for {dataset_name} pixel shuffle...")
    
    # Get dimensions from first image
    first_img = Image.open(files[0])
    width, height = first_img.size
    
    # Generate a fixed permutation for ALL images
    # Using a fixed seed for reproducibility
    np.random.seed(seed)
    num_pixels = width * height
    permutation = np.random.permutation(num_pixels)
    
    for i, file_path in enumerate(files):
        img = Image.open(file_path)
        img_array = np.array(img)
        
        # Determine shape and channels
        # RGB: (H, W, 3), Grayscale: (H, W) or (H, W, 1)
        shape = img_array.shape
        if len(shape) == 3:
            h, w, c = shape
            flat_img = img_array.reshape(-1, c)
            shuffled_flat = flat_img[permutation]
            shuffled_img_array = shuffled_flat.reshape(h, w, c)
        else:
            h, w = shape
            flat_img = img_array.reshape(-1)
            shuffled_flat = flat_img[permutation]
            shuffled_img_array = shuffled_flat.reshape(h, w)
            
        shuffled_img = Image.fromarray(shuffled_img_array)
        
        # Save to processed directory with the same filename
        shuffled_img.save(processed_dir / file_path.name)
        
        if (i + 1) % 5000 == 0:
            print(f"  Progress: {i + 1}/{len(files)}...")

    print(f"Successfully saved {dataset_name} shuffled images to {processed_dir}")

def main():
    # Process CIFAR-10
    process_dataset_pixel_shuffle("cifar10", "*.bmp")
    
    # Process MNIST
    process_dataset_pixel_shuffle("mnist", "*.png")

if __name__ == "__main__":
    main()

