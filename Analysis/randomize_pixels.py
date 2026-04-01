import os
import numpy as np
from PIL import Image
from pathlib import Path
import glob

def shuffle_pixels(img_array, percent=100):
    """Shuffles the pixels of an image array (H, W, C) or (H, W)."""
    shape = img_array.shape
    if len(shape) == 3:
        h, w, c = shape
        flat = img_array.reshape(-1, c)
    else:
        h, w = shape
        flat = img_array.reshape(-1)
        
    num_pixels = h * w
    
    if percent >= 100:
        # Full shuffle
        p = np.random.permutation(num_pixels)
        shuffled_flat = flat[p]
    else:
        # Partial shuffle
        num_to_swap = int(num_pixels * (percent / 100.0))
        if num_to_swap < 2:
            return img_array
            
        indices = np.random.choice(num_pixels, num_to_swap, replace=False)
        shuffled_indices = np.random.permutation(indices)
        
        shuffled_flat = flat.copy()
        shuffled_flat[indices] = flat[shuffled_indices]
        
    return shuffled_flat.reshape(shape)

def process_dataset(src_dir, dest_dir, percent, ext="*.png"):
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    files = list(src_path.glob(ext))
    if not files:
        print(f"No files found in {src_dir} with extension {ext}")
        return
        
    print(f"Processing {len(files)} images for {dest_path.name} ({percent}% shuffle)...")
    
    # We use a fixed seed for reproducibility, but also so that 
    # if the user wants "same shuffle for all", we can decide.
    # The paper says "fixed random permutation for all images".
    # I will generate one permutation per dataset per percent level.
    
    # Actually, if I use the function above inside the loop, 
    # it's better to pre-calculate the permutation for the 100% case if we want it fixed.
    
    # Let's read one image to get dimensions
    first_img = np.array(Image.open(files[0]))
    h, w = first_img.shape[:2]
    num_pixels = h * w
    
    np.random.seed(42)
    if percent >= 100:
        perm = np.random.permutation(num_pixels)
    else:
        # For partial, we also pick fixed indices to swap for all images
        num_to_swap = int(num_pixels * (percent / 100.0))
        indices = np.random.choice(num_pixels, num_to_swap, replace=False)
        shuffled_indices = np.random.permutation(indices)
    
    for i, f in enumerate(files):
        img = Image.open(f)
        arr = np.array(img)
        
        # Apply fixed permutation
        shape = arr.shape
        if len(shape) == 3:
            flat = arr.reshape(-1, shape[2])
            new_flat = flat.copy()
            if percent >= 100:
                new_flat = flat[perm]
            else:
                new_flat[indices] = flat[shuffled_indices]
        else:
            flat = arr.reshape(-1)
            new_flat = flat.copy()
            if percent >= 100:
                new_flat = flat[perm]
            else:
                new_flat[indices] = flat[shuffled_indices]
        
        shuffled_img = Image.fromarray(new_flat.reshape(shape))
        shuffled_img.save(dest_path / f.name)
        
        if (i + 1) % 5000 == 0:
            print(f"  Progress: {i + 1}/{len(files)}...")

def main():
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    
    # CIFAR-10 (BMP files)
    cifar_src = raw_dir / "cifar10_images"
    # MNIST (PNG files)
    mnist_src = raw_dir / "mnist_images"
    
    # Task 1: CIFAR-10 100%
    process_dataset(cifar_src, processed_dir / "cifar10_images_randomized", 100, "*.bmp")
    
    # Task 2: CIFAR-10 10%
    process_dataset(cifar_src, processed_dir / "cifar10_images_randomized_%10", 10, "*.bmp")
    
    # Task 3: MNIST 100%
    process_dataset(mnist_src, processed_dir / "mnist_images_randomized", 100, "*.png")
    
    # Task 4: MNIST 10%
    process_dataset(mnist_src, processed_dir / "mnist_images_randomized_%10", 10, "*.png")

if __name__ == "__main__":
    main()
