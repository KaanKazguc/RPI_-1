import os
import gzip
import csv
import struct
import urllib.request
from PIL import Image

def download_mnist(raw_dir):
    # Google's dependable mirror for MNIST data
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    for f in files:
        filepath = os.path.join(raw_dir, f)
        if not os.path.exists(filepath):
            print(f"Downloading {f}...", flush=True)
            urllib.request.urlretrieve(base_url + f, filepath)
    return files

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Ensure dataset is downloaded
    download_mnist(raw_dir)
    
    out_dir = os.path.join(raw_dir, "mnist_images")
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(raw_dir, "mnist_labels.csv")
    
    datasets = [
        ("train", "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
        ("test", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
    ]
    
    img_idx = 0
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'label'])
        
        for prefix, img_file, lbl_file in datasets:
            img_path = os.path.join(raw_dir, img_file)
            lbl_path = os.path.join(raw_dir, lbl_file)
            
            # Open the gzip files directly to read bytes
            with gzip.open(img_path, 'rb') as f_img:
                magic, num, rows, cols = struct.unpack(">IIII", f_img.read(16))
                img_data = f_img.read()
                
            with gzip.open(lbl_path, 'rb') as f_lbl:
                magic_lbl, num_lbl = struct.unpack(">II", f_lbl.read(8))
                lbl_data = f_lbl.read()
                
            if num != num_lbl:
                raise ValueError("Image and label counts do not match")
                
            print(f"Processing {prefix} dataset ({num} images)...", flush=True)
            
            for i in range(num):
                # MNIST is 28x28 grayscale bytes
                start = i * rows * cols
                end = start + rows * cols
                img_bytes = img_data[start:end]
                label = lbl_data[i]
                
                # Convert to PNG using pillow
                img = Image.frombytes('L', (cols, rows), img_bytes)
                
                filename = f"{prefix}_{i:05d}.png"
                full_path = os.path.join(out_dir, filename)
                img.save(full_path)
                
                csv_rel = os.path.relpath(full_path, start=raw_dir)
                writer.writerow([csv_rel, label])
                
                img_idx += 1

    print(f"Done! {img_idx} images saved to {out_dir}")
    print(f"CSV labels saved to {csv_path}")

if __name__ == "__main__":
    main()
