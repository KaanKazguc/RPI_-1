import pickle
import sys
import os
import csv
import urllib.request
import tarfile

def write_bmp(filename, r_bytes, g_bytes, b_bytes, width=32, height=32):
    # BMP header for 24-bit uncompressed image
    filesize = 54 + 3 * width * height
    header = bytearray([
        0x42, 0x4D,             # "BM"
        filesize & 0xFF, (filesize >> 8) & 0xFF, (filesize >> 16) & 0xFF, (filesize >> 24) & 0xFF,
        0, 0, 0, 0,             # reserved
        54, 0, 0, 0,            # data offset
        40, 0, 0, 0,            # info header size
        width & 0xFF, (width >> 8) & 0xFF, 0, 0,
        height & 0xFF, (height >> 8) & 0xFF, 0, 0,
        1, 0,                   # planes
        24, 0,                  # bit depth
        0, 0, 0, 0,             # uncompressed
        0, 0, 0, 0,             # image size (can be 0)
        0x13, 0x0B, 0, 0,       # x pixels per meter (2835)
        0x13, 0x0B, 0, 0,       # y pixels per meter (2835)
        0, 0, 0, 0,             # colors used
        0, 0, 0, 0              # important colors
    ])
    
    # Interleave to BGR and flip horizontally (BMP stores bottom-up)
    img_data = bytearray(3 * width * height)
    # CIFAR stores rows top-to-bottom, we need bottom-to-top for standard BMP
    for y in range(height):
        # target row in BMP (bottom-up)
        inv_y = height - 1 - y
        for x in range(width):
            src_idx = y * width + x
            dst_idx = (inv_y * width + x) * 3
            img_data[dst_idx] = b_bytes[src_idx]     # B
            img_data[dst_idx + 1] = g_bytes[src_idx] # G
            img_data[dst_idx + 2] = r_bytes[src_idx] # R
            
    with open(filename, 'wb') as f:
        f.write(header)
        f.write(img_data)

def main():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    
    # The user is running this in the project root hopefully, or in data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    archive_path = os.path.join(raw_dir, "cifar-10-python.tar.gz")
    if not os.path.exists(archive_path):
        print("Downloading CIFAR-10 dataset...", flush=True)
        urllib.request.urlretrieve(url, archive_path)
    
    cifar_dir = os.path.join(raw_dir, "cifar-10-batches-py")
    if not os.path.exists(cifar_dir):
        print("Extracting CIFAR-10...", flush=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=raw_dir)
            
    out_dir = os.path.join(raw_dir, "images")
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(raw_dir, "cifar10_labels.csv")
    
    print("Converting to images...", flush=True)
    batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
    
    img_idx = 0
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'label'])
        
        meta_path = os.path.join(cifar_dir, 'batches.meta')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f, encoding='bytes')
            label_names = [x.decode('utf-8') for x in meta[b'label_names']]
            
        for batch in batches:
            print(f"Processing {batch}...", flush=True)
            batch_path = os.path.join(cifar_dir, batch)
            with open(batch_path, 'rb') as f:
                d = pickle.load(f, encoding='bytes')
                data = d[b'data']
                labels = d[b'labels']
                
                # 'filenames' may not exist in all batches, fallback to generic
                filenames = d.get(b'filenames', [f"{batch}_{i}.bmp".encode('utf-8') for i in range(len(data))])
                
                for i in range(len(data)):
                    img_data = data[i]
                    label = labels[i]
                    label_name = label_names[label]
                    orig_filename = filenames[i].decode('utf-8')
                    
                    name_only = orig_filename.rsplit('.', 1)[0]
                    bmp_filename = f"{name_only}.bmp"
                    bmp_path = os.path.join(out_dir, bmp_filename)
                    
                    write_bmp(bmp_path, img_data[0:1024], img_data[1024:2048], img_data[2048:3072])
                    
                    rel_path = os.path.relpath(bmp_path, start=os.path.dirname(raw_dir))
                    # usually better to store just "images/filename.bmp"
                    csv_rel = os.path.relpath(bmp_path, start=raw_dir)
                    writer.writerow([csv_rel, label_name])
                    
                    img_idx += 1

    print(f"Done! {img_idx} images saved to {out_dir}")
    print(f"CSV labels saved to {csv_path}")

if __name__ == "__main__":
    main()
