import csv
import random
import os

def randomize_labels(input_csv, output_csv, seed=42):
    if not os.path.exists(input_csv):
        print(f"File not found: {input_csv}")
        return
        
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
        
    # Extract paths and labels
    paths = [row[0] for row in data]
    labels = [row[1] for row in data]
    
    # Shuffle only the labels to mismatch true labels and images,
    # mimicking the "Rethinking Generalization" methodology perfectly.
    random.seed(seed)
    random.shuffle(labels)
    
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for p, l in zip(paths, labels):
            writer.writerow([p, l])
            
    print(f"Randomized labels saved to {output_csv}")
    print(f"Processed {len(data)} rows.")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    cifar_in = os.path.join(base_dir, "data", "raw", "cifar10_labels.csv")
    cifar_out = os.path.join(base_dir, "data", "processed", "cifar10_labels_randomized.csv")
    
    mnist_in = os.path.join(base_dir, "data", "raw", "mnist_labels.csv")
    mnist_out = os.path.join(base_dir, "data", "processed", "mnist_labels_randomized.csv")
    
    print("Randomizing CIFAR-10 labels...")
    randomize_labels(cifar_in, cifar_out, seed=42)
    
    print("\nRandomizing MNIST labels...")
    randomize_labels(mnist_in, mnist_out, seed=42)

if __name__ == "__main__":
    main()
