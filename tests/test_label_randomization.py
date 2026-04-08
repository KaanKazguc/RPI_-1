import csv
import os
import pytest
from pathlib import Path

def get_csv_data(filepath):
    """Utility to read image_path and label from CSV."""
    if not os.path.exists(filepath):
        return None
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            data.append((row[0], row[1]))
    return data

@pytest.mark.parametrize("dataset_name", ["cifar10", "mnist"])
def test_randomization_is_effective(dataset_name):
    # Paths (adjust based on project layout)
    base_dir = Path(__file__).parent.parent
    raw_csv = base_dir / "data" / "raw" / f"{dataset_name}_labels.csv"
    processed_csv = base_dir / "data" / "processed" / f"{dataset_name}_labels_randomized.csv"

    # Skip if files don't exist yet
    if not raw_csv.exists() or not processed_csv.exists():
        pytest.skip(f"CSV files for {dataset_name} not found.")

    original_data = get_csv_data(raw_csv)
    randomized_data = get_csv_data(processed_csv)

    # 1. Total samples must match
    assert len(original_data) == len(randomized_data), f"{dataset_name}: Count mismatch!"
    total = len(original_data)

    # 2. Image paths must be in identical order
    for i in range(total):
        assert original_data[i][0] == randomized_data[i][0], f"Path mismatch at index {i}!"

    # 3. Frequency of each class should remain the same (we only shuffled the list)
    orig_labels = [row[1] for row in original_data]
    rand_labels = [row[1] for row in randomized_data]
    
    orig_counts = {}
    for l in orig_labels:
        orig_counts[l] = orig_counts.get(l, 0) + 1
        
    rand_counts = {}
    for l in rand_labels:
        rand_counts[l] = rand_counts.get(l, 0) + 1
        
    assert orig_counts == rand_counts, f"{dataset_name}: Label counts differ! Randomization might have lost labels."

    # 4. Check overlap (Randomization Test)
    # If the list is truly randomized, the number of samples that still match their
    # original label should be close to 1/num_classes (e.g., 10%).
    # If they are 100% matched, it failed.
    matches = sum(1 for i in range(total) if original_data[i][1] == randomized_data[i][1])
    match_rate = matches / total

    print(f"\n{dataset_name.upper()} Match Rate: {match_rate:.2%}")
    
    # Allow some statistical variance, but 100% match or 0% match is highly unlikely.
    # Expected is ~10%. If it's > 25%, it's definitely suspicious.
    # If it's exactly the same, it definitely failed.
    assert match_rate < 0.25, f"{dataset_name}: Too many matches ({match_rate:.2%}). Shuffling might be broken."
    assert original_data != randomized_data, f"{dataset_name}: Randomized data is identical to original!"

def test_processed_files_exist():
    base_dir = Path(__file__).parent.parent
    cifar_file = base_dir / "data" / "processed" / "cifar10_labels_randomized.csv"
    mnist_file = base_dir / "data" / "processed" / "mnist_labels_randomized.csv"
    
    # This test ensures the processing step was actually run
    assert cifar_file.exists(), "CIFAR randomized file is missing."
    assert mnist_file.exists(), "MNIST randomized file is missing."
