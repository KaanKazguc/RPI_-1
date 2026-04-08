import os
from pathlib import Path
from .modules.extract_cifar import main as extract_cifar
from .modules.extract_mnist import main as extract_mnist
from .modules.randomize_labels import main as randomize_labels
from .modules.randomize_pixels import main as randomize_pixels
from .modules.pixel_shuffle import main as pixel_shuffle

class GeneralizationProject:
    """
    Main entry point for managing the "Rethinking Generalization" dataset.
    This class coordinates extraction and various randomization tasks.
    """
    
    def __init__(self, root_dir=None):
        if root_dir is None:
            # Assume 1 level up from this file's directory
            self.root_dir = Path(__file__).resolve().parent.parent
        else:
            self.root_dir = Path(root_dir)
            
        self.data_dir = self.root_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

    def prepare_all(self):
        """Runs all extraction and randomization modules in the correct order."""
        print("--- Starting Data Preparation Pipeline ---")
        
        print("\n[1/5] Extracting CIFAR-10...")
        extract_cifar()
        
        print("\n[2/5] Extracting MNIST...")
        extract_mnist()
        
        print("\n[3/5] Randomizing Labels...")
        randomize_labels()
        
        print("\n[4/5] Randomizing Pixels...")
        randomize_pixels()
        
        print("\n[5/5] Shuffling Pixels (Same Way for all)...")
        pixel_shuffle()
        
        print("\n--- All Preparation Tasks Completed Successfully! ---")

def main():
    """CLI access to data preparation."""
    project = GeneralizationProject()
    project.prepare_all()

if __name__ == "__main__":
    main()
