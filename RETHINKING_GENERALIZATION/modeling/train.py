import os
import sys
import argparse
import json
from pathlib import Path

# Fix path for local imports if run from project root
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from MLP import get_mlp_model
from alexnet import get_alexnet_model
from config import MODELS_DIR, get_image_dir, get_label_csv, get_save_category

class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_csv, is_cifar=True):
        self.images_dir = Path(images_dir)
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not Path(labels_csv).exists():
            raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
            
        self.labels_df = pd.read_csv(labels_csv)
        self.is_cifar = is_cifar
        
        self.cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.cifar_classes)}
        
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        image_subpath = row['image_path']
        image_filename = Path(image_subpath).name
        
        img_path = self.images_dir / image_filename
        
        if self.is_cifar:
            image = Image.open(img_path).convert('RGB')
            label_str = str(row['label'])
            label = self.class_to_idx.get(label_str, 0) if not label_str.isdigit() else int(label_str)
        else:
            image = Image.open(img_path).convert('L')
            label = int(row['label'])
            
        image = self.transform(image)
        return image, label

def get_model(model_type, do, wd, is_cifar):
    model_type = model_type.upper()
    if model_type == "MLP":
        return get_mlp_model(do, wd, is_cifar)
    elif model_type == "ALEXNET":
        return get_alexnet_model(do, wd, is_cifar)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    parser = argparse.ArgumentParser(description="Train models for Rethinking Generalization")
    parser.add_argument("model_type", type=str, help="Model architecture (e.g. MLP, AlexNet)")
    parser.add_argument("image_dataset", type=str, help="Images dataset name (e.g. cifar10_images_randomized)")
    parser.add_argument("label_dataset", type=str, help="Labels dataset name (e.g. cifar10, cifar10_labels_randomized)")
    parser.add_argument("weight_decay", type=float, help="L2 regularization (weight decay)")
    parser.add_argument("dropout_rate", type=float, help="Dropout rate")
    parser.add_argument("epochs", type=int, help="Number of training epochs")
    parser.add_argument("batch_size", type=int, help="Batch size for training")
    
    args = parser.parse_args()
    
    # Resolve Paths via Config Module
    images_dir = get_image_dir(args.image_dataset)
    labels_csv = get_label_csv(args.label_dataset)
    
    is_cifar = 'cifar' in args.image_dataset.lower()
    
    # Meta Configuration Logic
    category = get_save_category(args.image_dataset, args.label_dataset)
    prefix = "cifar" if is_cifar else "mnist"
    model_name = f"{prefix}_{args.model_type}"
    if args.weight_decay > 0:
        model_name += f"_wd_{args.weight_decay}"
    if args.dropout_rate > 0:
        model_name += f"_do_{args.dropout_rate}"
        
    model_filename = model_name + ".pth"
    stats_filename = model_name + "_history.json"
    
    cat_dir = MODELS_DIR / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = cat_dir / model_filename
    stats_path = cat_dir / stats_filename
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Initializing {args.model_type}...")
    model = get_model(args.model_type, args.dropout_rate, args.weight_decay, is_cifar)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay)
    
    start_epoch = 0
    history = {"train_loss": [], "train_acc": []}
    
    if model_path.exists():
        print(f"Model found at {model_path}. Loading...")
        if hasattr(torch, "weights_only"):
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
            
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                history = json.load(f)
            start_epoch = len(history.get("train_loss", []))
            print(f"Resuming training from epoch {start_epoch + 1}...")
        else:
            print("History file not found. Starting history from scratch.")
    else:
        print(f"Model not found. Creating a new one and training from scratch...")
        
    if start_epoch >= args.epochs:
        print("Model is already trained for the requested number of epochs.")
        return
        
    print(f"Loading dataset from:\n  Images: {images_dir}\n  Labels: {labels_csv}")
    dataset = CustomDataset(images_dir, labels_csv, is_cifar)
    
    # Accelerated dataloading with multi-threading and optimized RAM to GPU transfers
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=(device.type == 'cuda')
    )
    
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
        
    print("Saving model and stats...")
    torch.save(model.state_dict(), model_path)
    with open(stats_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"Done. Model saved to {model_path}")

if __name__ == "__main__":
    main()
