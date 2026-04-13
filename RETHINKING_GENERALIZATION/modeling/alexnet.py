import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """
    A simplified AlexNet model adapted for CIFAR-10 and MNIST, 
    consistent with the 'Rethinking Generalization' paper.
    """
    def __init__(self, input_channels, dropout_rate, weight_decay, num_classes=10):
        super(AlexNet, self).__init__()
        
        # Store parameters for reference
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        
        self.features = nn.Sequential(
            # Conv Module 1
            nn.Conv2d(input_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv Module 2
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Determine the size of the features after pooling
        # For 32x32 input:
        # Conv1: 32x32x64 -> Pool1: 15x15x64
        # Conv2: 15x15x64 -> Pool2: 7x7x64
        # For 28x28 input:
        # Conv1: 28x28x64 -> Pool1: 13x13x64
        # Conv2: 13x13x64 -> Pool2: 6x6x64
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(64 * 6 * 6 if input_channels == 1 else 64 * 7 * 7, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_alexnet_model(dropout_rate, weight_decay, is_cifar=True):
    """
    Factory function to initialize and return an AlexNet model object.

    Args:
        dropout_rate (float): The dropout probability.
        weight_decay (float): The weight decay coefficient.
        is_cifar (bool): True for CIFAR-10 (3 channels), False for MNIST (1 channel).

    Returns:
        AlexNet: An instance of the AlexNet model.
    """
    if is_cifar:
        input_channels = 3
    else:
        input_channels = 1
        
    model = AlexNet(
        input_channels=input_channels, 
        dropout_rate=dropout_rate, 
        weight_decay=weight_decay
    )
    
    return model

if __name__ == "__main__":
    # Quick sanity check
    cifar_model = get_alexnet_model(0.5, 0.0001, is_cifar=True)
    mnist_model = get_alexnet_model(0.2, 0.0, is_cifar=False)
    
    print(f"CIFAR AlexNet Parameters: {sum(p.numel() for p in cifar_model.parameters())}")
    print(f"MNIST AlexNet Parameters: {sum(p.numel() for p in mnist_model.parameters())}")
    
    # Test with dummy data
    cifar_input = torch.randn(1, 3, 32, 32)
    mnist_input = torch.randn(1, 1, 28, 28)
    
    print(f"CIFAR Output Shape: {cifar_model(cifar_input).shape}")
    print(f"MNIST Output Shape: {mnist_model(mnist_input).shape}")
    
    print("\nModel Architecture (CIFAR):")
    print(cifar_model)
