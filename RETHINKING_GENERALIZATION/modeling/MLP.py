import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A 3x128 Multi-Layer Perceptron model designed for the Rethinking Generalization project.
    Consists of 3 hidden layers with 128 units each, with ReLU activations and Dropout.
    """
    def __init__(self, input_dim, dropout_rate, weight_decay, num_classes=10):
        super(MLP, self).__init__()
        
        # Store parameters for reference/optimizer configuration
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        
        self.network = nn.Sequential(
            # First Hidden Layer
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            # Second Hidden Layer
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            # Third Hidden Layer
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            # Output Layer
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass. Flattens input tensor to [Batch, Flattened_Features].
        """
        # Automatically handle image flattening: [B, C, H, W] -> [B, C*H*W]
        x = x.view(x.size(0), -1)
        return self.network(x)

def get_mlp_model(dropout_rate, weight_decay, is_cifar=True):
    """
    Factory function to initialize and return a 3x128 MLP model object.

    Args:
        dropout_rate (float): The dropout probability (0.0 to 1.0).
        weight_decay (float): The weight decay (L2 regularization) coefficient.
        is_cifar (bool): Boolean flag to set input dimension. 
                         True for CIFAR-10 (3072), False for MNIST (784).

    Returns:
        MLP: An instance of the 3x128 MLP model.
    """
    # Define input dimension based on the dataset type
    if is_cifar:
        # CIFAR-10: 32x32x3 = 3072
        input_dim = 3072
    else:
        # MNIST: 28x28x1 = 784
        input_dim = 784
        
    model = MLP(
        input_dim=input_dim, 
        dropout_rate=dropout_rate, 
        weight_decay=weight_decay
    )
    
    return model

if __name__ == "__main__":
    # Quick sanity check
    cifar_model = get_mlp_model(0.5, 0.0001, is_cifar=True)
    mnist_model = get_mlp_model(0.2, 0.0, is_cifar=False)
    
    print(f"CIFAR Model Parameters: {sum(p.numel() for p in cifar_model.parameters())}")
    print(f"MNIST Model Parameters: {sum(p.numel() for p in mnist_model.parameters())}")
    print("\nModel Architecture:")
    print(cifar_model)
