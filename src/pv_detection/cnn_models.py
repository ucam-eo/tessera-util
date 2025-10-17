import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class PVDetectionCNN(nn.Module):
    """
    Lightweight CNN for PV detection using 3x3 patches.
    Designed for 128-channel input features with 3x3 spatial resolution.
    """

    def __init__(self, input_channels: int = 128, num_classes: int = 2,
                 dropout_rate: float = 0.2):
        """
        Initialize CNN model.

        Args:
            input_channels: number of input channels (default: 128)
            num_classes: number of output classes (default: 2 for binary classification)
            dropout_rate: dropout probability
        """
        super(PVDetectionCNN, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        # First convolutional block - reduce channels while preserving spatial info
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Second convolutional block - further feature extraction
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        # Third convolutional block - final feature compression
        self.conv3 = nn.Conv2d(32, 16, kernel_size=1, bias=False)  # 1x1 conv for channel reduction
        self.bn3 = nn.BatchNorm2d(16)

        # Global average pooling to reduce spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Final classification layer
        self.classifier = nn.Linear(16, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input tensor of shape (batch_size, channels, 3, 3)

        Returns:
            output: logits of shape (batch_size, num_classes)
        """
        # Input shape: (batch_size, 128, 3, 3)

        # First conv block
        x = self.conv1(x)  # (batch_size, 64, 3, 3)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        # Second conv block
        x = self.conv2(x)  # (batch_size, 32, 3, 3)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        # Third conv block
        x = self.conv3(x)  # (batch_size, 16, 3, 3)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)

        # Global average pooling
        x = self.global_avg_pool(x)  # (batch_size, 16, 1, 1)
        x = torch.flatten(x, 1)  # (batch_size, 16)

        # Dropout and classification
        x = self.dropout(x)
        x = self.classifier(x)  # (batch_size, num_classes)

        return x

    def predict_proba(self, x):
        """
        Predict class probabilities.

        Args:
            x: input tensor

        Returns:
            probabilities: softmax probabilities
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x):
        """
        Predict class labels.

        Args:
            x: input tensor

        Returns:
            predictions: predicted class labels
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

class PVDetectionCNNDeep(nn.Module):
    """
    Deeper CNN variant for potentially better performance.
    """

    def __init__(self, input_channels: int = 128, num_classes: int = 2,
                 dropout_rate: float = 0.3):
        super(PVDetectionCNNDeep, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        # Feature extraction layers
        self.features = nn.Sequential(
            # First block: 128 -> 64 channels
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Second block: 64 -> 64 channels (deeper feature extraction)
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Third block: 64 -> 32 channels
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Fourth block: 32 -> 16 channels
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Global pooling and classification
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(16, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

def create_model(model_type: str = 'simple', input_channels: int = 128,
                num_classes: int = 2, **kwargs) -> nn.Module:
    """
    Factory function to create CNN models.

    Args:
        model_type: type of model ('simple' or 'deep')
        input_channels: number of input channels
        num_classes: number of output classes
        **kwargs: additional arguments for model

    Returns:
        CNN model
    """
    if model_type == 'simple':
        model = PVDetectionCNN(input_channels, num_classes, **kwargs)
    elif model_type == 'deep':
        model = PVDetectionCNNDeep(input_channels, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Created {model_type} CNN model:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    return model

if __name__ == "__main__":
    # Test model creation and forward pass
    logging.basicConfig(level=logging.INFO)

    # Test simple model
    model = create_model('simple', input_channels=128, num_classes=2)
    print(f"Simple model: {model}")

    # Test with dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 128, 3, 3)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
        probabilities = model.predict_proba(dummy_input)
        predictions = model.predict(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Predictions shape: {predictions.shape}")

    # Test deep model
    deep_model = create_model('deep', input_channels=128, num_classes=2)
    print(f"\nDeep model parameter count: {sum(p.numel() for p in deep_model.parameters()):,}")

    print("Model architecture test completed successfully!")