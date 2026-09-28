import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    CNN con 3 blocchi convoluzionali per MNIST (28x28, 1 canale).

        Conv(1->32)   -> ReLU -> MaxPool  (28x28 -> 14x14)
        Conv(32->64)  -> ReLU -> MaxPool  (14x14 -> 7x7)
        Conv(64->128) -> ReLU -> MaxPool  (7x7   -> 3x3, il pooling arrotonda per difetto)
        Dropout
        Flatten -> FC(128*3*3 -> 128) -> ReLU -> Dropout -> FC(128 -> num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # -> (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # -> (B, 64, 7, 7)
        x = self.pool(F.relu(self.conv3(x)))   # -> (B, 128, 3, 3)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)                # -> (B, 128*3*3)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


if __name__ == "__main__":
    model = SimpleCNN()
    out = model(torch.randn(4, 1, 28, 28))
    print("Output shape:", out.shape)  # atteso: torch.Size([4, 10])
    print(f"Numero parametri: {sum(p.numel() for p in model.parameters()):,}")
    print("Output shape:", out.shape)  # atteso: torch.Size([4, 10])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Numero parametri: {n_params:,}")
