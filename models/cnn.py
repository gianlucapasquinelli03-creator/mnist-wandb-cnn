import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    CNN semplice per la classificazione delle cifre MNIST (28x28, 1 canale).

    Architettura:
        Conv(1->32) -> ReLU -> MaxPool  (28x28 -> 14x14)
        Conv(32->64) -> ReLU -> MaxPool (14x14 -> 7x7)
        Dropout
        Flatten -> FC(64*7*7 -> 128) -> ReLU -> Dropout -> FC(128 -> num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # -> (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # -> (B, 64, 7, 7)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)                # -> (B, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)                        # logits, non serve softmax
        return x


if __name__ == "__main__":
    # Piccolo self-test: verifica che le shape siano corrette
    model = SimpleCNN()
    dummy = torch.randn(4, 1, 28, 28)
    out = model(dummy)
    print("Output shape:", out.shape)  # atteso: torch.Size([4, 10])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Numero parametri: {n_params:,}")
