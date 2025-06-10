import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


class CNNModel(nn.Module):
    def __init__(self, d_plus_1: int, m: int):
        super(CNNModel, self).__init__()

        # Input: (batch_size, 1, d+1, m)
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(12, 16, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=3)
        self.dropout = nn.Dropout(0.25)

        # Calculate the flattened size after conv + pool
        def get_flattened_size():
            with torch.no_grad():
                dummy = torch.zeros(1, 1, d_plus_1, m)
                x = self.pool(self.conv2(self.conv1(dummy)))
                return x.view(1, -1).shape[1]

        self.flat_dim = get_flattened_size()

        self.fc1 = nn.Linear(self.flat_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (B, 12, H-2, W-2)
        x = F.relu(self.conv2(x))  # (B, 16, H-4, W-4)
        x = self.pool(x)  # (B, 16, (H-4)//3, (W-4)//3)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def cnn_model(stimulus: np.ndarray, firing_rate: np.ndarray, d_plus_1: int = 45, m: int = 201,
              epochs: int = 20, batch_size: int = 64, lr: float = 1e-3,
              model_path: str = "cnn_model.pt"):
    """
    Train a CNNModel on the provided stimulus and firing rate, save the model and plots.

    Args:
        stimulus: Input features, shape (n_samples, d+1 * m)
        firing_rate: Target values, shape (n_samples,)
        d_plus_1: Height of input (default 45)
        m: Width of input (default 201)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        model_path: Path to save the trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(d_plus_1, m).to(device)

    # Reshape stimulus to (n_samples, d_plus_1, m)
    X = stimulus.reshape(-1, d_plus_1, m)
    y = firing_rate

    # Prepare data
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, d+1, m)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N, 1)
    dataset = TensorDataset(X_tensor, y_tensor)

    # Train/test split
    n_train = int(0.8 * len(dataset))
    n_test = len(dataset) - n_train
    train_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_set):.4f}")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluate and plot
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_test = []
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).cpu().numpy().flatten()
            y_pred.append(preds)
            y_test.append(yb.numpy().flatten())
        y_pred = np.concatenate(y_pred)
        y_test = np.concatenate(y_test)

    # Plot actual vs. predicted
    plt.figure(figsize=(12, 4))
    plt.plot(y_test[:2000], color="black", label="Actual")
    plt.plot(y_pred[:2000], color="red", label="Prediction", linewidth=0.7)
    plt.legend()
    plt.title("CNN: First 2000 Predictions (Test Set)")
    plt.tight_layout()
    plt.savefig("cnn_actual_vs_predicted.png")
    plt.close()

    # Scatter plot
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.figure(figsize=(6, 6))
    plt.scatter(y_pred, y_test, color="black", s=1, alpha=0.5)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("CNN: Predicted vs Actual")
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.tight_layout()
    plt.savefig("cnn_predicted_vs_actual.png")
    plt.close()

    # Plot first conv layer weights as a heatmap (for each filter)
    conv1_weights = model.conv1.weight.data.cpu().numpy()  # shape: (12, 1, 3, 3)
    for i in range(conv1_weights.shape[0]):
        plt.figure()
        plt.imshow(conv1_weights[i, 0], cmap="coolwarm")
        plt.colorbar(label="Weight")
        plt.title(f"CNN Conv1 Filter {i}")
        plt.tight_layout()
        plt.savefig(f"cnn_conv1_filter_{i}.png")
        plt.close()
