import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import random
import os


class CNNModel(nn.Module):
    def __init__(self, height: int = 18, width: int = 21):
        super(CNNModel, self).__init__()

        # CNN architecture remains the same
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(12, 16, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(0.25)

        # Calculate the flattened size after conv + pool
        def get_flattened_size():
            with torch.no_grad():
                dummy = torch.zeros(1, 1, height, width)
                x = self.pool(self.conv2(self.conv1(dummy)))
                return x.view(1, -1).shape[1]

        self.flat_dim = get_flattened_size()
        self.fc1 = nn.Linear(self.flat_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def set_seed(seed: int = 42):
    """Set seed for reproducibility across torch, numpy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_evaluate_fold(model, train_loader, val_loader, optimizer, criterion,
                            device, epochs, fold_idx, results_dir):
    """Train and evaluate a model on a single fold."""
    model.train()
    train_losses = []

    # Training loop
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

        avg_loss = epoch_loss / len(train_loader.sampler)
        train_losses.append(avg_loss)
        print(f"Fold {fold_idx + 1}, Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

    # Evaluate on validation set
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            preds = model(xb).cpu().numpy().flatten()
            val_preds.append(preds)
            val_true.append(yb.numpy().flatten())

    val_preds = np.concatenate(val_preds)
    val_true = np.concatenate(val_true)
    r2 = r2_score(val_true, val_preds)

    # Plot training loss for this fold
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold_idx + 1} Training Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'fold_{fold_idx + 1}_loss.png'))
    plt.close()

    return r2, model, val_preds, val_true


def cnn_model_kfold(stimulus: np.ndarray, firing_rate: np.ndarray,
                    height: int = 18, width: int = 21,
                    epochs: int = 40, batch_size: int = 64, lr: float = 1e-3,
                    model_path: str = "cnn_model.pt", seed: int = 42,
                    k_folds: int = 5):
    """
    Train a CNN model using k-fold cross-validation.

    Args:
        stimulus: Input features, shape (n_samples, n_features) where n_features=378
        firing_rate: Target values, shape (n_samples,)
        height, width: Dimensions for reshaping input
        epochs: Number of training epochs per fold
        batch_size: Batch size
        lr: Learning rate
        model_path: Path to save the best model
        seed: Random seed
        k_folds: Number of folds for cross-validation (default=5)
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create results directory
    results_dir = "cnn_kfold_results"
    os.makedirs(results_dir, exist_ok=True)

    # Reshape stimulus to (n_samples, height, width)
    X = stimulus.reshape(-1, height, width)
    y = firing_rate

    # Prepare data
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, height, width)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N, 1)
    dataset = TensorDataset(X_tensor, y_tensor)

    # Set up k-fold cross-validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_results = []
    best_r2 = -float('inf')
    best_model = None

    # Perform k-fold cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nTraining fold {fold_idx + 1}/{k_folds}")

        # Create data loaders for this fold
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

        # Initialize a new model instance for this fold
        model = CNNModel(height, width).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Train and evaluate on this fold
        r2, trained_model, val_preds, val_true = train_and_evaluate_fold(
            model, train_loader, val_loader, optimizer, criterion,
            device, epochs, fold_idx, results_dir
        )

        fold_results.append({
            'fold': fold_idx + 1,
            'r2': r2,
            'val_preds': val_preds,
            'val_true': val_true
        })

        # Save the best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = trained_model
            torch.save(trained_model.state_dict(), model_path)
            print(f"New best model saved (Fold {fold_idx + 1}, R² = {r2:.4f})")

    # Calculate average R² across all folds
    avg_r2 = np.mean([f['r2'] for f in fold_results])
    r2_std = np.std([f['r2'] for f in fold_results])
    print(f"\nCross-validation complete.")
    print(f"Average R² across {k_folds} folds: {avg_r2:.4f} ± {r2_std:.4f}")

    # Plot R² for each fold
    plt.figure(figsize=(8, 5))
    fold_nums = [f['fold'] for f in fold_results]
    r2_scores = [f['r2'] for f in fold_results]
    plt.bar(fold_nums, r2_scores)
    plt.axhline(y=avg_r2, color='r', linestyle='--', label=f'Avg R² = {avg_r2:.4f}')
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.title('R² Score per Fold')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'kfold_r2_scores.png'))
    plt.close()

    # Plot predictions from the best fold
    best_fold_idx = np.argmax([f['r2'] for f in fold_results])
    best_fold = fold_results[best_fold_idx]

    # Sample plot of predictions (first 2000 points)
    plt.figure(figsize=(12, 4))
    sample_size = min(2000, len(best_fold['val_true']))
    plt.plot(best_fold['val_true'][:sample_size], color="black", label="Actual")
    plt.plot(best_fold['val_preds'][:sample_size], color="red", label="Prediction", linewidth=0.7)
    plt.legend()
    plt.title(f"Best Fold ({best_fold_idx + 1}) Predictions")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "best_fold_predictions.png"))
    plt.close()

    # Scatter plot for best fold
    plt.figure(figsize=(6, 6))
    min_val = min(best_fold['val_true'].min(), best_fold['val_preds'].min())
    max_val = max(best_fold['val_true'].max(), best_fold['val_preds'].max())
    plt.scatter(best_fold['val_preds'], best_fold['val_true'], color="black", s=1, alpha=0.5)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Best Fold ({best_fold_idx + 1}) Predicted vs Actual")
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "best_fold_scatter.png"))
    plt.close()

    return best_model, fold_results

def evaluate_cnn_model(model_path: str, stimulus: np.ndarray, firing_rate: np.ndarray,
                       height: int = 18, width: int = 21,
                       batch_size: int = 64, seed: int = 42):
    """
    Loads a CNN model from file, predicts on the given stimulus, and returns the R² score.

    Args:
        model_path: Path to the saved model (.pt file)
        stimulus: Input features, shape (n_samples, n_features) where n_features=378
        firing_rate: Target values, shape (n_samples,)
        height: Height for reshaping input (default 18)
        width: Width for reshaping input (default 21)
        batch_size: Batch size for evaluation
        seed: Random seed

    Returns:
        r2: R² score of predictions
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(height, width).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Reshape stimulus to (n_samples, height, width)
    X = stimulus.reshape(-1, height, width)
    y = firing_rate

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, height, width)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N, 1)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    y_true = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy().flatten()
            preds.append(pred)
            y_true.append(yb.numpy().flatten())
    preds = np.concatenate(preds)
    y_true = np.concatenate(y_true)

    r2 = r2_score(y_true, preds)
    print(f"R² score: {r2:.4f}")
    return r2

if __name__ == "__main__":
    from tools import load_state

    stim = load_state("./data/train_stimuli.pkl")
    resp = load_state("./data/train_response.pkl")

    # Train model with 5-fold cross-validation
    # best_model, fold_results = cnn_model_kfold(
    #     stim, resp,
    #     k_folds=5,
    #     epochs=15,
    #     batch_size=64,
    #     model_path="cnn_best_kfold_model.pt"
    # )

    # Optionally evaluate on validation set
    val_stim = load_state("./data/val_stimuli.pkl")
    val_resp = load_state("./data/val_response.pkl")
    if val_stim is not None and val_resp is not None:
        print("\nEvaluating best model on validation set:")
        evaluate_cnn_model("cnn_best_kfold_model.pt", val_stim, val_resp)
