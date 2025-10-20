import os
from itertools import product
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class CryDataset(Dataset):
    """Cry dataset."""

    def __init__(
        self,
        npy_path: str,
        binned_label_path: str,
        scaler: StandardScaler | None = None,
    ):
        data: npt.NDarray = np.load(npy_path, allow_pickle=True)
        self.data: npt.NDArray = np.array([sample[0] for sample in data])
        self.labels: npt.NDArray = np.load(binned_label_path)
        
        if scaler:
            data_shape = self.data.shape
            self.data = scaler.transform(
                self.data.reshape(-1, data_shape[-1])
            ).reshape(data_shape)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        # add channel dimension: (1, time, freq)
        return (
            torch.tensor(self.data[i][np.newaxis], dtype=torch.float32),
            int(self.labels[i][0])
        )
    

class ConvolutionalBlock(nn.Module):
    """Convolutional block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_shape: tuple[int, int],
        pool_type: Literal["max", "adaptive_average"] = "max",
        kernel_size: int = 3,
        padding: int = 1,
        p: float = 0.3,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.pool = (
            nn.MaxPool2d(kernel_size=pool_shape)
            if pool_type == "max"
            else nn.AdaptiveAvgPool2d(output_size=pool_shape)
        )
        self.drop = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = self.conv(x)
        _x = self.norm(_x)
        _x = F.relu(_x)
        _x = self.pool(_x)
        return self.drop(_x)


class CNN(nn.Module):
    """Convolutional network."""

    def __init__(
        self,
        num_classes: int,
        pool_shape: tuple[int, int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        pool_shape = pool_shape or (2, 1)

        self.block1 = ConvolutionalBlock(
            in_channels=1,
            out_channels=32,
            pool_shape=pool_shape,
            p=dropout,
        )
        self.block2 = ConvolutionalBlock(
            in_channels=32,
            out_channels=64,
            pool_shape=pool_shape,
            p=dropout,
        )
        self.block3 = ConvolutionalBlock(
            in_channels=64,
            out_channels=128,
            pool_shape=(1, 1),
            pool_type="adaptive_average",
            p=0.4,
        )
        self.linear = nn.Linear(
            in_features=128, out_features=num_classes,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = self.block1(x)
        _x = self.block2(_x)
        _x = self.block3(_x)
        _x = _x.view(_x.size(0), -1)
        return self.linear(_x)
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward(x)
        return torch.argmax(y, dim=1)
    

def get_scaler() -> StandardScaler:
    train_data = np.load("data/mfcc_cnn/train_mfcc_cnn.npy", allow_pickle=True)
    X_train = np.array([sample[0] for sample in train_data])

    _, _, f_dim = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, f_dim)
    scaler.fit(X_train_flat)

    return scaler


def get_dataloaders(
    scaler: StandardScaler,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, int]:
    y_train = np.load("data/mfcc_cnn/train_labels_binned.npy")
    num_classes = len(np.unique(y_train))

    train_dataset = CryDataset(
        npy_path="data/mfcc_cnn/train_mfcc_cnn.npy",
        binned_label_path="data/mfcc_cnn/train_labels_binned.npy",
        scaler=scaler,
    )
    test_dataset = CryDataset(
        npy_path="data/mfcc_cnn/test_mfcc_cnn.npy",
        binned_label_path="data/mfcc_cnn/test_labels_binned.npy",
        scaler=scaler,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, num_classes


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> tuple[float, float]:
    model.eval()
    all_preds: list[npt.NDArray] = []
    all_true: list[npt.NDArray] = []
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        preds = model.predict(batch_X)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(batch_y.cpu().numpy())
    return (
        accuracy_score(y_true=all_true, y_pred=all_preds),
        f1_score(y_true=all_true, y_pred=all_preds, average="weighted"),
    )


def train_once(
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    pool_shape: tuple[int, int] = (2, 1),
    learning_rate: float = 1e-3,
    dropout: float = 0.3,
    num_epochs: int = 50,
    device: str = "cpu",
) -> tuple[list[float], list[float], float, float]:
    model = CNN(
        num_classes=num_classes,
        pool_shape=pool_shape,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses: list[float] = []
    val_accuracies: list[float] = []
    val_f1s: list[float] = []

    for _ in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logit = model(batch_X)
            loss = criterion(logit, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        train_losses.append(epoch_loss / len(train_loader.dataset))

        acc, f1 = evaluate(model, test_loader, device=device)
        val_accuracies.append(acc)
        val_f1s.append(f1)

    print(val_accuracies)
    print(val_f1s)
    final_val_acc = val_accuracies[-1]
    final_val_f1 = val_f1s[-1]
    return train_losses, val_accuracies, final_val_acc, final_val_f1

def sweep(
    learning_rates: Iterable[float],
    batch_sizes: Iterable[int],
    dropouts: Iterable[float],
    num_epochs: int = 15,
    device: str = "cpu",
) -> None:
    scaler = get_scaler()
    results: list[dict] = []

    hyperparams = product(learning_rates, batch_sizes, dropouts)

    for lr, batch_size, dropout in tqdm(hyperparams, desc="Sweeping hyperparameters"):
        print(f"\nConfig: lr={lr}, batch_size={batch_size}, dropout={dropout}.")

        train_loader, test_loader, num_classes = get_dataloaders(scaler, batch_size)
        _, _, val_acc, val_f1 = train_once(
            train_loader=train_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            learning_rate=lr,
            dropout=dropout,
            num_epochs=num_epochs,
            device=device,
        )
        results.append(
            {
                "learning_rate": lr,
                "batch_size": batch_size,
                "dropout": dropout,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }
        )

        print(f"\nVal acc: {val_acc:.4f}")

    df_path = "models/results/mfcc_cnn/sweep.csv"
    df = pd.DataFrame.from_dict(results)
    df.to_csv(df_path)


if __name__ == "__main__":
    lr_grid = [3e-4, 1e-3, 3e-3]
    batch_size_grid = [16, 32, 64]
    dropout_grid = [0.2, 0.3, 0.5]

    sweep(
        learning_rates=lr_grid,
        batch_sizes=batch_size_grid,
        dropouts=dropout_grid,
        num_epochs=15,
        device="cpu",
    )
