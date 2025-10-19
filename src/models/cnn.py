import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

class CryDataset(Dataset):
    def __init__(self, npy_path, binned_label_path, scaler=None):
        arr = np.load(npy_path, allow_pickle=True)
        self.X = np.array([sample[0] for sample in arr])
        self.y = np.load(binned_label_path)
        if scaler is not None:
            shape = self.X.shape
            self.X = scaler.transform(self.X.reshape(-1, shape[-1])).reshape(shape)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        # Add channel dimension: (1, time, freq)
        return torch.tensor(self.X[i][np.newaxis], dtype=torch.float32), int(self.y[i])

# Normalize and bin labels
train_data = np.load("data/mfcc_cnn/train_mfcc_cnn.npy", allow_pickle=True)
test_data = np.load("data/mfcc_cnn/test_mfcc_cnn.npy", allow_pickle=True)
X_train = np.array([sample[0] for sample in train_data])
y_train_raw = np.array([sample[1][0] if isinstance(sample[1], (list, np.ndarray)) else sample[1] for sample in train_data])
X_test = np.array([sample[0] for sample in test_data])
y_test_raw = np.array([sample[1][0] if isinstance(sample[1], (list, np.ndarray)) else sample[1] for sample in test_data])

num_samples, t_dim, f_dim = X_train.shape
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, f_dim)
scaler.fit(X_train_flat)

quantiles = np.percentile(y_train_raw, [0, 2, 8, 100])
bins = [-np.inf, quantiles[1], quantiles[2], np.inf]
y_train_binned = np.digitize(y_train_raw, bins=bins) - 1
y_test_binned = np.digitize(y_test_raw, bins=bins) - 1
num_classes = len(np.unique(y_train_binned))
np.save("data/mfcc_cnn/train_labels_binned.npy", y_train_binned)
np.save("data/mfcc_cnn/test_labels_binned.npy", y_test_binned)

batch_size = 32
train_set = CryDataset("data/mfcc_cnn/train_mfcc_cnn.npy", "data/mfcc_cnn/train_labels_binned.npy", scaler)
test_set = CryDataset("data/mfcc_cnn/test_mfcc_cnn.npy", "data/mfcc_cnn/test_labels_binned.npy", scaler)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)

# Determine the pooling shape
pool_shape = (2, 1) if f_dim == 1 else (2, 2)

class CNN2D(nn.Module):
    def __init__(self, num_classes, pool_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(pool_shape)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(pool_shape)
        self.drop2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.globpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop3 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.globpool(F.relu(self.bn3(self.conv3(x)))))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN2D(num_classes, pool_shape=pool_shape).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
num_epochs = 50
train_losses, val_accuracies = [], []

for epoch in range(num_epochs):
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

    # Validation
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = torch.argmax(model(batch_X), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(batch_y.cpu().numpy())
    acc = accuracy_score(all_true, all_preds)
    val_accuracies.append(acc)
    print(f"Epoch {epoch+1}: Loss={train_losses[-1]:.4f}, Val Acc={acc:.4f}")

# Final metrics, save
f1 = f1_score(all_true, all_preds, average='weighted')
print(f"Final Test Accuracy: {val_accuracies[-1]*100:.3f}%, F1 Score: {f1:.3f}")
if not os.path.exists('models'): os.makedirs('models')
torch.save(model.state_dict(), 'models/baby_cry_cnn_mel.pt')
print("Model saved as models/baby_cry_cnn_mel.pt")

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.legend(); plt.xlabel('Epoch'); plt.title('Train History')
plt.tight_layout(); plt.savefig('models/train_val_plot.png')
print("Saved train_val_plot.png")

# Example inference