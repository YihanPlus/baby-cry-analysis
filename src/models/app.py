import librosa
import numpy as np
import numpy.typing as npt
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


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
    

@st.cache_resource
def load_model(
    num_classes: int,
    pool_shape: tuple[int, int] | None = None,
    path: str = "models/baby_cry_cnn_mel.pt"
) -> CNN2D:
    model = CNN2D(
        num_classes=num_classes, pool_shape=(pool_shape or (2, 1)),
    )
    model.eval()
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def get_scaler() -> StandardScaler:
    train_data = np.load("data/mfcc_cnn/train_mfcc_cnn.npy", allow_pickle=True)
    X_train = np.array([sample[0] for sample in train_data])

    _, _, f_dim = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, f_dim)
    scaler.fit(X_train_flat)

    return scaler


@torch.no_grad
def predict(model: CNN2D, x: npt.NDArray, scaler: StandardScaler) -> int:
    x = x[0]  # (n, m) -> (m, )
    x = x[None, None, ..., None]  # (1, m) -> (1, 1, m, 1)
    x_shape = x.shape
    x = scaler.transform(x.reshape(-1, x_shape[-1])).reshape(x_shape)
    x = torch.tensor(x)
    y = torch.argmax((model.forward(x)), dim=1)
    return y.item()


def prediction_to_label(y: int) -> str:
    if y == 0:
        return "hungry"
    elif y == 1:
        return "burping"
    elif y == 2:
        return "belly pain"


def extract_mfcc(
    file_path: str,
    n_mfcc: int = 40,
    sr: int = 16_000,
    duration: float = 8.0,
) -> npt.NDArray:
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def pad_or_trim_mfcc(mfcc: npt.NDArray, max_frames: int = 173):
    if mfcc.shape[1] < max_frames:
        pad_width = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_frames]
    return mfcc


st.set_page_config(
    page_title="Baby Cry Classifier",
    page_icon="🔊",
    layout="centered"
)
st.title("🔊 Baby Cry Classifier")
st.markdown(
    """
    Upload a `.wav` file and the app will:
    1. Compute MFCCs aligned with the training scaler,  
    2. Run the trained CNN, and  
    3. Show the predicted class with probabilities.
    """
)

model = load_model(num_classes=3)
scaler = get_scaler()
file = st.file_uploader("Upload a WAV file", type=["wav"])

if file:
    mfcc = extract_mfcc(file_path=file)
    mfcc = pad_or_trim_mfcc(mfcc=mfcc)
    y = predict(model=model, x=mfcc, scaler=scaler)
    label = prediction_to_label(y=y)
    st.markdown(f"**Predicted label:** `{label}`")
