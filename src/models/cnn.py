# src/models/cnn.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import os

print("Current working directory:", os.getcwd())
print("Files in mfcc_cnn folder:", os.listdir("data/mfcc_cnn"))

train = np.load('data/mfcc_cnn/train_mfcc_cnn.npy', allow_pickle=True)
test = np.load('data/mfcc_cnn/test_mfcc_cnn.npy', allow_pickle=True)

X_train = np.array([sample[0] for sample in train])
y_train_raw = np.array([sample[1][0] if isinstance(sample[1], (list, np.ndarray)) else sample[1] for sample in train])
X_test = np.array([sample[0] for sample in test])
y_test_raw = np.array([sample[1][0] if isinstance(sample[1], (list, np.ndarray)) else sample[1] for sample in test])

# Feature normalization
num_samples, timesteps, features = X_train.shape
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, features)
X_train_scaled = scaler.fit_transform(X_train_flat).reshape(num_samples, timesteps, features)
X_test_flat = X_test.reshape(-1, features)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)
X_train, X_test = X_train_scaled, X_test_scaled

# Quartile-based bins for balanced classes
quantiles = np.percentile(y_train_raw, [0, 3, 30, 100])
bins = [-np.inf, quantiles[1], quantiles[2], np.inf]
y_train = np.digitize(y_train_raw, bins=bins) - 1
y_test = np.digitize(y_test_raw, bins=bins) - 1

y_train = np.array(y_train).flatten().astype(int)
y_test = np.array(y_test).flatten().astype(int)

print("Chosen bins:", bins)
print("Counts per bin:", np.bincount(y_train))
num_classes = len(np.unique(y_train))
print("Detected num_classes (bins):", num_classes)

y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)
print("y_train_cat shape:", y_train_cat.shape)

if len(X_train.shape) == 2:
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
print("X_train shape after expansion:", X_train.shape)

model = Sequential([
    Conv1D(64, 5, activation='relu', input_shape=X_train.shape[1:]),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),
    Conv1D(128, 5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
print("Model defined")

early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
history = model.fit(
    X_train, y_train_cat,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test_cat),
    callbacks=[early_stop]
)
loss, acc = model.evaluate(X_test, y_test_cat)
print(f"Test accuracy: {acc*100:.2f}%")

if not os.path.exists('models'):
    os.makedirs('models')
model.save('models/baby_cry_cnn_mfcc.h5')
print('Model saved at models/baby_cry_cnn_mfcc.h5')

# Visualization: Save test/train accuracy plot
plt.figure()
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Test Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig('models/test_accuracy_plot.png')
print("Saved accuracy plot as models/test_accuracy_plot.png")

# Visualization: Save test/train loss plot
plt.figure()
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Test Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig('models/test_loss_plot.png')
print("Saved loss plot as models/test_loss_plot.png")
