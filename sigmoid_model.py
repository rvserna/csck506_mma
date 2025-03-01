'''
Sigmoid Activation function model

This model implements a fully connected neural network with 2 hidden layers
It uses:
- Stochastic Gradient Descent (SGD) as the optimizer
- Sigmoid as the activation function
- Cross-entropy as the error function
- One-hot encoding for categorical labels
'''

# Import necessary libraries and modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define file paths for datasets (training and test sets)
train_file_path = r"C:\my-project\csck506_mma\csck506_mma\archive\fashion-mnist_train.csv"
test_file_path = r"C:\my-project\csck506_mma\csck506_mma\archive\fashion-mnist_test.csv"

# Load datasets
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Split features (pixel values) and labels
X_train = train_data.iloc[:, 1:].values  # Extract pixel values
y_train = train_data.iloc[:, 0].values   # Extract labels

X_test = test_data.iloc[:, 1:].values    # Extract pixel values
y_test = test_data.iloc[:, 0].values     # Extract labels

# Normalize pixel values (scale between 0 and 1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert labels to categorical (one-hot encoding)
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Define model using Sigmoid
model = Sequential([
    Dense(128, activation='sigmoid', input_shape=(X_train.shape[1],)),  # First hidden layer
    Dense(64, activation='sigmoid'),  # Second hidden layer
    Dense(10, activation='softmax')   # Output layer (10 classes)
])

# Compile model
model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model with epoch=10 and batch size=1000
history = model.fit(X_train, y_train, epochs=10, batch_size=1000, validation_data=(X_test, y_test))

# Evaluate model
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

# Print loss and accuracy results
print(f"Final Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Plot training and validation loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train vs Test Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train vs Test Accuracy')
plt.legend()

plt.show()