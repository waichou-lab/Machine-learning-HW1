"""
Homework 1: Machine Learning Practice
Task 1: Fashion MNIST Classification (Pages 298-307)
Task 2: California Housing Regression (Pages 307-313)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

# =============================================================================
# TASK 1: Fashion MNIST Classification (Pages 298-307)
# =============================================================================

print("\n" + "="*60)
print("TASK 1: FASHION MNIST CLASSIFICATION")
print("="*60)

# Load Fashion MNIST dataset
print("\n1. Loading Fashion MNIST dataset...")
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print("X_train_full shape:", X_train_full.shape)
print("X_train_full dtype:", X_train_full.dtype)

# Create validation set and scale pixel intensities
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# Class names for Fashion MNIST
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print("First training instance label:", class_names[y_train[0]])

# Build the model using Sequential API
print("\n2. Building the neural network model...")
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

print("Model summary:")
model.summary()

# Accessing layer information
print("\n3. Examining model layers...")
hidden1 = model.layers[1]
print("First hidden layer name:", hidden1.name)
print("Is get_layer('dense') same as hidden1?", model.get_layer('dense') is hidden1)

# Examine layer weights
weights, biases = hidden1.get_weights()
print("Weights shape:", weights.shape)
print("Biases shape:", biases.shape)
print("Biases initial values (first 10):", biases[:10])

# Compile the model
print("\n4. Compiling the model...")
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

# Train the model
print("\n5. Training the model...")
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

# Plot learning curves
print("\n6. Plotting learning curves...")
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.title("Learning Curves - Fashion MNIST")
plt.savefig('fashion_mnist_learning_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Evaluate on test set
print("\n7. Evaluating on test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Make predictions
print("\n8. Making predictions on new instances...")
X_new = X_test[:3]
y_proba = model.predict(X_new)
print("Predicted probabilities (first instance):")
for i, prob in enumerate(y_proba[0]):
    print(f"  {class_names[i]}: {prob:.2%}")

y_pred = np.argmax(y_proba, axis=1)
print("Predicted classes:", [class_names[i] for i in y_pred])
print("Actual classes:", [class_names[i] for i in y_test[:3]])

# =============================================================================
# TASK 2: California Housing Problem (Pages 307-313)
# =============================================================================

print("\n" + "="*60)
print("TASK 2: CALIFORNIA HOUSING REGRESSION")
print("="*60)

# Load and prepare California housing data
print("\n1. Loading California housing dataset...")
housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_valid.shape)
print("Test set shape:", X_test.shape)

# Build simple Sequential model
print("\n2. Building Sequential regression model...")
seq_model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

seq_model.compile(loss="mean_squared_error", optimizer="sgd")

print("Sequential model summary:")
seq_model.summary()

# Train the Sequential model
print("\n3. Training Sequential model...")
seq_history = seq_model.fit(X_train, y_train, epochs=20,
                           validation_data=(X_valid, y_valid))

# Evaluate Sequential model
print("\n4. Evaluating Sequential model...")
mse_test = seq_model.evaluate(X_test, y_test)
print(f"Test MSE: {mse_test:.4f}")

# Make predictions with Sequential model
X_new = X_test[:3]
y_pred_seq = seq_model.predict(X_new)
print("Sequential model predictions:", y_pred_seq.flatten())
print("Actual values:", y_test[:3])

# Build Wide & Deep model using Functional API
print("\n5. Building Wide & Deep model using Functional API...")

# Input layer
input_ = keras.layers.Input(shape=X_train.shape[1:])
# Deep path
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
# Concatenate wide and deep paths
concat = keras.layers.Concatenate()([input_, hidden2])
# Output layer
output = keras.layers.Dense(1)(concat)

wide_deep_model = keras.Model(inputs=[input_], outputs=[output])

wide_deep_model.compile(loss="mean_squared_error", 
                       optimizer=keras.optimizers.SGD(learning_rate=1e-3))

print("Wide & Deep model summary:")
wide_deep_model.summary()

# Train Wide & Deep model
print("\n6. Training Wide & Deep model...")
wide_deep_history = wide_deep_model.fit(X_train, y_train, epochs=20,
                                       validation_data=(X_valid, y_valid))

# Evaluate Wide & Deep model
print("\n7. Evaluating Wide & Deep model...")
mse_test_wide_deep = wide_deep_model.evaluate(X_test, y_test)
print(f"Wide & Deep Test MSE: {mse_test_wide_deep:.4f}")

# Compare model performance
print("\n8. Model Comparison:")
print(f"Sequential Model Test MSE: {mse_test:.4f}")
print(f"Wide & Deep Model Test MSE: {mse_test_wide_deep:.4f}")

# Plot comparison of training histories
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(seq_history.history['loss'], label='Training Loss')
plt.plot(seq_history.history['val_loss'], label='Validation Loss')
plt.title('Sequential Model - Training History')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(wide_deep_history.history['loss'], label='Training Loss')
plt.plot(wide_deep_history.history['val_loss'], label='Validation Loss')
plt.title('Wide & Deep Model - Training History')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('california_housing_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
