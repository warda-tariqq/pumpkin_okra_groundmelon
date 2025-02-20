import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM, Dropout, Reshape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib

# Load the dataset
file_path = "/content/sample_data/updated_pumpkin_okra_groundmelon_dataset.csv"
df = pd.read_csv(file_path)

# Generate a synthetic "Area" feature based on yield with some variation
np.random.seed(42)
df["Area (sq meters)"] = df["Yield (kg)"] / np.random.uniform(1.5, 2.5, size=len(df))

# Select features and target variable
X = df[["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "Soil pH", "NPK Levels", "Area (sq meters)"]]
y = df["Yield (kg)"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for future use
joblib.dump(scaler, "scaler.pkl")

# ANN Model
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
ann_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
ann_model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_data=(X_test_scaled, y_test), verbose=1)
ann_model.save("ann_model.h5")

# CNN Model
cnn_model = Sequential([
    Reshape((X_train_scaled.shape[1], 1), input_shape=(X_train_scaled.shape[1],)),
    Conv1D(filters=32, kernel_size=2, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)
])
cnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
cnn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_data=(X_test_scaled, y_test), verbose=1)
cnn_model.save("cnn_model.h5")

# RNN Model (LSTM)
rnn_model = Sequential([
    Reshape((X_train_scaled.shape[1], 1), input_shape=(X_train_scaled.shape[1],)),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(1)
])
rnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
rnn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_data=(X_test_scaled, y_test), verbose=1)
rnn_model.save("rnn_model.h5")

# Evaluate models
models = {"ANN": ann_model, "CNN": cnn_model, "RNN": rnn_model}
for name, model in models.items():
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=1)
    print(f"{name} Model - Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
