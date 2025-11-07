import sys
import os
# ensure project root parent is on sys.path so package imports resolve when running file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Try to import optional dependencies for standalone usage
try:
    from realtime_anomaly_project.config.settings import TICKERS
except ImportError:
    TICKERS = []

try:
    from realtime_anomaly_project.data_ingestion.yahoo import data_storage
except ImportError:
    data_storage = None

# Define the Transformer Autoencoder model
class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2):
        super(TransformerAutoencoder, self).__init__()
        
        # Project input to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )

        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Input shape: [batch_size, input_dim]
        # Project to hidden dimension
        x = self.input_projection(x)  # Shape: [batch_size, hidden_dim]
        
        # Need to add sequence dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Shape: [batch_size, 1, hidden_dim]
        
        # Transformer expects [sequence_length, batch_size, hidden_dim]
        x = x.transpose(0, 1)  # Shape: [1, batch_size, hidden_dim]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Return to [batch_size, input_dim]
        decoded = decoded.transpose(0, 1).squeeze(1)
        return decoded

# Function to train the model
def train_model(model, data, num_epochs=50, batch_size=32, learning_rate=1e-3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        # Train in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            if batch.shape[0] < batch_size:
                continue

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")

# Function to calculate reconstruction error
def compute_reconstruction_error(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data)
        # Calculate MSE for each sample (average across features)
        error = torch.mean((data - output) ** 2, dim=-1)
        # If error is 0-D (single value), convert to array
        if error.dim() == 0:
            error = error.unsqueeze(0)
    return error.numpy()

# Function to detect anomalies based on reconstruction error
def detect_anomalies(ticker, model, data, threshold=0.5):
    error = compute_reconstruction_error(model, data)
    anomalies = error > threshold
    print(f"Anomalies for {ticker}: {np.sum(anomalies)} anomalies detected.")
    return anomalies

# Function to fit the Transformer Autoencoder on stock data
def fit_transformer(ticker):
    df = data_storage.get(ticker)
    if df is None or df.empty:
        print(f"No data for {ticker}, skipping transformer model training.")
        return None

    # Prepare data (use closing prices for anomaly detection)
    close_prices = df["close"].values.reshape(-1, 1)
    close_prices = torch.tensor(close_prices, dtype=torch.float32)

    # Define the model
    model = TransformerAutoencoder(input_dim=1)

    # Train the model
    train_model(model, close_prices)

    # Detect anomalies based on reconstruction error
    anomalies = detect_anomalies(ticker, model, close_prices)
    return anomalies

def compute_deep_anomalies():
    deep_anomalies = {}

    for ticker in TICKERS:
        anomalies = fit_transformer(ticker)
        deep_anomalies[ticker] = anomalies

    return deep_anomalies

if __name__ == "__main__":
    deep_anomalies = compute_deep_anomalies()
    print(deep_anomalies)
