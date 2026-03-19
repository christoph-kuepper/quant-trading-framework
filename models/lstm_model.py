import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :]).squeeze(-1)


class LSTMPredictor:
    def __init__(self, lookback: int = 20, hidden_size: int = 64, epochs: int = 50, lr: float = 0.001):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None):
        sequences, targets = [], []
        for i in range(self.lookback, len(X)):
            sequences.append(X[i - self.lookback:i])
            if y is not None:
                targets.append(y[i])
        return np.array(sequences), np.array(targets) if y is not None else None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_train)
        X_seq, y_seq = self._create_sequences(X_scaled, y_train)

        if len(X_seq) < 10:
            logger.warning("Insufficient sequences for LSTM training")
            return

        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        self.model = LSTMNet(X_train.shape[1], self.hidden_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                logger.info(f"LSTM epoch {epoch+1}/{self.epochs}, loss={total_loss/len(loader):.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), 0.5)

        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_scaled)

        if len(X_seq) == 0:
            return np.full(len(X), 0.5)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            proba = self.model(X_tensor).cpu().numpy()

        result = np.full(len(X), 0.5)
        result[self.lookback:self.lookback + len(proba)] = proba
        return result
