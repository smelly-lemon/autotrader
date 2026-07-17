"""Sequence-based models for temporal pattern recognition.

Provides LSTM and GRU models that consume sliding windows of bar features
instead of independent single-bar rows, learning temporal patterns like
"volume declining for 3 bars then spiking" that tree models cannot see.

Also provides the adaptive retraining infrastructure for online learning.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """Converts a tabular DataFrame into sliding windows per product_id.

    Each sample is a (seq_len, n_features) tensor with the target being the
    label at the final bar of the window. Windows are built per-pair to avoid
    mixing time series from different assets.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        product_ids: np.ndarray,
        timestamps: np.ndarray,
        seq_len: int = 20,
    ):
        self.seq_len = seq_len
        self.sequences = []
        self.targets = []

        unique_pids = np.unique(product_ids)
        for pid in unique_pids:
            mask = product_ids == pid
            X_pid = X[mask]
            y_pid = y[mask]
            ts_pid = timestamps[mask]

            # Sort by time within this pair
            order = np.argsort(ts_pid)
            X_pid = X_pid[order]
            y_pid = y_pid[order]

            # Create sliding windows
            for i in range(seq_len, len(X_pid)):
                window = X_pid[i - seq_len:i]
                if not np.any(np.isnan(window)):
                    self.sequences.append(window.astype(np.float32))
                    self.targets.append(float(y_pid[i]))

        logger.info("  SequenceDataset: %d sequences (seq_len=%d) from %d pairs",
                     len(self.sequences), seq_len, len(unique_pids))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx]),
            torch.tensor(self.targets[idx]),
        )


class LSTMPredictor(nn.Module):
    """LSTM with dropout for binary direction prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, n_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # take final timestep
        return self.head(last_hidden).squeeze(-1)


class GRUPredictor(nn.Module):
    """GRU variant — fewer parameters, sometimes better on small data."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.head(gru_out[:, -1, :]).squeeze(-1)


def train_sequence_model(
    train_dataset: SequenceDataset,
    val_dataset: SequenceDataset | None = None,
    input_dim: int = 52,
    model_type: str = "lstm",
    hidden_dim: int = 64,
    n_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    patience: int = 10,
    device: str = "cpu",
) -> nn.Module:
    """Train an LSTM or GRU model with early stopping."""
    if model_type == "gru":
        model = GRUPredictor(input_dim, hidden_dim, n_layers, dropout)
    else:
        model = LSTMPredictor(input_dim, hidden_dim, n_layers, dropout)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCELoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(X_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train = train_loss / max(n_batches, 1)

        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    pred = model(X_batch)
                    val_loss += criterion(pred, y_batch).item()
                    n_val += 1

            avg_val = val_loss / max(n_val, 1)
            scheduler.step(avg_val)

            if epoch % 5 == 0:
                logger.info("    epoch %d/%d: train=%.4f val=%.4f (best=%.4f)",
                            epoch, epochs, avg_train, avg_val, best_val_loss)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                logger.info("    early stop at epoch %d (patience=%d)", epoch, patience)
                break
        else:
            if avg_train < best_val_loss:
                best_val_loss = avg_train
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_sequence_model(
    model: nn.Module,
    dataset: SequenceDataset,
    device: str = "cpu",
    batch_size: int = 128,
) -> np.ndarray:
    """Run inference and return predicted probabilities."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])


def save_model(model: nn.Module, path: str | Path, metadata: dict | None = None):
    """Save model weights and optional metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model_state": model.state_dict()}
    if metadata:
        state["metadata"] = metadata
    # Save model class info for reconstruction
    if isinstance(model, LSTMPredictor):
        state["model_type"] = "lstm"
    elif isinstance(model, GRUPredictor):
        state["model_type"] = "gru"
    state["input_dim"] = model.lstm.input_size if hasattr(model, "lstm") else model.gru.input_size
    state["hidden_dim"] = model.lstm.hidden_size if hasattr(model, "lstm") else model.gru.hidden_size
    state["n_layers"] = model.lstm.num_layers if hasattr(model, "lstm") else model.gru.num_layers
    torch.save(state, path)
    logger.info("Saved model to %s", path)


def load_model(path: str | Path, device: str = "cpu") -> nn.Module:
    """Load a saved sequence model."""
    state = torch.load(path, map_location=device, weights_only=False)
    model_type = state.get("model_type", "lstm")
    input_dim = state["input_dim"]
    hidden_dim = state["hidden_dim"]
    n_layers = state["n_layers"]

    if model_type == "gru":
        model = GRUPredictor(input_dim, hidden_dim, n_layers)
    else:
        model = LSTMPredictor(input_dim, hidden_dim, n_layers)

    model.load_state_dict(state["model_state"])
    model.eval()
    return model
