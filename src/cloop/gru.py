"""Small GRU timing model (Portiloop-style), as a scikit-learn regressor.

Reads the feature rows from predict.features as two 25 Hz sequences (slow-wave band,
delta band) plus three per-wave scalars, runs a one-layer GRU over the sequences and
predicts the score-vs-delay curve from the last hidden state + scalars. ~10k weights,
so it is as cheap to run live as the MLP.
"""

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn


class _Net(nn.Module):
    def __init__(self, hidden: int, n_out: int):
        super().__init__()
        self.gru = nn.GRU(2, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + 3, 64), nn.ReLU(), nn.Linear(64, n_out))

    def forward(self, seq, scalars):
        _, h = self.gru(seq)
        return self.head(torch.cat([h[-1], scalars], dim=1))


def _split(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    """Rows -> (sequence, scalars), standardised per channel (not per time step, which
    would blow up the near-constant samples around the trough)."""
    n = (X.shape[1] - 3) // 2
    seq = (np.stack([X[:, :n], X[:, n : 2 * n]], axis=2) - mu[:2]) / sd[:2]  # (batch, time, 2)
    sc = (X[:, -3:] - mu[2:]) / sd[2:]
    return torch.as_tensor(seq, dtype=torch.float32), torch.as_tensor(sc, dtype=torch.float32)


def _channel_stats(X: np.ndarray):
    n = (X.shape[1] - 3) // 2
    cols = [X[:, :n].ravel(), X[:, n : 2 * n].ravel(), *X[:, -3:].T]
    return np.array([c.mean() for c in cols]), np.array([c.std() + 1e-9 for c in cols])


class GRURegressor(RegressorMixin, BaseEstimator):
    def __init__(self, hidden=32, lr=3e-3, weight_decay=1e-4, batch=256, max_epochs=60, patience=6, seed=0):
        self.hidden, self.lr, self.weight_decay = hidden, lr, weight_decay
        self.batch, self.max_epochs, self.patience, self.seed = batch, max_epochs, patience, seed

    def fit(self, X, Y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        # early stopping on the last 10% of rows; rows arrive grouped by recording,
        # so this holds out whole nights rather than neighbouring waves
        n_val = max(len(X) // 10, 1)
        self.mu_, self.sd_ = _channel_stats(X)
        seq, sc = _split(X, self.mu_, self.sd_)
        y = torch.as_tensor(Y, dtype=torch.float32)
        tr, va = slice(0, len(X) - n_val), slice(len(X) - n_val, None)
        self.net_ = _Net(self.hidden, Y.shape[1])
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.net_.train()
            for idx in np.array_split(rng.permutation(len(X) - n_val), max((len(X) - n_val) // self.batch, 1)):
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.net_(seq[tr][idx], sc[tr][idx]), y[tr][idx])
                loss.backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                val = nn.functional.mse_loss(self.net_(seq[va], sc[va]), y[va]).item()
            if val < best - 1e-5:
                best, bad = val, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.net_.eval()
        with torch.no_grad():
            return self.net_(*_split(X, self.mu_, self.sd_)).numpy()
