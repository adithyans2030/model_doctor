"""A small PyTorch multilayer perceptron with a scikit-learn interface.

Optional: only offered when ``torch`` is importable. It receives the same
preprocessed (imputed, scaled, one-hot) matrix as every other algorithm, so it
plugs into the shared pipeline, cross-validation, SHAP/permutation/LIME and the
optimiser unchanged.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

try:  # pragma: no cover - availability depends on the environment
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001
    TORCH_AVAILABLE = False


def _mlp(n_in: int, n_out: int, hidden: tuple[int, ...], dropout: float):
    layers, width = [], n_in
    for h in hidden:
        layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
        width = h
    layers.append(nn.Linear(width, n_out))
    return nn.Sequential(*layers)


class _TorchMLPBase(BaseEstimator):
    def __init__(self, hidden=(64, 32), epochs=60, lr=1e-3, batch_size=128, weight_decay=1e-4,
                 dropout=0.1, random_state=42, n_jobs=1):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _train(self, X, target, loss_fn, n_out, sample_weight):
        torch.manual_seed(self.random_state)
        if self.n_jobs and self.n_jobs > 0:
            torch.set_num_threads(self.n_jobs)
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        w_t = torch.as_tensor(np.ones(len(X_t), dtype=np.float32) if sample_weight is None
                              else np.asarray(sample_weight, dtype=np.float32))
        self.model_ = _mlp(X_t.shape[1], n_out, tuple(self.hidden), self.dropout)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        gen = torch.Generator().manual_seed(self.random_state)
        self.model_.train()
        for _ in range(self.epochs):
            for idx in torch.randperm(len(X_t), generator=gen).split(self.batch_size):
                opt.zero_grad()
                loss = (loss_fn(self.model_(X_t[idx]), target[idx]) * w_t[idx]).sum() / w_t[idx].sum()
                loss.backward()
                opt.step()
        self.model_.eval()
        self.n_features_in_ = X_t.shape[1]
        return self

    def _forward(self, X):
        with torch.no_grad():
            return self.model_(torch.as_tensor(np.asarray(X, dtype=np.float32)))


class TorchMLPClassifier(ClassifierMixin, _TorchMLPBase):
    def fit(self, X, y, sample_weight=None):
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        target = torch.as_tensor(y_idx, dtype=torch.long)
        loss = nn.CrossEntropyLoss(reduction="none")
        return self._train(X, target, loss, len(self.classes_), sample_weight)

    def predict_proba(self, X):
        return torch.softmax(self._forward(X), dim=1).numpy().astype(float)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TorchMLPRegressor(RegressorMixin, _TorchMLPBase):
    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=np.float32)
        # Train on a standardised target so the default learning rate works for any scale.
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() or 1.0)
        target = torch.as_tensor(((y - self.y_mean_) / self.y_std_).reshape(-1, 1))
        loss = nn.MSELoss(reduction="none")
        self._train(X, target, lambda out, t: loss(out, t).squeeze(1), 1, sample_weight)
        return self

    def predict(self, X):
        return self._forward(X).numpy().ravel().astype(float) * self.y_std_ + self.y_mean_
