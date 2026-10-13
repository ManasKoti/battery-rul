"""A small 1D CNN that reads raw ΔQ(V) curves instead of hand-made features."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

LATE_CYCLES = (20, 40, 60, 80, 100)   # one input channel per later cycle
DOWNSAMPLE = 4                        # 1000 voltage points -> 250


def curve_stack(qdlin: np.ndarray, early: int = 10, late_cycles=LATE_CYCLES) -> np.ndarray:
    """Channels = Q_k(V) - Q_early(V) for several k. Shape (len(late_cycles), 250)."""

    base = qdlin[early - 1]
    x = np.stack([qdlin[k - 1] - base for k in late_cycles])[:, ::DOWNSAMPLE]
    return np.nan_to_num(x, nan=0.0).astype(np.float32)


class CurveDataset(Dataset):
    """Each item is (curves, standardized log10 cycle life)."""

    def __init__(self, qdlins, lives, scale, y_mean, y_std, augment=False):
        self.qdlins, self.augment = qdlins, augment
        self.y = ((np.log10(lives) - y_mean) / y_std).astype(np.float32)
        self.scale = scale

    def __len__(self):
        return len(self.qdlins)

    def __getitem__(self, i):
        early = np.random.randint(8, 13) if self.augment else 10
        x = curve_stack(self.qdlins[i], early) / self.scale
        return torch.from_numpy(x), torch.tensor(self.y[i])


class DeltaQCNN(nn.Module):
    def __init__(self, in_ch=len(LATE_CYCLES), width=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=7, padding=3), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(16, width, kernel_size=7, padding=3), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(width, width, kernel_size=7, padding=3), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_cnn(train_qdlins, train_lives, val_qdlins, val_lives, seed=0, epochs=400, lr=1e-3, weight_decay=1e-3, batch_size=8, patience=60, device=None, verbose=False):
    """Train one CNN with early stopping on a validation set of *cells*.""" 

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Normalisation constants come from TRAINING cells only (no peeking at val/test).
    scale = float(np.std([curve_stack(q) for q in train_qdlins])) + 1e-8
    y_log = np.log10(train_lives)
    y_mean, y_std = float(y_log.mean()), float(y_log.std())

    tr = DataLoader(CurveDataset(train_qdlins, train_lives, scale, y_mean, y_std, augment=True), batch_size=batch_size, shuffle=True)
    va = DataLoader(CurveDataset(val_qdlins, val_lives, scale, y_mean, y_std), batch_size=64)

    model = DeltaQCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best, best_state, wait = np.inf, None, 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = np.mean([loss_fn(model(xb.to(device)), yb.to(device)).item() for xb, yb in va])
        if val_loss < best - 1e-4:
            best, wait = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
        if verbose and epoch % 50 == 0:
            print(f"epoch {epoch:4d}  val_loss {val_loss:.4f}")

    model.load_state_dict(best_state)
    model.meta = {"scale": scale, "y_mean": y_mean, "y_std": y_std}
    return model


@torch.no_grad()
def predict_cnn(model, qdlins, device=None):
    device = device or next(model.parameters()).device
    m = model.meta
    x = torch.from_numpy(np.stack([curve_stack(q) for q in qdlins]) / m["scale"]).float().to(device)
    model.eval()
    z = model(x).cpu().numpy()
    return 10 ** (z * m["y_std"] + m["y_mean"])