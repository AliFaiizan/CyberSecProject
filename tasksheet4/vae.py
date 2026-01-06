"""
vae_module.py

Tasksheet 3 - Task 1 compliant VAE module.

Framework: PyTorch

Features:
- Supports 3 layer types: dense | conv1d | lstm
- Supports 3 activations: relu | tanh | leaky_relu
- Encoder has >= 3 hidden layers for all layer types
- Two decoder branches:
    * reconstruction: VAE with reconstruction + KLD
    * classification: latent -> class logits (cross-entropy)
- Hyperparameter search over basic VAE configs
- Saves latent features to .npy files per model"""

import argparse
from glob import glob
import os
from typing import Tuple, List, Dict, Optional

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils import load_data, create_windows_for_vae

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    else:
        raise ValueError(f"Unknown activation: {name}")


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------
# Dense encoder/decoder
# ---------------------------------------------------------------------

class DenseEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dims: Tuple[int, ...], activation: str):
        super().__init__()
        act_layer = get_activation(activation)

        layers = []
        prev = input_dim
        # ≥ 3 hidden layers enforced by Tasksheet
        if len(hidden_dims) < 3:
            raise ValueError("DenseEncoder requires at least 3 hidden layers.")
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_layer)
            prev = h
        self.backbone = nn.Sequential(*layers) # this is stack of fully connected layers with activations, through which the input is passes in sequence

        self.fc_mu = nn.Linear(prev, latent_dim) #A linear layer that outputs the mean (mu) of the latent distribution for each input sample.
        self.fc_logvar = nn.Linear(prev, latent_dim) # A linear layer that outputs the log-variance (logvar) of the latent distribution for each input sample.

    def forward(self, x):
        """
        x: [batch, input_dim] = [batch, M*F]
        """
        h = self.backbone(x) # h is final hidden representation after passing through all hidden layers
        mu = self.fc_mu(h) # h passed through fc_mu layer to get mean of latent distribution
        logvar = self.fc_logvar(h) # h passed through fc_logvar layer to get log-variance of latent distribution
        return mu, logvar


class DenseDecoderRecon(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dims: Tuple[int, ...], activation: str):
        super().__init__()
        act_layer = get_activation(activation)

        layers = []
        prev = latent_dim
        # reverse hidden dims for decoder
        for h in reversed(hidden_dims):
            layers.append(nn.Linear(prev, h))
            layers.append(act_layer)
            prev = h
        layers.append(nn.Linear(prev, input_dim))
        self.backbone = nn.Sequential(*layers)

    def forward(self, z):
        """
        z: [batch, latent_dim]
        returns reconstruction [batch, input_dim] (= flattened M*F)
        """
        return self.backbone(z)


class DenseDecoderClass(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int,
                 hidden_dim: int, activation: str):
        super().__init__()
        act_layer = get_activation(activation)

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            act_layer,
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------
# Conv1D encoder/decoder (sequence-aware)
# ---------------------------------------------------------------------

class Conv1dEncoder(nn.Module):
    """
    Input shape: [batch, seq_len, feature_dim] = [batch, M, F]
    We reinterpret it as [batch, F, M] for Conv1d (channels=F, seq_len=M).
    """
    def __init__(self, seq_len: int, feature_dim: int, latent_dim: int,
                 activation: str):
        super().__init__()
        act_layer = get_activation(activation)

        self.seq_len = seq_len
        self.feature_dim = feature_dim

        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, 32, kernel_size=3, padding=1),
            act_layer,
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            act_layer,
        )
        # After Conv: [batch, 64, seq_len]
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * seq_len, latent_dim)
        self.fc_logvar = nn.Linear(64 * seq_len, latent_dim)

    def forward(self, x):
        """
        x: [batch, seq_len, feature_dim] = [batch, M, F]
        """
        x = x.permute(0, 2, 1)            # [batch, F, M]
        h = self.conv(x)                  # [batch, 64, M]
        h = self.flatten(h)               # [batch, 64*M]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Conv1dDecoderRecon(nn.Module):
    """
    Mirror Conv1dEncoder:
    z -> FC -> [batch, 64, seq_len] -> Conv1d -> [batch, F, seq_len] -> permute back to [batch, seq_len, F].
    """
    def __init__(self, seq_len: int, feature_dim: int,
                 latent_dim: int, activation: str):
        super().__init__()
        act_layer = get_activation(activation)

        self.seq_len = seq_len
        self.feature_dim = feature_dim

        self.fc = nn.Linear(latent_dim, 64 * seq_len)
        self.deconv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            act_layer,
            nn.Conv1d(32, feature_dim, kernel_size=3, padding=1),
            # no activation; regression output
        )

    def forward(self, z):
        h = self.fc(z)                                # [batch, 64 * M]
        h = h.view(-1, 64, self.seq_len)             # [batch, 64, M]
        x_rec = self.deconv(h)                       # [batch, F, M]
        x_rec = x_rec.permute(0, 2, 1)               # [batch, M, F]
        return x_rec


class Conv1dDecoderClass(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int,
                 activation: str):
        super().__init__()
        act_layer = get_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            act_layer,
            nn.Linear(64, num_classes)
        )

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------
# LSTM encoder/decoder (sequence-aware)
# ---------------------------------------------------------------------

class LSTMEncoder(nn.Module):
    """
    Input: [batch, seq_len, feature_dim] = [batch, M, F]
    We feed this directly into LSTM with input_size = F.
    """
    def __init__(self, seq_len: int, feature_dim: int,
                 latent_dim: int, hidden_size: int, activation: str):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.activation = get_activation(activation)

        # 3 stacked LSTM layers (>=3 hidden layers)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=False
        )

        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        """
        x: [batch, M, F]
        """
        out, (h_n, c_n) = self.lstm(x)     # h_n: [num_layers, batch, hidden_size]
        h_last = h_n[-1]                   # [batch, hidden_size]
        h_last = self.activation(h_last)
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar


class LSTMDecoderRecon(nn.Module):
    """
    Sequence decoder:
    z -> FC -> [batch, hidden_size]
      -> repeat over time (M) -> LSTM -> FC_out -> [batch, M, F]
    """
    def __init__(self, seq_len: int, feature_dim: int,
                 latent_dim: int, hidden_size: int, activation: str):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.activation = get_activation(activation)

        self.fc_init = nn.Linear(latent_dim, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size, feature_dim)

    def forward(self, z):
        """
        z: [batch, latent_dim]
        returns: [batch, M, F]
        """
        h0 = self.activation(self.fc_init(z))             # [batch, hidden_size]
        h_seq = h0.unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch, M, hidden_size]
        out, _ = self.lstm(h_seq)                         # [batch, M, hidden_size]
        x_rec = self.fc_out(out)                          # [batch, M, F]
        return x_rec


class LSTMDecoderClass(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int,
                 hidden_size: int, activation: str):
        super().__init__()
        act_layer = get_activation(activation)

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            act_layer,
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------
# Main VAE wrapper
# ---------------------------------------------------------------------

class VAE(nn.Module):
    """
    Flexible VAE that wraps:
    - DenseEncoder / Conv1dEncoder / LSTMEncoder
    - reconstruction decoder
    - classification decoder (optional)

    For Dense:
      input: [batch, M*F]
    For Conv1d / LSTM:
      input: [batch, M, F]
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        layer_type: str,
        activation: str,
        hidden_dims_dense: Tuple[int, ...] = (128, 64, 32),
        lstm_hidden_size: int = 64,
        num_classes: Optional[int] = None,
        seq_len: Optional[int] = None,
        feature_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim        # used for dense
        self.latent_dim = latent_dim
        self.layer_type = layer_type.lower()
        self.activation_name = activation
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        if self.layer_type in ("conv1d", "lstm"):
            if self.seq_len is None or self.feature_dim is None:
                raise ValueError("seq_len and feature_dim must be provided for conv1d/lstm VAE.")

        if self.layer_type == "dense":
            # input_dim == M * F (flattened)
            self.encoder = DenseEncoder(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims_dense,
                activation=activation,
            )
            self.decoder_recon = DenseDecoderRecon(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims_dense,
                activation=activation,
            )
            if num_classes is not None:
                self.decoder_class = DenseDecoderClass(
                    latent_dim=latent_dim,
                    num_classes=num_classes,
                    hidden_dim=hidden_dims_dense[-1],
                    activation=activation,
                )
            else:
                self.decoder_class = None

        elif self.layer_type == "conv1d":
            self.encoder = Conv1dEncoder(
                seq_len=self.seq_len,
                feature_dim=self.feature_dim,
                latent_dim=latent_dim,
                activation=activation,
            )
            self.decoder_recon = Conv1dDecoderRecon(
                seq_len=self.seq_len,
                feature_dim=self.feature_dim,
                latent_dim=latent_dim,
                activation=activation,
            )
            if num_classes is not None:
                self.decoder_class = Conv1dDecoderClass(
                    latent_dim=latent_dim,
                    num_classes=num_classes,
                    activation=activation,
                )
            else:
                self.decoder_class = None

        elif self.layer_type == "lstm":
            self.encoder = LSTMEncoder(
                seq_len=self.seq_len,
                feature_dim=self.feature_dim,
                latent_dim=latent_dim,
                hidden_size=lstm_hidden_size,
                activation=activation,
            )
            self.decoder_recon = LSTMDecoderRecon(
                seq_len=self.seq_len,
                feature_dim=self.feature_dim,
                latent_dim=latent_dim,
                hidden_size=lstm_hidden_size,
                activation=activation,
            )
            if num_classes is not None:
                self.decoder_class = LSTMDecoderClass(
                    latent_dim=latent_dim,
                    num_classes=num_classes,
                    hidden_size=lstm_hidden_size,
                    activation=activation,
                )
            else:
                self.decoder_class = None
        else:
            raise ValueError(f"Unknown layer_type: {self.layer_type}")

    # --- VAE core ---

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_recon(self, z):
        return self.decoder_recon(z)

    def decode_class(self, z):
        if self.decoder_class is None:
            raise RuntimeError("Classification decoder not initialized (num_classes not set).")
        return self.decoder_class(z)

    def forward(self, x, mode: str = "reconstruction"):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        if mode == "reconstruction":
            x_rec = self.decode_recon(z)
            return x_rec, mu, logvar
        elif mode == "classification":
            logits = self.decode_class(z)
            return logits, mu, logvar
        else:
            raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------

def vae_loss_reconstruction(x, x_rec, mu, logvar, recon_type: str = "mse"):
    """
    Reconstruction loss + KL divergence.

    
    - Dense: x, x_rec shape [batch, M*F]
    - Conv1d/LSTM: x, x_rec shape [batch, M, F]
    """
    if recon_type == "mse":
        recon = F.mse_loss(x_rec, x, reduction="sum")
    elif recon_type == "mae":
        recon = F.l1_loss(x_rec, x, reduction="sum")
    else:
        raise ValueError(f"Unknown recon_type: {recon_type}")

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = (recon + kld) / x.size(0)
    return loss


def vae_loss_classification(logits, y_true):
    return F.cross_entropy(logits, y_true)


# ---------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------

def train_vae_reconstruction(
    model: VAE,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: str,
    epochs: int,
    lr: float = 1e-3,
    recon_type: str = "mse",
) -> Dict[str, float]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train() # switch model to training mode effecting certain layers like dropout, batchnorm etc.
        total_train = 0.0
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device) # move input batch to specified device (CPU/GPU)
            optimizer.zero_grad() # reset gradients of model parameters to zero before backpropagation
            x_rec, mu, logvar = model(x_batch, mode="reconstruction")
            loss = vae_loss_reconstruction(x_batch, x_rec, mu, logvar, recon_type)
            loss.backward()
            optimizer.step()
            total_train += loss.item() * x_batch.size(0)

        avg_train = total_train / len(train_loader.dataset)

        avg_val = None
        if val_loader is not None:
            model.eval()
            total_val = 0.0
            with torch.no_grad():
                for (x_batch,) in val_loader:
                    x_batch = x_batch.to(device)
                    x_rec, mu, logvar = model(x_batch, mode="reconstruction")
                    loss = vae_loss_reconstruction(x_batch, x_rec, mu, logvar, recon_type)
                    total_val += loss.item() * x_batch.size(0)
            avg_val = total_val / len(val_loader.dataset)
            if avg_val < best_val_loss:
                best_val_loss = avg_val

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        if avg_val is not None:
            print(f"[Recon] Epoch {epoch}/{epochs} - train: {avg_train:.4f}, val: {avg_val:.4f}")
        else:
            print(f"[Recon] Epoch {epoch}/{epochs} - train: {avg_train:.4f}")

    return {"best_val_loss": best_val_loss, "history": history}


def train_vae_classification(
    model: VAE,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: str,
    epochs: int,
    lr: float = 1e-3,
) -> Dict[str, float]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_train = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits, mu, logvar = model(x_batch, mode="classification")
            loss = vae_loss_classification(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_train += loss.item() * x_batch.size(0)
        avg_train = total_train / len(train_loader.dataset)

        avg_val = None
        if val_loader is not None:
            model.eval()
            total_val = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    logits, mu, logvar = model(x_batch, mode="classification")
                    loss = vae_loss_classification(logits, y_batch)
                    total_val += loss.item() * x_batch.size(0)
            avg_val = total_val / len(val_loader.dataset)
            if avg_val < best_val_loss:
                best_val_loss = avg_val

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        if avg_val is not None:
            print(f"[Class] Epoch {epoch}/{epochs} - train: {avg_train:.4f}, val: {avg_val:.4f}")
        else:
            print(f"[Class] Epoch {epoch}/{epochs} - train: {avg_train:.4f}")

    return {"best_val_loss": best_val_loss, "history": history}


def extract_latent_features(
    model: VAE,
    X: np.ndarray,
    device: str,
    batch_size: int = 1024
):
    import psutil
    import time
    process = psutil.Process()
    cpu_mem_before = process.memory_info().rss / 1024**2   # MB

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    model.to(device)
    model.eval()
    X_tensor = torch.from_numpy(X).float()
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)

    zs = []
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            mu, logvar = model.encode(x_batch)
            z = model.reparameterize(mu, logvar)
            zs.append(z.cpu().numpy())

    Z = np.concatenate(zs, axis=0)

    end_time = time.time()
    cpu_mem_after = process.memory_info().rss / 1024**2

    runtime = end_time - start_time
    cpu_mem_used = cpu_mem_after - cpu_mem_before

    gpu_mem_used = None
    if device == "cuda":
        gpu_mem_used = torch.cuda.max_memory_allocated() / 1024**2

    print("\n===== FEATURE EXTRACTION METRICS =====")
    print(f"Runtime: {runtime:.4f} seconds")
    print(f"CPU memory used: {cpu_mem_used:.2f} MB")
    if gpu_mem_used is not None:
        print(f"GPU memory used: {gpu_mem_used:.2f} MB")
    print("======================================\n")

    return Z, runtime, cpu_mem_used, gpu_mem_used


def reconstruct_physical_readings(model, X, device="cuda"):
    """
    Reconstruct the inputs X using the VAE in reconstruction mode.

    NOTE: X must have the SAME SHAPE as used in training:
      - Dense: [N, M*F]
      - Conv1d/LSTM: [N, M, F]
    """
    model.to(device)
    model.eval()
    X_tensor = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        mu, logvar = model.encode(X_tensor)
        z = model.reparameterize(mu, logvar)
        X_hat = model.decode_recon(z)
    return X_hat.cpu().numpy()


# ---------------------------------------------------------------------
# (Optional) Hyperparameter search – currently only makes sense for dense.
# ---------------------------------------------------------------------

def hyperparameter_search_vae_recon(
    X_train: np.ndarray,
    X_val: np.ndarray,
    input_dim: int,
    layer_types: List[str],
    activations: List[str],
    latent_dims: List[int],
    epochs: int,
    batch_size: int,
    device: str,
) -> Dict:
    """
    Simple grid search for reconstruction VAE.

    NOTE: This implementation assumes flattened input (Dense).
    For Conv1d/LSTM you'd need to adapt seq_len & feature_dim.
    """
    best_cfg = None
    best_val = float("inf")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float()),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float()),
        batch_size=batch_size,
        shuffle=False,
    )

    for lt in layer_types:
        if lt != "dense":
            continue  # keep it simple: search only dense here
        for act in activations:
            for ld in latent_dims:
                print(f"\n[HP-Search] layer_type={lt}, activation={act}, latent_dim={ld}")
                model = VAE(
                    input_dim=input_dim,
                    latent_dim=ld,
                    layer_type=lt,
                    activation=act,
                    num_classes=None,
                )
                result = train_vae_reconstruction(
                    model,
                    train_loader,
                    val_loader,
                    device=device,
                    epochs=epochs,
                    lr=1e-3,
                    recon_type="mse",
                )
                val_loss = result["best_val_loss"]
                print(f"  -> best_val_loss={val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    best_cfg = {
                        "layer_type": lt,
                        "activation": act,
                        "latent_dim": ld,
                        "val_loss": val_loss,
                    }

    print("\n[HP-Search] Best config:", best_cfg)
    return best_cfg


# ---------------------------------------------------------------------
# CLI demo (hook to Task 1)
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="reconstruction",
                        choices=["reconstruction", "classification"],
                        help="Which decoder to train (Task 1).")
    parser.add_argument("--layer-type", type=str, default="dense",
                        choices=["dense", "conv1d", "lstm"])
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "tanh", "leaky_relu"])
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out-dir", type=str, default="vae_features")
    parser.add_argument("--hp-search", action="store_true")
    parser.add_argument(
        "--window-size", type=int, default=20,
        help="M: number of consecutive physical readings per VAE input set."
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.out_dir)

    # --------------------------------------------------------------
    # LOAD REAL DATA
    # --------------------------------------------------------------
    # train_files = sorted(glob("../datasets/hai-22.04/train1.csv"))
    # test_files  = sorted(glob("../datasets/hai-22.04/test1.csv"))
    # Load train and test data
    train_data = np.load("synthetic_train.npy")  # shape: [N_train, F]
    test_data = np.load("synthetic_test.npy")    # shape: [N_test, F]
    test_labels = np.load("synthetic_test_labels.npy")  # shape: [N_test,] or [N_test, 1]

    # Ensure test_labels is a column vector
    if test_labels.ndim == 1:
        test_labels = test_labels[:, None]

    # Add label column to train (all zeros)
    train_labels = np.zeros((train_data.shape[0], 1))
    train_data_with_label = np.hstack([train_data, train_labels])
    test_data_with_label = np.hstack([test_data, test_labels])

    # Combine
    all_data = np.vstack([train_data_with_label, test_data_with_label])

    # Now, features and labels:
    X = all_data[:, :-1]  # all columns except last
    y = all_data[:, -1]   # last column

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    T, F = X.shape
    print(f"[INFO] Raw dataset: {T} timesteps, {F} features.")

    # --------------------------------------------------------------
    # CREATE WINDOWS (M×N sets of physical readings)
    # --------------------------------------------------------------
    M = args.window_size

    if args.mode == "reconstruction":
        X_win, _ = create_windows_for_vae(X, y, M, mode="reconstruction")
        y_win = None
    else:
        X_win, y_win = create_windows_for_vae(X, y, M, mode="classification")

    print(f"[INFO] Windowed dataset: {X_win.shape[0]} windows, shape per window: {X_win.shape[1:]} (M={M}, F={F})")

    from sklearn.model_selection import train_test_split

    # Prepare data depending on layer type
    if args.layer_type == "dense":
        # Flatten windows: [N, M, F] -> [N, M*F]
        X_flat = X_win.reshape(X_win.shape[0], -1)
        dense_input_dim = X_flat.shape[1]   # = M * F

        if args.mode == "reconstruction":
            X_train, X_val = train_test_split(
                X_flat, test_size=0.2, random_state=42, shuffle=True
            )
            y_train = y_val = None
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_flat, y_win, test_size=0.2, random_state=42, shuffle=True
            )

    else:
        # Conv1d / LSTM: keep windows as [N, M, F]
        dense_input_dim = M * F   # not used for conv1d/lstm but required for VAE init
        if args.mode == "reconstruction":
            X_train, X_val = train_test_split(
                X_win, test_size=0.2, random_state=42, shuffle=True
            )
            y_train = y_val = None
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_win, y_win, test_size=0.2, random_state=42, shuffle=True
            )

    # --------------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------------
    model = VAE(
        input_dim=dense_input_dim,
        latent_dim=args.latent_dim,
        layer_type=args.layer_type,
        activation=args.activation,
        num_classes=None if args.mode == "reconstruction" else args.num_classes,
        seq_len=M,
        feature_dim=F,
    )

    # Data loaders
    if args.mode == "reconstruction":
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train).float()),
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val).float()),
            batch_size=args.batch_size,
            shuffle=False,
        )
    else:
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train).long()
            ),
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(y_val).long()
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )

    # --------------------------------------------------------------
    # TRAIN THE VAE
    # --------------------------------------------------------------
    if args.mode == "reconstruction":
        train_vae_reconstruction(
            model, train_loader, val_loader,
            device=device, epochs=args.epochs
        )
    else:
        train_vae_classification(
            model, train_loader, val_loader,
            device=device, epochs=args.epochs
        )

    # --------------------------------------------------------------
    # EXTRACT FEATURES FOR THE FULL WINDOWED DATASET
    # --------------------------------------------------------------
    # Use the same representation as used for training:
    if args.layer_type == "dense":
        X_for_Z = X_win.reshape(X_win.shape[0], -1)
    else:
        X_for_Z = X_win  # [N, M, F]

    Z_full, runtime, cpu_mem_used, gpu_mem_used = extract_latent_features(
        model, X_for_Z, device=device
    )

    # --------------------------------------------------------------
    # SAVE FEATURES (Task 1 output)
    # --------------------------------------------------------------
    base_name = f"task1_{args.layer_type}_{args.activation}_ld{args.latent_dim}_{args.mode}_M{M}"
    np.save(f"{args.out_dir}/{base_name}.npy", Z_full)

    print(f"[INFO] Saved Task 1 features to {args.out_dir}/{base_name}.npy")


if __name__ == "__main__":
    main()
