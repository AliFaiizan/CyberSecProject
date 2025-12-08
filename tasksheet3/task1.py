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
- Saves latent features to .npy files per model & fold

You must:
- Provide X_train, X_val, X_test (and y_* for classification)
- Provide scenario & fold logic from Tasksheet 2
"""

import argparse
from glob import glob
import os
from typing import Tuple, List, Dict, Optional

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils import load_data

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
        x: [batch, input_dim]
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
        returns reconstruction [batch, input_dim]
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
        """
        z: [batch, latent_dim]
        returns logits [batch, num_classes]
        """
        return self.net(z)


# ---------------------------------------------------------------------
# Conv1D encoder/decoder
# ---------------------------------------------------------------------

class Conv1dEncoder(nn.Module):
    """
    Input shape: [batch, seq_len] (flattened)
    We reshape to [batch, 1, seq_len] and use Conv1d.
    """
    def __init__(self, input_dim: int, latent_dim: int,
                 activation: str):
        super().__init__()
        act_layer = get_activation(activation)

        self.input_dim = input_dim

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            act_layer,
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            act_layer,
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            act_layer,
        )
        # After Conv: [batch, 64, input_dim]
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * input_dim, latent_dim)
        self.fc_logvar = nn.Linear(64 * input_dim, latent_dim)

    def forward(self, x):
        # x: [batch, input_dim]
        x = x.unsqueeze(1)              # [batch, 1, input_dim]
        h = self.conv(x)                # [batch, 64, input_dim]
        h = self.flatten(h)             # [batch, 64 * input_dim]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Conv1dDecoderRecon(nn.Module):
    """
    Mirror Conv1dEncoder:
    z -> FC -> [batch, 64, input_dim] -> ConvTranspose1d stacks -> [batch, 1, input_dim]
    """
    def __init__(self, input_dim: int, latent_dim: int, activation: str):
        super().__init__()
        act_layer = get_activation(activation)

        self.input_dim = input_dim
        self.fc = nn.Linear(latent_dim, 64 * input_dim)
        self.deconv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            act_layer,
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            act_layer,
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
            # no activation here; treat as regression
        )

    def forward(self, z):
        h = self.fc(z)                       # [batch, 64 * input_dim]
        h = h.view(-1, 64, self.input_dim)   # [batch, 64, input_dim]
        x_rec = self.deconv(h)               # [batch, 1, input_dim]
        x_rec = x_rec.squeeze(1)             # [batch, input_dim]
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
# LSTM encoder/decoder
# ---------------------------------------------------------------------

class LSTMEncoder(nn.Module):
    """
    We treat the input vector as a sequence of length input_dim with 1 feature.
    """
    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_size: int, activation: str):
        super().__init__()
        # activation is used later in FC layers
        self.input_dim = input_dim
        self.activation = get_activation(activation)

        # 3 stacked LSTM layers (>=3 hidden layers)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=False
        )

        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        # x: [batch, input_dim] -> [batch, seq_len, features=1]
        x = x.unsqueeze(-1)
        out, (h_n, c_n) = self.lstm(x)
        # Use final hidden state: [num_layers, batch, hidden_size]
        h_last = h_n[-1]  # [batch, hidden_size]
        h_last = self.activation(h_last)
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar


class LSTMDecoderRecon(nn.Module):
    """
    Sequence decoder:
    z -> FC -> [batch, hidden_size] -> repeat over time -> LSTM -> Dense -> sequence of length input_dim
    """
    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_size: int, activation: str):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.activation = get_activation(activation)

        self.fc_init = nn.Linear(latent_dim, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, z):
        # z: [batch, latent_dim]
        h0 = self.activation(self.fc_init(z))      # [batch, hidden_size]
        # repeat over time dimension 'input_dim'
        h_seq = h0.unsqueeze(1).repeat(1, self.input_dim, 1)  # [batch, seq_len, hidden_size]
        out, _ = self.lstm(h_seq)                  # [batch, seq_len, hidden_size]
        x_rec = self.fc_out(out).squeeze(-1)       # [batch, seq_len]
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
    - one of: DenseEncoder, Conv1dEncoder, LSTMEncoder
    - reconstruction decoder
    - classification decoder (optional)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        layer_type: str,
        activation: str,
        hidden_dims_dense: Tuple[int, ...] = (128, 64, 32),
        conv_used: bool = True,
        lstm_hidden_size: int = 64,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layer_type = layer_type.lower()
        self.activation_name = activation
        self.num_classes = num_classes

        # Encoder / Decoders selection
        if self.layer_type == "dense":
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
                input_dim=input_dim,
                latent_dim=latent_dim,
                activation=activation,
            )
            self.decoder_recon = Conv1dDecoderRecon(
                input_dim=input_dim,
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
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_size=lstm_hidden_size,
                activation=activation,
            )
            self.decoder_recon = LSTMDecoderRecon(
                input_dim=input_dim,
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
        z = self.reparameterize(mu, logvar) # z is sampled latent vector

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

    Tasksheet says:
    - "The first decoder should be able to reconstruct the initial sets of physical readings
       and the Kullback-Leibler divergence (KLD) should be used as a reconstruction error."
    We use the standard VAE objective: reconstruction term + KLD.

    recon_type: "mse" or "mae"
    """
    if recon_type == "mse":
        recon = F.mse_loss(x_rec, x, reduction="sum")
    elif recon_type == "mae":
        recon = F.l1_loss(x_rec, x, reduction="sum")
    else:
        raise ValueError(f"Unknown recon_type: {recon_type}")

    # KL divergence: sum over latent dimensions
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    loss = (recon + kld) / x.size(0)
    return loss


def vae_loss_classification(logits, y_true):
    """
    Classification decoder:
    - Tasksheet: "categorical cross-entropy should be optimized during VAE training."
    - Here we do NOT add KLD, matching that wording.
    """
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
        print(f"[Recon] Epoch {epoch}/{epochs} - train: {avg_train:.4f}, val: {avg_val:.4f}" if avg_val is not None
              else f"[Recon] Epoch {epoch}/{epochs} - train: {avg_train:.4f}")

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
        print(f"[Class] Epoch {epoch}/{epochs} - train: {avg_train:.4f}, val: {avg_val:.4f}" if avg_val is not None
              else f"[Class] Epoch {epoch}/{epochs} - train: {avg_train:.4f}")

    return {"best_val_loss": best_val_loss, "history": history}


def extract_latent_features(
    model: VAE,
    X: np.ndarray,
    device: str,
    batch_size: int = 1024
):
    import psutil
    import time
    # -------------------------
    # Measure memory & time
    # -------------------------
    process = psutil.Process()
    cpu_mem_before = process.memory_info().rss / 1024**2   # MB

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    # -------------------------
    # Standard feature extraction
    # -------------------------
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

            # For debugging:
            # print("latent batch shape:", z.shape)

            zs.append(z.cpu().numpy())

    Z = np.concatenate(zs, axis=0)

    # -------------------------
    # Time & memory after extraction
    # -------------------------
    end_time = time.time()
    cpu_mem_after = process.memory_info().rss / 1024**2

    runtime = end_time - start_time
    cpu_mem_used = cpu_mem_after - cpu_mem_before

    # GPU memory
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
    model.eval()
    X_tensor = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        mu, logvar = model.encode(X_tensor)
        z = model.reparameterize(mu, logvar)
        X_hat = model.decode(z)
    return X_hat.cpu().numpy()
# ---------------------------------------------------------------------
# Simple hyperparameter search for Task 1(c)
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
    Very simple grid-search-style hyperparameter search:
    loops over combinations of layer_type, activation, latent_dim
    and returns the best config based on validation loss.
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
# CLI demo (you will hook this to your scenarios)
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
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.out_dir) 

    # --------------------------------------------------------------
    # LOAD REAL DATA (same for Task 1 and Task 2)
    # --------------------------------------------------------------
    train_files = sorted(glob("../datasets/hai-22.04/train1.csv"))
    test_files  = sorted(glob("../datasets/hai-22.04/test1.csv"))

    X, y = load_data(train_files, test_files)   # X: readings, y: attack labels
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    input_dim = X.shape[1] # number of features
    print(f"[INFO] Loaded dataset with {X.shape[0]} samples and {input_dim} features.")

    # --------------------------------------------------------------
    # TASK 1 SPLIT — NO SCENARIOS
    # --------------------------------------------------------------

    from sklearn.model_selection import train_test_split

    if args.mode == "reconstruction":
        # Train VAE on NORMAL data only
        X_normal = X[y == 0]
        X_train, X_val = train_test_split(
            X_normal, test_size=0.2, random_state=42, shuffle=True
        )
        y_train = y_val = None

    else:  # classification decoder
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

    # --------------------------------------------------------------
    # OPTIONAL: HYPERPARAMETER SEARCH
    # --------------------------------------------------------------
    # if args.hp_search and args.mode == "reconstruction":
    #     best_cfg = hyperparameter_search_vae_recon(
    #         X_train=X_train,
    #         X_val=X_val,
    #         input_dim=input_dim,
    #         layer_types=["dense", "conv1d", "lstm"],
    #         activations=["relu", "tanh", "leaky_relu"],
    #         latent_dims=[4, 8, 16],
    #         epochs=max(3, args.epochs // 2),
    #         batch_size=args.batch_size,
    #         device=device,
    #     )
    #     args.layer_type = best_cfg["layer_type"]
    #     args.activation = best_cfg["activation"]
    #     args.latent_dim = best_cfg["latent_dim"]

    # --------------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------------
    model = VAE(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        layer_type=args.layer_type,
        activation=args.activation,
        num_classes=None if args.mode == "reconstruction" else args.num_classes
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
    # EXTRACT FEATURES FOR THE FULL DATASET
    # --------------------------------------------------------------
    Z_full = extract_latent_features(model, X, device=device)

    # --------------------------------------------------------------
    # SAVE FEATURES (Task 1 output)
    # --------------------------------------------------------------
    base_name = f"task1_{args.layer_type}_{args.activation}_ld{args.latent_dim}_{args.mode}"
    np.save(f"{args.out_dir}/{base_name}.npy", Z_full)

    print(f"[INFO] Saved Task 1 features to {args.out_dir}/{base_name}.npy")

if __name__ == "__main__":
    main()