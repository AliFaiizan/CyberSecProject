import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Utility: activation factory ----------

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


# ---------- VAE model (MLP encoder + 2 decoders) ----------

class VAE(nn.Module):
    """
    Demo VAE with:
      - 3 hidden layers in encoder
      - Reconstruction decoder (for VAE)
      - Classification decoder (for labels)
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims=(128, 64, 32),
        activation="relu",
        num_classes: int = None,  # needed for classification decoder
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = get_activation(activation)
        self.num_classes = num_classes

        # ----- Encoder -----
        encoder_layers = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev, h))
            encoder_layers.append(self.activation)
            prev = h
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent parameters
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # ----- Decoder A: reconstruction -----
        decoder_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev, h))
            decoder_layers.append(self.activation)
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))  # output: same dim as input
        self.decoder_recon = nn.Sequential(*decoder_layers)

        # ----- Decoder B: classification -----
        if num_classes is not None:
            self.decoder_class = nn.Sequential(
                nn.Linear(latent_dim, hidden_dims[-1]),
                self.activation,
                nn.Linear(hidden_dims[-1], num_classes)
            )
        else:
            self.decoder_class = None

    def encode(self, x):
        """
        x: [batch_size, input_dim]
        returns: mu, logvar
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_recon(self, z):
        """
        Reconstruction decoder.
        """
        return self.decoder_recon(z)

    def decode_class(self, z):
        """
        Classification decoder (logits).
        """
        if self.decoder_class is None:
            raise RuntimeError("Classification decoder not initialized (num_classes=None).")
        return self.decoder_class(z)

    def forward(self, x, mode="reconstruction"):
        """
        mode:
          - "reconstruction": returns recon_x, mu, logvar
          - "classification": returns class_logits, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        if mode == "reconstruction":
            recon_x = self.decode_recon(z)
            return recon_x, mu, logvar
        elif mode == "classification":
            logits = self.decode_class(z)
            return logits, mu, logvar
        else:
            raise ValueError(f"Unknown mode: {mode}")


# ---------- Loss functions ----------

def vae_reconstruction_loss(x, recon_x, mu, logvar):
    """
    Classic VAE loss:
      reconstruction (MSE) + KL divergence
    """
    # Reconstruction loss (MSE)
    recon = F.mse_loss(recon_x, x, reduction="sum")

    # KL Divergence term
    # KL(N(mu, sigma) || N(0,1)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (recon + kld) / x.size(0)  # average over batch


def classification_vae_loss(logits, y_true):
    """
    For the classification decoder:
      categorical cross-entropy (CrossEntropyLoss in PyTorch)
    NOTE: Here we *do not* add KL, matching your sheet's text that
          decoder B optimizes categorical cross-entropy.
    """
    ce = F.cross_entropy(logits, y_true)
    return ce


# ---------- Simple training demos ----------

def train_vae_reconstruction(model, dataloader, optimizer, device="cpu", epochs=10):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x in dataloader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x, mode="reconstruction")
            loss = vae_reconstruction_loss(batch_x, recon_x, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"[Recon] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


def train_vae_classification(model, dataloader, optimizer, device="cpu", epochs=10):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits, mu, logvar = model(batch_x, mode="classification")
            loss = classification_vae_loss(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"[Class] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


# ---------- Feature extraction after training ----------

def extract_latent_features(model, X, device="cpu", batch_size=1024):
    """
    X: numpy array [num_samples, input_dim]
    returns: Z (numpy array) [num_samples, latent_dim]
    """
    model.to(device)
    model.eval()
    X_tensor = torch.from_numpy(X).float()
    loader = torch.utils.data.DataLoader(X_tensor, batch_size=batch_size, shuffle=False)

    zs = []
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            mu, logvar = model.encode(batch_x)
            z = model.reparameterize(mu, logvar)
            zs.append(z.cpu().numpy())

    Z = np.concatenate(zs, axis=0)
    return Z


# ---------- Example main (for quick testing) ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, default=86, help="Number of physical readings (M).")
    parser.add_argument("--latent-dim", type=int, default=8, help="Latent feature dimension.")
    parser.add_argument("--activation", type=str, default="relu", help="relu|tanh|leaky_relu")
    parser.add_argument("--mode", type=str, default="reconstruction", help="reconstruction|classification")
    parser.add_argument("--num-classes", type=int, default=5, help="For classification decoder.")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dummy data for demo
    N = 1000
    X = np.random.randn(N, args.input_dim).astype(np.float32)

    if args.mode == "reconstruction":
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X))
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        model = VAE(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            hidden_dims=(128, 64, 32),
            activation=args.activation,
            num_classes=None,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_vae_reconstruction(model, loader, opt, device=device, epochs=args.epochs)

        # Extract latent features
        Z = extract_latent_features(model, X, device=device)
        print("Latent features shape:", Z.shape)

    elif args.mode == "classification":
        y = np.random.randint(0, args.num_classes, size=(N,))
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y).long()
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        model = VAE(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            hidden_dims=(128, 64, 32),
            activation=args.activation,
            num_classes=args.num_classes,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_vae_classification(model, loader, opt, device=device, epochs=args.epochs)

        # Extract latent features (for later ML/CNN classifiers)
        Z = extract_latent_features(model, X, device=device)
        print("Latent features shape:", Z.shape)
    else:
        raise ValueError("Unknown mode.")
