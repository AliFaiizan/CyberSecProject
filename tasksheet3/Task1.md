# Task 1: VAE Feature Extraction

Implements a Variational Autoencoder (VAE) for unsupervised feature extraction from industrial control system time-series data. The VAE compresses 86 sensor features into 8 latent dimensions for downstream anomaly detection.

## Architecture Support

Three encoder types available:

1. **Dense**: Flattened windows `[batch, M×F]`, stacked linear layers (≥3 layers)
2. **Conv1D**: Temporal windows `[batch, M, F]`, 1D convolutions (≥3 layers)
3. **LSTM**: Sequential windows `[batch, M, F]`, LSTM layers (≥3 layers)

## Training Modes

**Reconstruction Mode** (Scenario 1):
```bash
python task1.py --mode reconstruction --epochs 50
```
Loss: MSE + KLD | Output: `task1_dense_relu_ld8_reconstruction_M20.npy`

**Classification Mode** (Scenarios 2 & 3):
```bash
python task1.py --mode classification --epochs 50
```
Loss: MSE + KLD + Cross-Entropy | Output: `task1_dense_relu_ld8_classification_M20.npy`

## Key Parameters

| Parameter | Default | Options |
|-----------|---------|---------|
| `--layer-type` | `dense` | `dense`, `conv1d`, `lstm` |
| `--activation` | `relu` | `relu`, `tanh`, `leaky_relu` |
| `--latent-dim` | `8` | Any integer |
| `--window-size` | `20` | Any integer |
| `--mode` | `reconstruction` | `reconstruction`, `classification` |
| `--epochs` | `50` | Any integer |


## Usage

```bash
# Basic training
python task1.py --mode reconstruction

# Conv1D classifier
python task1.py --layer-type conv1d --mode classification

```

## Output

Files saved to `vae_features/`:
```
task1_{type}_{activation}_ld{dim}_{mode}_M{window}.npy
```
Shape: `(178982, 8)` for default settings

## Integration

```python
# Load for Task 2/3
Z = np.load("vae_features/task1_dense_relu_ld8_reconstruction_M20.npy")

```

