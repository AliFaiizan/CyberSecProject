import numpy as np

# --- Change these paths to your files ---
file1 = "vae_features/task1_dense_relu_ld8_reconstruction_M20.npy"
file2 = "vae_features/task1_dense_relu_ld8_classification_M20.npy"

# --- Load both latent files ---
print("\nLoading latent files...\n")

Z1 = np.load(file1)
Z2 = np.load(file2)

# --- Print shapes ---
print("============== LATENT FILE 1 ==============")
print(f"File: {file1}")
print(f"Shape: {Z1.shape}")
print(f"Samples (rows): {Z1.shape[0]}")
print(f"Latent dimension: {Z1.shape[1]}")
print("")

print("============== LATENT FILE 2 ==============")
print(f"File: {file2}")
print(f"Shape: {Z2.shape}")
print(f"Samples (rows): {Z2.shape[0]}")
print(f"Latent dimension: {Z2.shape[1]}")
print("")

# Compare sizes
print("============== COMPARISON ==============")
if Z1.shape[0] == Z2.shape[0]:
    print("✔ Both latent files have the SAME number of rows.")
else:
    print("✘ Latent files have DIFFERENT number of rows:")
    print(f"  File1 rows = {Z1.shape[0]}")
    print(f"  File2 rows = {Z2.shape[0]}")
