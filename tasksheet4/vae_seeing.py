import numpy as np

train_data = np.load("synthetic_train.npy")
test_data = np.load("synthetic_test.npy")
test_labels = np.load("synthetic_test_labels.npy")

print("BEFORE COMBINING:")
print(f"train_data: {train_data.shape}")
print(f"test_data: {test_data.shape}")
print(f"test_labels: {test_labels.shape}, attacks: {test_labels.sum()}")

# Simulate VAE's combining
train_labels = np.zeros((train_data.shape[0], 1))
train_data_with_label = np.hstack([train_data, train_labels])
test_data_with_label = np.hstack([test_data, test_labels[:, None] if test_labels.ndim == 1 else test_labels])

all_data = np.vstack([train_data_with_label, test_data_with_label])

X = all_data[:, :-1]
y = all_data[:, -1]

print("\nAFTER COMBINING:")
print(f"X: {X.shape}")
print(f"y: {y.shape}, attacks: {y.sum()}, attack%: {y.mean()*100:.2f}%")

# Simulate train_test_split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print("\nAFTER TRAIN_TEST_SPLIT:")
print(f"X_train: {X_train.shape}, y_train attacks: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"X_val: {X_val.shape}, y_val attacks: {y_val.sum()} ({y_val.mean()*100:.2f}%)")
