# Practical Sheet 4 – Synthetic Data Generation, Experiments, and Error Analysis

## Overview

This practical sheet focuses on improving anomaly detection for Industrial Control Systems (ICS) by introducing synthetic data generation using a Generative Adversarial Network (GAN), executing multiple machine learning and deep learning experiments across different evaluation scenarios, and performing a detailed analysis of prediction errors.  

The work is structured into three main tasks:

- **Task 1:** Generation of synthetic datasets using a custom GAN  
- **Task 2:** Execution of classification experiments using synthetic data  
- **Task 3a:** Analysis of prediction errors using Venn diagrams
- **Task 3b-c:** LIME and SHAP analysison predicted values  

All experiments are implemented in Python and evaluated using k-fold cross-validation.

---

## Structure 
|-datasets
|  - hai22.04
|-tasksheet4
|  - rest of the code

## Task 1: Generation of Synthetic Dataset

### Objective

The goal of Task 1 is to generate a synthetic dataset that closely resembles the normal operation of an Industrial Control System. This synthetic dataset is later used as training data in Task 2 to evaluate whether synthetic data can replace or complement real data.

### Approach

A **Generative Adversarial Network (GAN)** was implemented from scratch. The GAN consists of two components:

- **Generator:** Produces synthetic physical readings
- **Discriminator:** Distinguishes between real and synthetic samples

The GAN is trained only on **normal system behavior** to ensure that the generated data represents benign operation.

### GAN Architecture

#### Generator
- Input: Random noise vector
- 3 fully connected layers with **LeakyReLU** activation
- 9 convolutional layers with **LeakyReLU**
- Upsampling applied before the first 4 convolutional layers
- Output shape matches the original physical readings

#### Discriminator
- Input: Real or synthetic physical readings
- 9 convolutional layers with **ReLU**
- Dropout applied after all convolutional layers except the first and last
- 3 fully connected layers with **ReLU**
- Final output: Single value in range [0,1), representing how real the input is

### Custom Loss Function

Instead of using only binary cross-entropy, a **feature matching loss** is implemented.  
This loss is computed on the input of the first fully connected layer of the discriminator and minimizes the distance between real and synthetic feature representations. This improves training stability and prevents mode collapse.

### Hyperparameter Tuning

A tuning framework is included to experiment with:
- Learning rates
- Number of epochs
- Latent space dimensionality
- Batch size
- Convolutional filter sizes

The best-performing configuration is selected based on discriminator stability and generator output quality.

---

## Task 2: Execution of Experiments

### Objective

The goal of Task 2 is to evaluate multiple anomaly detection and classification methods using **synthetic training data generated in Task 1**, following the evaluation scenarios defined in previous practical sheets.

All experiments use **5-fold cross-validation** and report **precision and recall**.

---

### Data Preparation Pipeline

1. **GAN / VAE**  
   - trained model to be used for data generation.


2. **GenerateFolds**  
   It will use models to export train and test split based on scenerio where training data is synthetic and testing kept as it is `generate_folds.py`:
   `python generate_folds.py -sc <number> -k 5 -M 20 --vae-checkpoint vae_classification_real.pt`

3. **Run Task 2**  
   - `run task2.py` It loads data folds and run experiments

---

### Evaluation Scenarios and Methods

#### Scenario 1
- Training: Normal-only data
- Methods:
  - OCSVM
  - LOF
  - Elliptic Envelope
  - N-gram (N = 2, 5, 8)
- N-gram experiments are executed:
  - Once using real data
  - Once using GAN-generated synthetic data

#### Scenario 2
- Training includes selected attacks
- Methods:
  - SVM
  - kNN
  - Random Forest
  - CNN
- Synthetic data replaces real training samples
- CNN is trained on VAE latent features

#### Scenario 3
- Leave-one-attack-out setting
- Methods:
  - SVM
  - kNN
  - Random Forest
  - CNN
- Synthetic training data generated using GAN
- CNN evaluated with k-fold cross-validation

`task2.py`
---

### N-gram Detection (Scenario 1)

Two separate implementations are used:

- **Task 2d:**  
  N-gram detection using raw physical readings and real training data
 `task2d.py`

- **Task 2e:**  
  N-gram detection using raw physical readings and GAN-generated synthetic training data
`task2e.py`

For each N in {2, 5, 8}:
- Bloom filters are constructed from training sequences
- Anomaly scores are computed
- Thresholds are optimized using F1-score
- Precision and recall are reported per fold

Predictions are saved as:
- `Predictions_N2_FoldX.csv`
- `Predictions_N5_FoldX.csv`
- `Predictions_N8_FoldX.csv`

---

### Result Visualization

All results are visualized using `plot.py`, which generates:
- Precision and recall comparison plots
- Fold-wise performance plots
- Runtime and memory usage plots
- Ensemble comparison plots (included but not emphasized)

All plots are saved in the `exports_sheet4/` directory and submitted as part of the results.

---

## Task 3a: Analysis of Prediction Errors

### Objective

The goal of Task 3a is to analyze **where and how classifiers make mistakes**, rather than only comparing performance metrics.
`task3_venn.py`

### Methodology

For each evaluation scenario:
- Misclassified test instances are extracted from prediction files
- Each classifier contributes a set of error indices
- **Venn diagrams** are generated to visualize overlap of errors

Each circle represents the error set of one classifier:
- ML-based classifiers
- CNN classifier
- N-gram detector

Intersections indicate test instances misclassified by multiple methods.

### Analysis Focus

- Comparison of error overlap between classifiers
- Differences between models trained on real data vs synthetic data
- Identification of complementary detection behavior
- Evaluation of whether synthetic training data leads to more diverse or overlapping errors

### Outputs

- Venn diagrams for Scenario 1
- UpSet plots for Scenarios 2 and 3
- Stored in `exports_sheet4/Scenario*/Venn` and `exports_sheet4/Scenario*/UpSet`

---

## Libraries Used

The following libraries are used across all tasks:

### Core Python Libraries
- `os`
- `sys`
- `math`
- `json`
- `pickle`
- `hashlib`
- `argparse`

### Numerical and Data Processing
- `numpy`
- `pandas`

### Machine Learning
- `scikit-learn`

### Deep Learning
- `torch`
- `torch.nn`
- `torch.optim`
- `tensorflow`
- `keras`

### Visualization
- `matplotlib`
- `seaborn`
- `matplotlib-venn`
- `upsetplot`

### Utilities
- `joblib`
- `warnings`

---

## Commands Reference

### Task 1 – GAN (Synthetic Data Generation)

```bash
python task1.py --epochs 50
python task1.py --M 86 --epochs 50
python task1.py --epochs 50 --tune
```

### Task 2 – Feature Extraction and Classification
#### VAE Feature Extraction
```bash
python vae.py --mode classification
```
#### ML & CNN Experiments (Scenarios 1–3)
```bash
python task2.py --scenario <number>

```
#### N-gram Detection (Scenario 1)
```bash
python task2d.py
python task2e.py
```

#### Plotting Results
```bash
python plot.py
```
### Task 3 – Prediction Error Analysis
```bash
python task3_venn.py
```
### Task3 - LIME AND SHAP analysis
```bash
python task3b_lime.py --scenario <number>
python task3c_shap.py --scenario <number>
```
### Explainers Used

| Model | Explainer | Notes |
|-------|-----------|-------|
| RandomForest | TreeExplainer | Fast, exact |
| SVM | KernelExplainer | Uses decision_function wrapper |
| kNN | KernelExplainer | 500 samples for accuracy |
| OCSVM | KernelExplainer | Anomaly score → probability |
| LOF | Not supported | Skip (use LIME instead) |
| CNN | DeepExplainer | Aggregates across timesteps |


## Data Requirements

Both scripts use:
- **Latent features**: `vae_features/task1_dense_relu_ld8_{mode}_M20.npy`
- **Trained models**: `saved_models/Scenario{N}/{Model}_Fold{F}.joblib` or `.h5`
- **Labels**: Loaded from `../synthetic_train and synthetic_test/` 

## Note

All models being used need to be present in coresponding directory in order to run experiments. make sure to run each task consectively to have all the model available for use

## Conclusion

This practical sheet demonstrates that synthetic data generated using a GAN can effectively replace real training data for multiple anomaly detection methods. The experiments show comparable performance across scenarios, while the error analysis reveals that different classifiers capture different aspects of anomalous behavior. The combination of synthetic data generation, deep feature extraction, and detailed error analysis provides a robust framework for evaluating ICS anomaly detection systems.
