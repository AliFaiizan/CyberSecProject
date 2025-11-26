# Task 1: Anomaly Detection for HAI-End 22.04 Dataset

## Overview
This task focuses on detecting anomalies (attacks) in the HAI-End 22.04 dataset using various machine learning models. The workflow includes data loading, preprocessing, scenario-based cross-validation ,implementing classical classifiers, prediction, and result export.

## Data Structure
- **Train files:** `../datasets/haiend-22.04/train*.csv`
- **Test files:** `../datasets/haiend-22.04/test*.csv`
- Hai 22.04 has most physical reading parsed. and uses label = 0/1 for indication of attack 

## Main Steps
1. **Data Loading & Cleaning**
   - All train and test files are loaded and concatenated and merged.
   - Final dataset contains a `label` column indicating attack or normal.

2. **Feature Preparation**
   - Features are all columns except `label` and `timestamp`.
   - Labels are extracted as a separate vector.

3. **Scenario-Based Cross-Validation**
   - **Scenario 1:** Train on normal data only, test on normal + all attacks.
   - **Scenario 2:** Train on normal + (n-1) attack types, test on normal fold + all attacks (one attack type held out).
   - **Scenario 3:** Train on normal + one attack type, test on normal fold + all attacks.
   - K-fold splitting is used for robust evaluation.

4. **Model Training & Prediction**
   - Supported models: OneClassSVM, EllipticEnvelope, LocalOutlierFactor, SVC (binary), kNN, RandomForest.
   - PCA is used for dimensionality reduction (optional, default 95% variance).
   - Models are trained and evaluated per fold.

5. **Result Export**
   - Predictions for each fold are exported to CSV files in the `exports/` directory.
   - Each output includes row indices, predicted labels, and attack/normal status.

## Usage
pip install -r requirements.txt

Run the main script with desired model and scenario:


```bash
python task1.py --model <model> --scenario <scenario>
```
- `<model>`: `ocsvm`, `lof`, `ee`, `knn`, `svm`, `rf`
- `<scenario>`: `1`, `2`, or `3` (2,3 are required for supervised models)

Example:
```bash
python task1.py --model ocsvm
python task1.py --model svm --scenario 2
```

## Output
- Results are saved in `exports/Scenario<scenario>/<Model>/Predictions_Fold<k>.csv`.
- Console output includes accuracy, number of detected attacks, and summary statistics per fold.

## Notes
- Ensure all data files are present in the `../datasets/haiend-22.04/` directory.
- One class models only implements scenario 1 while for binary you have to specify scenario type with --scenario command

Hyperparamerter search is commented out to save execution time , and only one set of params in provided. 
loading entire dataset which has around 1.3 million rows getting output from models can become slow even on server. 
and extremely slow on some models like svm

## File Structure
- `task1.py`: Main script for running experiments.
- `models.py`: Model training and prediction functions.
- `utils.py`: Data loading, cleaning, and utility functions.
- `exports.py`: Functions for exporting scenario folds.


# Task 2 & Task 3  
Deep Learning, Ensemble Classification, and Integrated Toolbox

---

## Overview

This repository contains the implementation of Task 2 (Deep Learning and Ensemble Classifier) and Task 3 (Toolbox Integration) from the ICS Security Study Project.  
The objective is to extend the anomaly detection workflow by introducing a convolutional neural network (CNN), an ensemble of traditional ML classifiers, and a unified command-line toolbox that integrates all tasks from Practical Sheet 1 and Sheet 2.

---

## Folder Structure

```
src/
│
├── task1.py                     # Traditional ML classifiers (Task 2 – Part 1)
├── task2.py                     # CNN deep learning classifier
├── task2_ensemble.py            # Ensemble classifier implementation
├── toolbox.py                   # Unified command-line toolbox (Task 3)
│
├── similarity_analysis/         # Spearman, PCA, t-SNE, histogram scripts
│   ├── spearman_distance.py
│   ├── histogram.py
│   ├── task2c.py
│   ├── task2d.py
│
├── statistical_analysis/        # Practical Sheet 1 scripts
```

---

## Task 2 — Deep Learning Classifier (task2.py)

### Description

The deep learning component is implemented as a six-block Convolutional Neural Network (CNN).  
The model performs binary classification on sliding windows of ICS sensor data.  
Each window contains M consecutive rows of parsed physical readings.

### Main Features

- Six-block network architecture:
  - Four convolutional blocks (Conv1D → BatchNorm → ReLU → Dropout).
  - Two fully connected blocks.
- Window generation using a sliding window approach.
- Normalization using StandardScaler applied only to training data.
- Balanced training using scikit-learn class weights.
- K-fold cross-validation based on the three evaluation scenarios defined in Task 1.
- One-hot encoded training labels and softmax output.
- Integrated `run_from_toolbox()` function for compatibility with the main toolbox.

### Output

All CNN predictions are saved in:

```
exports/ScenarioX/CNN/Predictions_FoldY.csv
```

Each file contains:

```
predicted_label, Attack
```

---

## Task 2 — Ensemble Classifier (task2_ensemble.py)

### Description

The ensemble classifier fuses the predictions of three traditional ML models from Task 1.  
The selected models depend on the scenario:

- Scenario 1: One-class classifiers  
  - OCSVM, LOF, Elliptic Envelope
- Scenario 2 and 3: Binary classifiers  
  - Random Forest, K-Nearest Neighbors, Support Vector Machine

### Supported Ensemble Methods

- random: Selects one of the three model predictions randomly for each sample.
- majority: Outputs 1 if at least two of the three models predict an attack.
- all: Outputs 1 only if all models predict an attack.

### Output

Results are stored under:

```
exports/ScenarioX/Ensemble/Ensemble_<method>_FoldY.csv
```

Each file contains:

```
predicted_, Attack
```

### Toolbox Integration

The script includes `run_from_toolbox()` to allow execution through the unified toolbox.

---

## Task 3 — Integrated Toolbox (toolbox.py)

### Description

The toolbox integrates all tasks from Sheet 1 and Sheet 2 into a single command-line tool.  
It supports dataset analysis, similarity computations, n-gram detection, ML models, DL models, and ensemble classification.

### Available Modes

| Mode        | Description                                                |
|-------------|------------------------------------------------------------|
| analyze     | Statistical dataset analysis (KS-tests, CCDFs)             |
| similarity  | Spearman, PCA, t-SNE, histogram                            |
| ngram       | N-gram anomaly detection                                   |
| ml          | Traditional ML classifiers (Task 2 – Part 1)               |
| dl          | CNN deep learning classifier (Task 2 – Part 2)             |
| ensemble    | Ensemble classifier (Task 2 – Part 3)                      |

### Running Examples

Traditional ML:

```
python toolbox.py ml -m rf -sc 3 -k 5
```

Deep learning:

```
python toolbox.py dl -sc 2 -M 50 -e 10 --stride 5
```

Ensemble classifier:

```
python toolbox.py ensemble -sc 2 -m majority -f 3
```

Similarity analysis:

```
python toolbox.py similarity --dataset hai-22.04
```

N-gram anomaly detection:

```
python toolbox.py ngram --dataset hai-21.03 --ngram-order 3
```

---

## Libraries Used

### Machine Learning and Deep Learning

- TensorFlow / Keras  
  - Sequential model  
  - Conv1D, BatchNormalization, Dropout  
  - Adam optimizer, EarlyStopping, Precision, Recall

- scikit-learn  
  - StandardScaler  
  - compute_class_weight  
  - Model implementations (in task1.py)

### Data Processing

- Pandas  
- NumPy

### System and Utilities

- argparse  
- os, sys  
- re  
- glob  

### Custom Modules

- load_and_clean_data() from utils.py  
- scenario_1_split / scenario_2_split / scenario_3_split from task1.py  

---

## Output Structure

All generated files are stored under:

```
exports/
│
├── Scenario1/
│   ├── OCSVM/
│   ├── LOF/
│   ├── EE/
│   ├── CNN/
│   └── Ensemble/
│
├── Scenario2/
│   └── ...
└── Scenario3/
    └── ...
```

Each classifier and fold produces a `Predictions_FoldX.csv` file.

---

## Summary

This project implements:

- A six-block CNN for anomaly detection (Task 2).
- Ensemble fusion of three ML models using Random, Majority, and All methods (Task 2).
- A unified command-line toolbox covering all tasks from Sheet 1 and Sheet 2 (Task 3).
- Support for all evaluation scenarios defined in the assignment.
- A modular structure that separates statistical analysis, ML models, DL models, ensemble methods, and toolbox logic.

This README provides an overview of the scripts, their purpose, how to execute them, and the structure of their outputs.
