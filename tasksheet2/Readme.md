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
install scikit library

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

