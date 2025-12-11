# Task 3(c) & 3(d): Model Explainability

Generate local and global explanations for trained ML/CNN models using LIME and SHAP.

## Task 3(c): LIME Explanations

**LIME** (Local Interpretable Model-agnostic Explanations) provides instance-level explanations showing which features influenced specific predictions.

### Usage

```bash
# Scenario 1 (One-class models)
python task3_c.py --scenario 1

# Scenarios 2 & 3 (Binary classifiers)
python task3_c.py --scenario 2
python task3_c.py --scenario 3
```

### Models Explained

**Scenario 1**: OCSVM, LOF, Elliptic Envelope  
**Scenarios 2 & 3**: SVM, kNN, RandomForest, CNN

### Output

PNG files saved to `Task3_Results/Scenario{N}/LIME/Fold{F}/{Model}/`:
```
LIME_{ModelName}_sample{idx}.png
```
- 3 test samples per model
- 15 most important features shown
- Red bars: push toward attack class
- Green bars: push toward normal class

---

## Task 3(d): SHAP Explanations

**SHAP** (SHapley Additive exPlanations) provides global feature importance using game-theoretic approach.

### Usage

```bash
# Scenario 1
python task3_d.py --scenario 1

# Scenarios 2 & 3
python task3_d.py --scenario 2
python task3_d.py --scenario 3
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

### Output

Summary plots saved to `Task3_Results/Scenario{N}/SHAP/Fold{F}/{Model}/`:
```
SHAP_{ModelName}_summary.png
```
- Beeswarm plot showing all features
- Color: feature value (high/low)
- X-axis: SHAP value (impact on prediction)

### CNN Special Handling

CNN SHAP values are **aggregated across timesteps**:
- Raw shape: `(samples, 20 timesteps, 8 features)`
- Aggregated: `(samples, 8 features)` by summing absolute values
- Feature names: `z0-z7` (latent features)



## Data Requirements

Both scripts use:
- **Latent features**: `vae_features/task1_dense_relu_ld8_{mode}_M20.npy`
- **Trained models**: `saved_models/Scenario{N}/{Model}_Fold{F}.joblib` or `.h5`
- **Labels**: Loaded from `../datasets/hai-22.04/` and trimmed to match latent features


## Note

All models being used need to be present in coresponding directory in order to run experiments. make sure to run each task consectively to have all the model available for use
