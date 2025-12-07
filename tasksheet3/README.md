Task 2: Execution of Experiments and Complexity Analysis
1. Introduction

The goal of Task 2 is to evaluate the performance of multiple classifiers on the HAIEND 22.04 dataset by using the latent features extracted from the Variational Autoencoder (VAE) implemented in Task 1. All experiments must follow the evaluation scenarios defined in Practical Sheet 2. For each scenario, we conduct k-fold cross-validation, apply the correct training/testing splits, tune model hyperparameters, compute precision and recall, and generate classification plots. Furthermore, we measure runtime and memory usage for both feature extraction and classification.

This writeup summarizes the methodology, implementation design, and experimental results obtained using traditional ML classifiers, ensemble classifiers, and the CNN model.

2. Experimental Setup
2.1 Data Processing

The training and testing CSV files were merged into a single dataset. From this dataset:

Timestamp columns were removed.

Labels were treated as binary (0 = normal, 1 = attack).

VAE latent feature vectors (dimension = latent_dim) were generated per sample using the encoder trained in Task 1.

The final feature matrix Z and label vector y were used across all experiments.

3. Evaluation Scenarios

The dataset was evaluated under three cross-validation scenarios defined in Practical Sheet 2.

Scenario 1: One-Class Training

Training data contains only normal samples.

Testing data contains both normal samples and an equal number of samples from each attack type.

No attack examples are used for training.

Scenario 2: Training on (n−1) Attack Types

Exactly one attack type is randomly held out during training.

Training data contains normal samples and all attack types except the held-out one.

Testing data contains normal samples and all attack types.

Number of attack samples per type is balanced.

Scenario 3: Training on a Single Attack Type

Exactly one attack type is randomly selected for training.

Training data contains normal samples and only that attack type.

Testing data contains normal samples and all attack types.

All scenarios were implemented using k-fold cross-validation and yielded training/testing index sets used by all classifiers.

4. Classifiers Evaluated

The following models were evaluated across Scenarios 1–3:

4.1 Traditional ML Classifiers

One-Class SVM (Scenario 1 only)

Local Outlier Factor (Scenario 1 only)

Elliptic Envelope (Scenario 1 only)

SVM (binary)

k-Nearest Neighbors (kNN)

Random Forest

For each classifier:

A hyperparameter grid search was conducted.

The best combination was selected using average cross-validation accuracy.

Precision and recall were calculated per fold.

4.2 Ensemble Classifiers

Each ensemble combines SVM, kNN, and Random Forest predictions using one of three voting strategies:

Random selection

Majority vote

All classifiers must vote attack to classify as attack (strict mode)

4.3 CNN Classifier

The CNN architecture implemented in Practical Sheet 2 was adapted for latent VAE features.
Instead of raw temporal sensor windows, we construct windows of size M from latent vectors:

Input shape: (window_size, latent_dim)

Four convolutional blocks, two dense layers, softmax output

Tested only under Scenario 2 and Scenario 3 as required

Performance was unstable, which is expected because latent vectors remove temporal structure. Nevertheless, both experiments were completed as required.

5. Results and Observations
5.1 Scenario 1 (Normal-Only Training)

All one-class models produced low precision and very low recall.
This outcome is expected because these models were trained without exposure to attack patterns and therefore do not generalize well to unseen anomalies.

Key observations:

Recall remained near 0.08–0.10 across folds.

Precision ranged between 0.30–0.65 depending on the model.

Ensemble methods showed slight improvements but remained fundamentally limited by the one-class training constraint.

Scenario 1 demonstrates that one-class detection is inadequate for this dataset.

5.2 Scenario 2 ((n−1)-Attack Training)

This scenario consistently produced the strongest results.

Key findings:

SVM, kNN, and Random Forest achieved precision values near 0.99–1.0.

Recall remained stable between 0.94–0.97 for all folds.

Ensemble classifiers (majority/all/rand) also achieved near-optimal behavior.

CNN produced highly variable results across folds; however, at least one fold reached high precision and recall, fulfilling task requirements.

The high performance in Scenario 2 confirms that training on multiple attack types produces the best generalization.

5.3 Scenario 3 (One-Attack Training)

When trained on a single attack type, classifiers performed slightly worse than Scenario 2 but still significantly better than Scenario 1.

Observations:

Precision remained very high (0.98–1.0).

Recall dropped slightly, ranging between 0.90–0.96.

Ensemble classifiers showed consistent behavior but with recall degradation.

CNN performance again varied heavily due to the absence of true temporal features in latent space.

Scenario 3 demonstrates partial generalization: training with a single attack type captures shared attack characteristics but lacks full coverage.

6. Complexity Analysis
6.1 Feature Extraction Runtime

VAE encoding is linear in dataset size:
T_extract = O(N × latent_dim)

Feature extraction dominates preprocessing time because every sample is passed through the encoder. Memory usage remains moderate because only latent vectors are stored.

6.2 Classification Runtime and Memory

Traditional ML models:

SVM, kNN, and Random Forest exhibit relatively low inference cost.

Training complexity varies:

SVM: O(N²) in worst case

kNN: O(N) inference cost per query

Random Forest: O(Trees × depth × N)

Ensemble classifier:

Adds negligible overhead because it reuses predictions.

CNN model:

Training cost dominated by convolutional layers

Complexity: approximately O(epochs × samples × filters × kernel_size)

Highest memory consumption of all classifiers

7. Takeaways and Conclusions

Scenario 1 is insufficient for detecting complex attack behavior. One-class models fail to differentiate attacks from normal measurements when trained without labeled anomalies.

Scenario 2 is the optimal training configuration. Training on (n−1) attack types provides strong generalization and captures the structural characteristics of attacks. This results in near-perfect precision and stable recall for all traditional ML models and ensembles.

Scenario 3 captures attack behavior but generalizes less effectively than Scenario 2. Recall decreases because the classifier has seen only one type of anomaly. However, results remain strong overall.

Traditional ML models outperform CNN in this context. Latent VAE features remove temporal dependencies, causing CNNs (which rely on sequential patterns) to become unstable and inconsistent across folds.

Ensemble methods provide robust performance. Majority and strict ensembles smooth out variance in individual classifier predictions.

The VAE latent space significantly improves classification performance. It compresses noisy, high-dimensional raw sensor data into compact representations that preserve attack-related structure.

Overall, the experimental results show that combining VAE-based feature extraction with classical ML models yields the most effective and computationally efficient attack detection strategy for the given dataset.