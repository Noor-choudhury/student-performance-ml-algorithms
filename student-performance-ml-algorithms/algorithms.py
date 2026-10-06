"""
============================================================
MACHINE LEARNING ALGORITHM ASSIGNMENT
Dataset: UCI Student Performance (student-mat.csv)
File: algorithms.py
============================================================
This script automatically executes 19 machine learning algorithms across:
- Clustering (K-Means, Modified K-Means, Hierarchical, Fuzzy C-Means, DBSCAN, HDBSCAN)
- Semi-Supervised Learning (Self-Training)
- Ensemble Learning (Random Forest Classification/Regression, XGBoost, AdaBoost, CatBoost)
- Neural Networks (MLP, RNN, GRNN)
- Sequence & Map Models (SOM, HMM)
- Supervised Classification (SVM)
- Language Modeling Application (LLM Demonstration)
============================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

# Ignore warnings for clean terminal presentation
warnings.filterwarnings("ignore")

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, adjusted_rand_score
)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Self-Training import with fallback
try:
    from sklearn.semi_supervised import SelfTrainingClassifier
except ImportError:
    SelfTrainingClassifier = None

# Advanced ML Libraries
import xgboost as xgb
import catboost as cb

# Deep Learning (PyTorch)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# MiniSom
try:
    from minisom import MiniSom
except ImportError:
    MiniSom = None

# HMM
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None

# Fuzzy C-Means
try:
    import skfuzzy as fuzz
except ImportError:
    fuzz = None

# HDBSCAN
try:
    import hdbscan
except ImportError:
    hdbscan = None

# Transformers (LLM)
try:
    from transformers import pipeline
except ImportError:
    pipeline = None


# ============================================================
# GLOBAL DATA & PREPROCESSING
# ============================================================

def load_and_preprocess_data(filepath="student-mat.csv"):
    """
    Verifies dataset existence, loads student-mat.csv, sets up target features,
    and returns encoded/scaled numerical matrices for downstream algorithms.
    """
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} was not found.")
        print(f"Please place {filepath} in the same folder as algorithms.py.")
        sys.exit(1)

    df = pd.read_csv(filepath, sep=";")
    print("============================================================")
    print("MACHINE LEARNING ALGORITHM ASSIGNMENT")
    print("============================================================")
    print("Dataset loaded successfully.")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Pass rate (G3 >= 10): {(df['G3'] >= 10).mean() * 100:.2f}%")
    print("============================================================\n")

    # Classification Target Definition:
    # Academic Scheme: G3 < 10 -> Fail (0), G3 >= 10 -> Pass (1)
    # We exclude G1, G2, and G3 from feature sets to prevent data leakage.
    df['grade_category'] = (df['G3'] >= 10).astype(int)

    feature_cols = [c for c in df.columns if c not in ['G1', 'G2', 'G3', 'grade_category']]
    
    categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Compatible with both newer and older scikit-learn versions.
    try:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    except TypeError:
        encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', encoder, categorical_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(df[feature_cols])
    y_class = df['grade_category'].values
    y_reg = df['G3'].values

    # Scaled numerical features only
    scaler_num = StandardScaler()
    X_num_scaled = scaler_num.fit_transform(df[numerical_cols])

    return {
        'df': df,
        'X': X_processed,
        'X_num': X_num_scaled,
        'y_class': y_class,
        'y_reg': y_reg,
        'feature_cols': feature_cols
    }


# ============================================================
# HELPER REPORTING FUNCTIONS
# ============================================================

def print_header(algo_name):
    print("=" * 60)
    print(f"Algorithm: {algo_name}")
    print("Dataset: Student Performance")
    print("=" * 60)

def print_class_eval(y_true, y_pred, result_msg):
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    
    print("\nResult:")
    print(result_msg)
    print("\nEvaluation:")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Precision: {prec:.2f}%")
    print(f"Recall: {rec:.2f}%")
    print(f"F1-Score: {f1:.2f}%")
    print("-" * 60 + "\n")
    return f"Accuracy: {acc:.2f}%"

def print_reg_eval(y_true, y_pred, result_msg):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print("\nResult:")
    print(result_msg)
    print("\nEvaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("-" * 60 + "\n")
    return f"RMSE: {rmse:.4f}, R²: {r2:.4f}"

def print_cluster_eval(X, labels, ground_truth, result_msg):
    """
    Robust clustering evaluation.

    DBSCAN/HDBSCAN may legitimately produce fewer than two non-noise
    clusters. In that case silhouette is reported as N/A rather than
    using -1.0000 as if it were a real score.
    """
    labels = np.asarray(labels)
    non_noise = labels != -1
    unique_clusters = np.unique(labels[non_noise])
    n_clusters = len(unique_clusters)
    n_noise = int(np.sum(~non_noise))

    sil_text = "N/A"
    if n_clusters >= 2 and np.sum(non_noise) > n_clusters:
        try:
            sil = silhouette_score(X[non_noise], labels[non_noise])
            sil_text = f"{sil:.4f}"
        except Exception:
            sil_text = "N/A"

    ari = adjusted_rand_score(ground_truth, labels)

    print("\nResult:")
    print(result_msg)
    print(f"Clusters found: {n_clusters}")
    if n_noise > 0:
        print(f"Noise/unassigned samples: {n_noise}")
    print("\nEvaluation:")
    print(f"Silhouette Score: {sil_text}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print("-" * 60 + "\n")

    return f"Silhouette: {sil_text}, ARI: {ari:.4f}"


# ============================================================
# ALGORITHM IMPLEMENTATIONS
# ============================================================

def run_kmeans(data):
    print_header("K-Means")
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(data['X'])
    msg = "Students were grouped into 3 clusters based on academic & socio-demographic features."
    return print_cluster_eval(data['X'], labels, data['y_class'], msg)


def run_modified_kmeans(data):
    """
    MODIFIED K-MEANS IMPLEMENTATION:
    - Custom Centroid Initialization: Uses percentile distribution along feature density instead of random k-means++.
    - Outlier-Aware Distance Update: Re-assigns noisy points exceeding 2 standard deviations in distance.
    """
    print_header("Modified K-Means")
    X = data['X']
    k = 3
    
    mags = np.linalg.norm(X, axis=1)
    sorted_idx = np.argsort(mags)
    init_indices = [sorted_idx[int(i * len(sorted_idx) / k)] for i in range(k)]
    centroids = X[init_indices].copy()

    for _ in range(20):
        dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(dists, axis=1)
        
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                c_mean = cluster_points.mean(axis=0)
                pt_dists = np.linalg.norm(cluster_points - c_mean, axis=1)
                valid = pt_dists < (np.mean(pt_dists) + 2 * np.std(pt_dists))
                new_centroids[i] = cluster_points[valid].mean(axis=0) if np.sum(valid) > 0 else c_mean
            else:
                new_centroids[i] = centroids[i]
        centroids = new_centroids

    msg = "Executed Modified K-Means with custom percentile initialization and outlier-pruned updates."
    return print_cluster_eval(X, labels, data['y_class'], msg)


def run_hierarchical(data):
    print_header("Hierarchical Clustering")
    agg = AgglomerativeClustering(n_clusters=3)
    labels = agg.fit_predict(data['X'])
    msg = "Agglomerative hierarchical clustering completed with Ward linkage."
    return print_cluster_eval(data['X'], labels, data['y_class'], msg)


def run_fuzzy_cmeans(data):
    print_header("Fuzzy C-Means")
    if fuzz is None:
        raise ImportError("skfuzzy is not installed.")
    
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data['X'].T, c=3, m=2.0, error=0.005, maxiter=1000, seed=42
    )
    labels = np.argmax(u, axis=0)
    msg = f"Soft clustering completed. Final Partition Coefficient (FPC): {fpc:.4f}"
    return print_cluster_eval(data['X'], labels, data['y_class'], msg)


def run_dbscan(data):
    print_header("DBSCAN")

    X = data['X']
    best = None

    # Try several reasonable eps values because one fixed value can mark
    # nearly every student as noise in a high-dimensional encoded dataset.
    for eps in [0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.0, 2.3, 2.6, 3.0]:
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(X)

        mask = labels != -1
        n_clusters = len(np.unique(labels[mask]))
        n_valid = int(np.sum(mask))

        if n_clusters >= 2 and n_valid > n_clusters:
            try:
                score = silhouette_score(X[mask], labels[mask])
                if best is None or score > best[0]:
                    best = (score, eps, labels)
            except Exception:
                pass

    if best is None:
        # Fall back to a deterministic DBSCAN result and explain why
        # silhouette is not applicable.
        db = DBSCAN(eps=2.0, min_samples=5)
        labels = db.fit_predict(X)
        eps_used = 2.0
        msg = (
            f"DBSCAN completed with eps={eps_used:.1f}. "
            "The data did not produce at least two non-noise clusters, "
            "so silhouette score is not applicable."
        )
    else:
        score, eps_used, labels = best
        n_noise = int(np.sum(labels == -1))
        msg = (
            f"DBSCAN automatically selected eps={eps_used:.1f} "
            f"using the best valid silhouette score. "
            f"Noise points: {n_noise}."
        )

    return print_cluster_eval(X, labels, data['y_class'], msg)


def run_hdbscan(data):
    print_header("HDBSCAN")
    if hdbscan is None:
        raise ImportError("hdbscan is not installed.")

    X = data['X']
    best = None

    # Try several minimum cluster sizes and select a valid clustering
    # with the strongest silhouette score.
    for min_size in [5, 8, 10, 12, 15, 20, 25, 30]:
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_size,
                min_samples=max(3, min_size // 2),
                cluster_selection_method="eom"
            )
            labels = clusterer.fit_predict(X)

            mask = labels != -1
            n_clusters = len(np.unique(labels[mask]))
            n_valid = int(np.sum(mask))

            if n_clusters >= 2 and n_valid > n_clusters:
                score = silhouette_score(X[mask], labels[mask])
                if best is None or score > best[0]:
                    best = (score, min_size, labels)
        except Exception:
            continue

    if best is None:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=10,
            min_samples=5,
            cluster_selection_method="eom"
        )
        labels = clusterer.fit_predict(X)
        msg = (
            "HDBSCAN completed, but fewer than two valid non-noise clusters "
            "were found. Silhouette score is therefore not applicable."
        )
    else:
        score, min_size, labels = best
        n_noise = int(np.sum(labels == -1))
        msg = (
            f"HDBSCAN automatically selected min_cluster_size={min_size}. "
            f"Noise points: {n_noise}."
        )

    return print_cluster_eval(X, labels, data['y_class'], msg)


def run_self_training(data):
    """
    Genuine semi-supervised self-training.

    A portion of training labels is hidden. A classifier is trained on
    the remaining labeled samples, high-confidence predictions are added
    as pseudo-labels, and the classifier is retrained iteratively.
    This implementation is deliberately independent of sklearn's
    SelfTrainingClassifier API so it works across sklearn versions.
    """
    print_header("Self-Training")

    X_train, X_test, y_train, y_test = train_test_split(
        data['X'],
        data['y_class'],
        test_size=0.30,
        random_state=42,
        stratify=data['y_class']
    )

    rng = np.random.RandomState(42)

    # Keep 40% labeled and hide 60% of the training labels.
    labeled_mask = rng.rand(len(y_train)) < 0.40

    # Guarantee at least one labeled sample from every class.
    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]
        if not np.any(labeled_mask[cls_idx]):
            labeled_mask[rng.choice(cls_idx)] = True

    y_masked = np.full_like(y_train, -1)
    y_masked[labeled_mask] = y_train[labeled_mask]

    X_current = X_train[labeled_mask].copy()
    y_current = y_train[labeled_mask].copy()
    unlabeled_idx = np.where(~labeled_mask)[0].tolist()

    initial_labeled = len(y_current)
    total_pseudo = 0

    for iteration in range(10):
        if not unlabeled_idx:
            break

        clf = RandomForestClassifier(
            n_estimators=150,
            random_state=42 + iteration,
            class_weight="balanced"
        )
        clf.fit(X_current, y_current)

        X_unlabeled = X_train[unlabeled_idx]
        probabilities = clf.predict_proba(X_unlabeled)
        confidence = np.max(probabilities, axis=1)
        pseudo_labels = clf.classes_[np.argmax(probabilities, axis=1)]

        # Gradually relax the confidence threshold if necessary.
        threshold = max(0.70, 0.90 - iteration * 0.02)
        selected = confidence >= threshold

        if not np.any(selected):
            # If no sample is sufficiently confident, take the strongest
            # prediction only when confidence is reasonably reliable.
            best_pos = int(np.argmax(confidence))
            if confidence[best_pos] >= 0.65:
                selected = np.zeros(len(confidence), dtype=bool)
                selected[best_pos] = True
            else:
                break

        selected_positions = np.where(selected)[0]
        selected_indices = [unlabeled_idx[i] for i in selected_positions]

        X_current = np.vstack([X_current, X_train[selected_indices]])
        y_current = np.concatenate([y_current, pseudo_labels[selected_positions]])

        total_pseudo += len(selected_indices)
        unlabeled_idx = [
            idx for i, idx in enumerate(unlabeled_idx)
            if not selected[i]
        ]

    final_clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    final_clf.fit(X_current, y_current)
    y_pred = final_clf.predict(X_test)

    msg = (
        f"Initial labeled samples: {initial_labeled} | "
        f"Pseudo-labeled samples added: {total_pseudo} | "
        f"Final training samples: {len(y_current)}"
    )
    return print_class_eval(y_test, y_pred, msg)


def run_rfc(data):
    print_header("Random Forest Classification")
    X_train, X_test, y_train, y_test = train_test_split(
        data['X'], data['y_class'], test_size=0.30,
        random_state=42, stratify=data['y_class']
    )
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    msg = "Student pass/fail grade category prediction completed."
    return print_class_eval(y_test, y_pred, msg)


def run_rfr(data):
    print_header("Random Forest Regression")
    X_train, X_test, y_train, y_test = train_test_split(
        data['X'], data['y_reg'], test_size=0.3, random_state=42
    )
    rfr = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    )
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    msg = "G3 numerical final grade prediction completed."
    return print_reg_eval(y_test, y_pred, msg)


def run_xgboost(data):
    print_header("XGBoost")
    X_train, X_test, y_train, y_test = train_test_split(
        data['X'], data['y_class'], test_size=0.30,
        random_state=42, stratify=data['y_class']
    )
    xgb_cls = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_cls.fit(X_train, y_train)
    y_pred = xgb_cls.predict(X_test)
    msg = "Gradient boosting classification completed for student performance target."
    return print_class_eval(y_test, y_pred, msg)


def run_adaboost(data):
    print_header("AdaBoost")
    X_train, X_test, y_train, y_test = train_test_split(
        data['X'], data['y_class'], test_size=0.30,
        random_state=42, stratify=data['y_class']
    )
    ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    msg = "Adaptive Boosting classification completed."
    return print_class_eval(y_test, y_pred, msg)


def run_catboost(data):
    print_header("CatBoost")
    X_train, X_test, y_train, y_test = train_test_split(
        data['X'], data['y_class'], test_size=0.30,
        random_state=42, stratify=data['y_class']
    )
    cb_cls = cb.CatBoostClassifier(verbose=0, random_state=42)
    cb_cls.fit(X_train, y_train)
    y_pred = cb_cls.predict(X_test)
    msg = "CatBoost classification completed."
    return print_class_eval(y_test, y_pred, msg)


def run_mlp(data):
    print_header("Multilayer Perceptron (MLP)")
    X_train, X_test, y_train, y_test = train_test_split(
        data['X'], data['y_class'], test_size=0.30,
        random_state=42, stratify=data['y_class']
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0005,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    msg = f"MLP convergence status: Reached max_iter or converged cleanly."
    return print_class_eval(y_test, y_pred, msg)


# Custom PyTorch RNN Module
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

def run_rnn(data):
    """
    RECURRENT NEURAL NETWORK IMPLEMENTATION:
    Feature Transformation: Tabular student feature vector of size D is restructured
    into a sequence of 4 temporal feature chunks (Seq_Len=4, Feat_Dim=D//4).
    """
    print_header("Recurrent Neural Network (RNN)")
    X = data['X']
    y = data['y_class']
    
    n_samples, n_features = X.shape
    seq_len = 4
    padded_dim = int(np.ceil(n_features / seq_len) * seq_len)
    
    if padded_dim > n_features:
        pad = np.zeros((n_samples, padded_dim - n_features))
        X_padded = np.hstack([X, pad])
    else:
        X_padded = X

    feat_dim = padded_dim // seq_len
    X_seq = X_padded.reshape(n_samples, seq_len, feat_dim)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y, test_size=0.3, random_state=42
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    model = SimpleRNN(input_dim=feat_dim, hidden_dim=16, output_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(20):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t)
        y_pred = torch.argmax(preds, dim=1).numpy()

    msg = f"Tabular data reshaped into sequential input (Seq Length: {seq_len}, Feature Dim: {feat_dim})."
    return print_class_eval(y_test, y_pred, msg)


def run_som(data):
    print_header("Self-Organizing Map (SOM)")
    if MiniSom is None:
        raise ImportError("minisom is not installed.")

    X = data['X_num']
    som = MiniSom(x=5, y=5, input_len=X.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
    som.train_random(X, num_iteration=500)

    q_error = som.quantization_error(X)
    winner_coordinates = [som.winner(x) for x in X]
    labels = [x * 5 + y for x, y in winner_coordinates]

    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else -1.0
    print("\nResult:")
    print("Self-Organizing Map 5x5 grid training completed.")
    print("\nEvaluation:")
    print(f"Quantization Error: {q_error:.4f}")
    print(f"Silhouette Score: {sil:.4f}")
    print("-" * 60 + "\n")
    return f"Quantization Error: {q_error:.4f}"


def run_hmm(data):
    """
    Gaussian HMM demonstration.

    Student records are treated as a sequence of observations. Since HMM
    state IDs are arbitrary, each learned state is mapped to its majority
    grade category before calculating accuracy.
    """
    print_header("Hidden Markov Model (HMM)")
    X = data['X_num']
    y = data['y_class']

    if GaussianHMM is None:
        raise ImportError("hmmlearn is not installed.")

    # Use a small number of hidden states corresponding to performance
    # regimes. The model remains unsupervised.
    hmm_model = GaussianHMM(
        n_components=2,
        covariance_type="diag",
        n_iter=200,
        random_state=42
    )
    hmm_model.fit(X)
    hidden_states = hmm_model.predict(X)

    # Map hidden state IDs to the majority observed class.
    state_to_class = {}
    for state in np.unique(hidden_states):
        mask = hidden_states == state
        if np.any(mask):
            values, counts = np.unique(y[mask], return_counts=True)
            state_to_class[state] = int(values[np.argmax(counts)])

    mapped_predictions = np.array(
        [state_to_class.get(state, 0) for state in hidden_states]
    )

    acc = accuracy_score(y, mapped_predictions) * 100
    prec = precision_score(y, mapped_predictions, average='weighted', zero_division=0) * 100
    rec = recall_score(y, mapped_predictions, average='weighted', zero_division=0) * 100
    f1 = f1_score(y, mapped_predictions, average='weighted', zero_division=0) * 100

    print("\nResult:")
    print("Gaussian HMM discovered hidden student performance states.")
    print("\nEvaluation:")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Precision: {prec:.2f}%")
    print(f"Recall: {rec:.2f}%")
    print(f"F1-Score: {f1:.2f}%")
    print("-" * 60 + "\n")

    return f"Accuracy: {acc:.2f}%"


def run_svm(data):
    print_header("Support Vector Machine (SVM)")
    X_train, X_test, y_train, y_test = train_test_split(
        data['X'], data['y_class'], test_size=0.30,
        random_state=42, stratify=data['y_class']
    )
    svm = SVC(kernel='rbf', C=2.0, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    msg = "Support Vector Classification with RBF kernel completed."
    return print_class_eval(y_test, y_pred, msg)


def run_llm(data):
    """
    LARGE LANGUAGE MODEL (LLM) APPLICATION / DEMONSTRATION:
    Contextual Text Summarization: Tabular dataset metrics are converted into structured text prompts
    and processed by a lightweight pre-trained language model (DistilGPT2 / Pipeline) or local synthesis.
    """
    print_header("Large Language Model (LLM)")
    
    df = data['df']
    avg_age = df['age'].mean()
    pass_rate = (df['G3'] >= 10).mean() * 100
    avg_absences = df['absences'].mean()

    prompt_context = (
        f"Academic Cohort Summary: The dataset contains {len(df)} students with an average age of {avg_age:.1f}. "
        f"The overall passing rate is {pass_rate:.1f}%, and average absence rate is {avg_absences:.1f} days."
    )

    result_summary = ""
    llm_status = "Fallback summary"

    try:
        if pipeline is not None:
            generator = pipeline(
                'text-generation',
                model='distilgpt2',
                max_new_tokens=40
            )
            llm_out = generator(
                prompt_context,
                num_return_sequences=1,
                do_sample=False
            )[0]['generated_text']
            result_summary = f"LLM Generated Academic Insight:\n\"{llm_out}\""
            llm_status = "Pretrained LLM summary generated"
        else:
            result_summary = (
                f"Generated Academic Summary Context:\n{prompt_context}\n"
                "Transformers is not installed, so the program used a "
                "deterministic local summary instead."
            )
    except Exception as e:
        result_summary = (
            f"Local academic summary:\n{prompt_context}\n"
            f"Pretrained LLM unavailable: {str(e)[:120]}"
        )


    print("\nResult:")
    print("Generated academic performance summary using LLM application context.")
    print("\nOutput Text / Insight:")
    print(result_summary)
    print("-" * 60 + "\n")
    return llm_status


def run_grnn(data):
    """
    Generalized Regression Neural Network (GRNN).

    The GRNN is implemented directly using Gaussian kernel regression.
    A validation set is used to select a sensible smoothing parameter
    instead of fixing sigma=1.0 blindly.
    """
    print_header("Generalized Regression Neural Network (GRNN)")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        data['X'],
        data['y_reg'],
        test_size=0.30,
        random_state=42
    )

    # Split the training data into train/validation for sigma selection.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=42
    )

    def grnn_predict(X_query, X_ref, y_ref, sigma):
        distances = np.linalg.norm(
            X_query[:, np.newaxis, :] - X_ref[np.newaxis, :, :],
            axis=2
        )
        weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))
        denominator = np.sum(weights, axis=1)
        denominator = np.maximum(denominator, 1e-12)
        return (weights @ y_ref) / denominator

    # The encoded feature space is relatively high-dimensional, so test
    # multiple bandwidths and select the one with the lowest validation RMSE.
    sigma_candidates = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
    best_sigma = sigma_candidates[0]
    best_rmse = float("inf")

    for sigma in sigma_candidates:
        val_pred = grnn_predict(X_val, X_train, y_train, sigma)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_sigma = sigma

    # Refit conceptually by using all available training observations.
    y_pred = grnn_predict(
        X_test, X_train_full, y_train_full, best_sigma
    )

    msg = (
        f"Gaussian kernel regression completed. "
        f"Selected smoothing parameter sigma = {best_sigma:.2f} "
        f"(validation RMSE = {best_rmse:.4f})."
    )
    return print_reg_eval(y_test, y_pred, msg)


# ============================================================
# MAIN EXECUTION ROUTINE
# ============================================================

def main():
    data = load_and_preprocess_data("student-mat.csv")

    algorithms = [
        ("K-Means", "Clustering", run_kmeans),
        ("Modified K-Means", "Clustering", run_modified_kmeans),
        ("Hierarchical", "Clustering", run_hierarchical),
        ("Fuzzy C-Means", "Clustering", run_fuzzy_cmeans),
        ("DBSCAN", "Clustering", run_dbscan),
        ("HDBSCAN", "Clustering", run_hdbscan),
        ("Self-Training", "Classification", run_self_training),
        ("Random Forest Classifier", "Classification", run_rfc),
        ("Random Forest Regression", "Regression", run_rfr),
        ("XGBoost", "Classification", run_xgboost),
        ("AdaBoost", "Classification", run_adaboost),
        ("CatBoost", "Classification", run_catboost),
        ("MLP", "Classification", run_mlp),
        ("RNN", "Classification", run_rnn),
        ("SOM", "Clustering", run_som),
        ("HMM", "Sequence Model", run_hmm),
        ("SVM", "Classification", run_svm),
        ("LLM", "Language Task", run_llm),
        ("GRNN", "Regression", run_grnn)
    ]

    summary_records = []
    successful_count = 0
    failed_count = 0

    for name, task, func in algorithms:
        try:
            eval_metric = func(data)
            summary_records.append((name, task, eval_metric))
            successful_count += 1
        except Exception as e:
            print(f"Algorithm Failure [{name}]: {e}\n" + "-"*60)
            summary_records.append((name, task, f"FAILED: {str(e)[:55]}"))
            failed_count += 1

    # Print Final Summary Report Table
    print("\n" + "=" * 60)
    print("FINAL MACHINE LEARNING SUMMARY")
    print("=" * 60)
    print("Dataset: Student Performance")
    print("=" * 60)
    print(f"{'Algorithm':<28} {'Result':<18} {'Evaluation'}")
    print("-" * 68)

    for name, task, eval_metric in summary_records:
        print(f"{name:<28} {task:<18} {eval_metric}")

    print("-" * 68)
    print("============================================================")
    print("ASSIGNMENT COMPLETED")
    print("============================================================")
    print(f"Total Algorithms: {len(algorithms)}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")
    print("\nAll results were generated using:")
    print("Student Performance Dataset")
    print("============================================================\n")


if __name__ == "__main__":
    main()