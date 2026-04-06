"""
Train a learned stopping classifier from collected trajectories.

Loads trajectories.json, trains logistic regression + gradient boosted tree,
reports metrics, and saves the best model to experiments/models/stopping_classifier.pkl.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# ── Project root on sys.path ─────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Constants ────────────────────────────────────────────────────────────────
SEED = 42
DATA_DIR = Path(__file__).resolve().parent / "data"
MODELS_DIR = Path(__file__).resolve().parent / "models"
TRAJECTORIES_FILE = DATA_DIR / "trajectories.json"
MODEL_FILE = MODELS_DIR / "stopping_classifier.pkl"

FEATURE_NAMES = [
    "n_workspace_items",
    "max_relevance",
    "mean_relevance",
    "min_relevance",
    "n_unique_sources",
    "relevance_diversity",
    "step_number",
    "new_items_added",
    "max_relevance_improvement",
]


def load_trajectories(path: Path = TRAJECTORIES_FILE) -> list[dict]:
    print(f"Loading trajectories from {path} ...")
    with open(path, "r", encoding="utf-8") as fh:
        trajectories = json.load(fh)
    print(f"Loaded {len(trajectories)} trajectories.")
    return trajectories


def build_feature_matrix(trajectories: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X and label vector y from trajectories.

    Each row corresponds to one (question, step) pair.
    Label = is_optimal_stop.
    """
    X_rows = []
    y_rows = []

    for traj in trajectories:
        for step_rec in traj["steps"]:
            features = step_rec["features"]
            labels = step_rec["labels"]

            row = [features.get(fname, 0.0) for fname in FEATURE_NAMES]
            X_rows.append(row)
            y_rows.append(int(labels.get("is_optimal_stop", 0)))

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    return X, y


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """Train logistic regression and GBT classifiers."""

    # ── Logistic Regression ──────────────────────────────────────────────────
    print("\nTraining Logistic Regression ...")
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            random_state=SEED,
            max_iter=500,
            class_weight="balanced",  # handle class imbalance
            C=1.0,
        )),
    ])
    lr_pipeline.fit(X_train, y_train)
    print("  Done.")

    # ── Gradient Boosted Tree ────────────────────────────────────────────────
    print("\nTraining Gradient Boosted Tree ...")
    gbt_clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=SEED,
    )
    gbt_clf.fit(X_train, y_train)
    print("  Done.")

    return {
        "logistic_regression": lr_pipeline,
        "gradient_boosted_tree": gbt_clf,
    }


def evaluate_models(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate all models on the test split and report metrics."""
    results = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        results[model_name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        print(f"\n=== {model_name} ===")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Confusion matrix:")
        print(f"    {cm}")
        print(f"\n  Classification report:")
        print(classification_report(y_test, y_pred, target_names=["continue", "stop"]))

    return results


def print_feature_importances(models: dict) -> None:
    """Print feature importances for both models."""

    print("\n=== Feature Importances ===\n")

    # Logistic Regression coefficients
    lr = models["logistic_regression"]
    lr_clf = lr.named_steps["clf"]
    coefs = lr_clf.coef_[0]

    print("Logistic Regression coefficients (scaled):")
    lr_importances = sorted(
        zip(FEATURE_NAMES, coefs), key=lambda x: abs(x[1]), reverse=True
    )
    for fname, coef in lr_importances:
        print(f"  {fname:<30}: {coef:+.4f}")

    # GBT feature importances
    gbt = models["gradient_boosted_tree"]
    gbt_importances = sorted(
        zip(FEATURE_NAMES, gbt.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nGradient Boosted Tree feature importances:")
    for fname, imp in gbt_importances:
        print(f"  {fname:<30}: {imp:.4f}")


def select_best_model(models: dict, eval_results: dict) -> tuple[str, object]:
    """Select the model with the highest F1 on the test set."""
    best_name = max(eval_results.keys(), key=lambda k: eval_results[k]["f1"])
    best_model = models[best_name]
    print(f"\nBest model: {best_name} (F1={eval_results[best_name]['f1']:.4f})")
    return best_name, best_model


def find_optimal_threshold(model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Find probability threshold that maximises F1 on test set.
    Used to tune the binary prediction boundary.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    best_thresh = 0.5
    best_f1 = 0.0

    for thresh in np.arange(0.1, 0.95, 0.05):
        y_pred_thresh = (y_prob >= thresh).astype(int)
        f = f1_score(y_test, y_pred_thresh, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = float(thresh)

    print(f"Optimal threshold: {best_thresh:.2f} (F1={best_f1:.4f})")
    return best_thresh


def save_model(
    model,
    model_name: str,
    threshold: float,
    feature_importances: list,
    eval_metrics: dict,
    path: Path = MODEL_FILE,
) -> None:
    """Save model + metadata to pickle."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "model_name": model_name,
        "feature_names": FEATURE_NAMES,
        "threshold": threshold,
        "feature_importances": feature_importances,
        "eval_metrics": {
            k: v for k, v in eval_metrics.items()
            if not isinstance(v, np.ndarray)
        },
        "seed": SEED,
    }
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"\nModel saved to: {path}")


def train_stopping_model() -> dict:
    """Full training pipeline: load -> build features -> train -> evaluate -> save."""
    np.random.seed(SEED)

    # Load trajectories
    if not TRAJECTORIES_FILE.exists():
        raise FileNotFoundError(
            f"Trajectories not found at {TRAJECTORIES_FILE}. "
            "Run collect_trajectories.py first."
        )
    trajectories = load_trajectories()

    # Build feature matrix
    print("\nBuilding feature matrix ...")
    X, y = build_feature_matrix(trajectories)
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Positive examples (is_optimal_stop=1): {y.sum()} ({100*y.mean():.1f}%)")
    print(f"  Negative examples (is_optimal_stop=0): {(1-y).sum()} ({100*(1-y.mean()):.1f}%)")

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    eval_results = evaluate_models(models, X_test, y_test)

    # Feature importances
    print_feature_importances(models)

    # Select best model
    best_name, best_model = select_best_model(models, eval_results)

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(best_model, X_test, y_test)

    # Feature importances for saving
    if best_name == "logistic_regression":
        lr_clf = best_model.named_steps["clf"]
        importances = list(zip(FEATURE_NAMES, lr_clf.coef_[0].tolist()))
    else:
        importances = list(zip(FEATURE_NAMES, best_model.feature_importances_.tolist()))

    # Save best model
    save_model(
        model=best_model,
        model_name=best_name,
        threshold=optimal_threshold,
        feature_importances=importances,
        eval_metrics={
            "accuracy": eval_results[best_name]["accuracy"],
            "precision": eval_results[best_name]["precision"],
            "recall": eval_results[best_name]["recall"],
            "f1": eval_results[best_name]["f1"],
        },
    )

    print("\n=== Summary ===")
    print(f"Best model: {best_name}")
    print(f"Test accuracy: {eval_results[best_name]['accuracy']:.4f}")
    print(f"Test F1: {eval_results[best_name]['f1']:.4f}")
    print(f"Optimal threshold: {optimal_threshold:.2f}")
    print(f"Feature importances:")
    for fname, imp in sorted(importances, key=lambda x: abs(x[1]), reverse=True):
        if best_name == "logistic_regression":
            print(f"  {fname:<30}: {imp:+.4f}")
        else:
            print(f"  {fname:<30}: {imp:.4f}")

    return {
        "best_model_name": best_name,
        "eval_results": eval_results,
        "optimal_threshold": optimal_threshold,
    }


if __name__ == "__main__":
    train_stopping_model()
