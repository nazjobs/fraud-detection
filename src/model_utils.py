import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates a trained model and prints key metrics.
    Returns a dictionary of metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    )

    print(f"--- {model_name} Evaluation ---")
    print(classification_report(y_test, y_pred))

    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)

    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR:  {auc_pr:.4f}")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    return {"Model": model_name, "AUC-ROC": auc_roc, "AUC-PR": auc_pr}


def run_cross_validation(model, X, y, k=5):
    """
    Performs Stratified K-Fold CV and returns the mean AUC-PR.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    # Note: 'average_precision' is the scoring key for AUC-PR
    scores = cross_val_score(model, X, y, cv=skf, scoring="average_precision", n_jobs=1)

    print(
        f"Cross-Validation AUC-PR (k={k}): {scores.mean():.4f} +/- {scores.std():.4f}"
    )
    return scores.mean(), scores.std()
