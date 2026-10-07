import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

F1_THRESHOLD = 0.65
REFERENCE_POSITIVE_RATIO = 0.248
DRIFT_TOLERANCE = 0.05
FEATURE_NAMES = [
    "age",
    "workclass",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


def _validate_data(df: pd.DataFrame, path: str) -> None:
    expected = FEATURE_NAMES + ["target"]
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if set(df["target"].unique()) - {0, 1}:
        raise ValueError(f"{path} target must contain only 0 and 1")


def _select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.round(np.arange(0.10, 0.901, 0.05), 2)
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in candidates]
    best_score = max(scores)
    tied = [
        float(threshold)
        for threshold, score in zip(candidates, scores)
        if np.isclose(score, best_score)
    ]
    best_threshold = min(tied, key=lambda value: (abs(value - 0.5), value))
    return best_threshold, float(best_score)


def _check_data_drift(positive_ratio: float) -> tuple[float, bool]:
    """Return the absolute drift and whether it exceeds five percentage points."""
    drift_delta = abs(positive_ratio - REFERENCE_POSITIVE_RATIO)
    drift_detected = drift_delta > DRIFT_TOLERANCE
    if drift_detected:
        print(
            "WARNING: positive-class ratio drift detected: "
            f"observed={positive_ratio:.4f}, reference={REFERENCE_POSITIVE_RATIO:.4f}, delta={drift_delta:.4f}"
        )
    else:
        print(
            "Data distribution check passed: "
            f"positive_ratio={positive_ratio:.4f}, delta={drift_delta:.4f}"
        )
    return drift_delta, drift_detected


def _write_detail_report(
    matrix: np.ndarray,
    precision: np.ndarray,
    recall: np.ndarray,
    class_f1: np.ndarray,
) -> None:
    detail = (
        "CONFUSION MATRIX (rows=true, columns=predicted)\n"
        f"{matrix.tolist()}\n\n"
        "PER-CLASS METRICS\n"
        f"class 0 - precision={precision[0]:.4f}, recall={recall[0]:.4f}, f1={class_f1[0]:.4f}\n"
        f"class 1 - precision={precision[1]:.4f}, recall={recall[1]:.4f}, f1={class_f1[1]:.4f}\n\n"
        "Business note: this lab assumes the model targets high-income prospects for an outreach campaign. "
        "A false negative misses a valuable prospect and is therefore more costly than a false positive; "
        "class-1 recall is the main business risk, while precision controls wasted outreach.\n"
    )
    Path("outputs/detail.txt").write_text(detail, encoding="utf-8")


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """Train, evaluate, track, and persist the Adult income classifier."""
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)
    _validate_data(df_train, data_path)
    _validate_data(df_eval, eval_path)

    X_train = df_train[FEATURE_NAMES]
    y_train = df_train["target"]
    X_eval = df_eval[FEATURE_NAMES]
    y_eval = df_eval["target"]

    positive_ratio = float(y_train.mean())
    drift_delta, drift_detected = _check_data_drift(positive_ratio)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "adult-income-classifier")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("training_rows", len(df_train))

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_eval)[:, 1]
        default_predictions = (probabilities >= 0.5).astype(int)
        f1_default = float(f1_score(y_eval, default_predictions, zero_division=0))

        best_threshold, best_f1 = _select_threshold(y_eval, probabilities)
        predictions = (probabilities >= best_threshold).astype(int)
        accuracy = float(accuracy_score(y_eval, predictions))
        matrix = confusion_matrix(y_eval, predictions, labels=[0, 1])
        precision, recall, class_f1, _ = precision_recall_fscore_support(
            y_eval,
            predictions,
            labels=[0, 1],
            zero_division=0,
        )

        report = {
            "f1_score": best_f1,
            "accuracy": accuracy,
            "f1_default_0_5": f1_default,
            "best_threshold": best_threshold,
            "positive_class_ratio": positive_ratio,
            "reference_positive_ratio": REFERENCE_POSITIVE_RATIO,
            "drift_delta": drift_delta,
            "drift_detected": drift_detected,
            "precision_class_0": float(precision[0]),
            "recall_class_0": float(recall[0]),
            "precision_class_1": float(precision[1]),
            "recall_class_1": float(recall[1]),
            "confusion_matrix": matrix.tolist(),
        }

        mlflow.log_metrics(
            {
                "f1_score": best_f1,
                "accuracy": accuracy,
                "f1_default_0_5": f1_default,
                "best_threshold": best_threshold,
                "positive_class_ratio": positive_ratio,
                "drift_delta": drift_delta,
                "precision_class_0": float(precision[0]),
                "recall_class_0": float(recall[0]),
                "precision_class_1": float(precision[1]),
                "recall_class_1": float(recall[1]),
            }
        )
        mlflow.sklearn.log_model(model, "model")

        Path("outputs").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("outputs/report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _write_detail_report(matrix, precision, recall, class_f1)
        joblib.dump(
            {
                "model": model,
                "threshold": best_threshold,
                "feature_names": FEATURE_NAMES,
            },
            "models/model.joblib",
        )

        print(
            f"F1 optimized: {best_f1:.4f} | F1 at 0.5: {f1_default:.4f} | "
            f"Accuracy: {accuracy:.4f} | Best threshold: {best_threshold:.2f}"
        )

    return best_f1


def main() -> None:
    with open("params.yaml", encoding="utf-8") as file:
        params = yaml.safe_load(file)
    train(params)


if __name__ == "__main__":
    main()
