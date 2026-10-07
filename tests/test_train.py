import json
import os

import numpy as np
import pandas as pd

from src.train import (
    DRIFT_TOLERANCE,
    FEATURE_NAMES,
    REFERENCE_POSITIVE_RATIO,
    _check_data_drift,
    _select_threshold,
    train,
)


def _make_temp_data(tmp_path, positive_ratio: float = 0.5):
    rng = np.random.default_rng(0)
    n = 200
    X = rng.random((n, len(FEATURE_NAMES)))
    y = np.zeros(n, dtype=int)
    y[: int(n * positive_ratio)] = 1
    rng.shuffle(y)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "holdout.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)
    return train_path, eval_path


def _train_small(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    return train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )


def test_train_returns_float(tmp_path):
    f1 = _train_small(tmp_path)
    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0


def test_report_and_detail_files_created(tmp_path):
    _train_small(tmp_path)
    assert os.path.exists("outputs/report.json")
    assert os.path.exists("outputs/detail.txt")

    with open("outputs/report.json", encoding="utf-8") as file:
        report = json.load(file)
    expected = {
        "f1_score",
        "accuracy",
        "f1_default_0_5",
        "best_threshold",
        "positive_class_ratio",
        "drift_detected",
        "precision_class_0",
        "recall_class_0",
        "precision_class_1",
        "recall_class_1",
        "confusion_matrix",
    }
    assert expected.issubset(report)
    assert 0.10 <= report["best_threshold"] <= 0.90

    with open("outputs/detail.txt", encoding="utf-8") as file:
        detail = file.read()
    assert "CONFUSION MATRIX" in detail
    assert "class 1" in detail


def test_model_bundle_created(tmp_path):
    _train_small(tmp_path)
    assert os.path.exists("models/model.joblib")


def test_threshold_scan_uses_best_f1():
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.30, 0.40, 0.90])
    threshold, score = _select_threshold(y_true, probabilities)
    assert threshold == 0.4
    assert score == 1.0


def test_data_drift_boundary_and_warning(capsys):
    delta, detected = _check_data_drift(REFERENCE_POSITIVE_RATIO + DRIFT_TOLERANCE)
    assert np.isclose(delta, DRIFT_TOLERANCE)
    assert detected is False
    assert "check passed" in capsys.readouterr().out

    delta, detected = _check_data_drift(
        REFERENCE_POSITIVE_RATIO + DRIFT_TOLERANCE + 0.001
    )
    assert delta > DRIFT_TOLERANCE
    assert detected is True
    assert "WARNING" in capsys.readouterr().out
