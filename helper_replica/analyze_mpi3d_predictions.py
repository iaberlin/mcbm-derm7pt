from pathlib import Path
import csv
import json

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS = {
    "vanilla_res18": ROOT / (
        "results/mpi3d-vanilla-res20-diagnostic/42/"
        "predictions/epoch_50.pth"
    ),
    "cbm_res18": ROOT / (
        "results/mpi3d-cbm-res20-diagnostic/42/"
        "predictions/epoch_50.pth"
    ),
    "mcbm_res18": ROOT / (
        "results/mpi3d-mcbm-res20-diagnostic/42/"
        "predictions/epoch_50.pth"
    ),
}

OUTPUT_DIR = ROOT / "results/mpi3d_metrics_analysis/res20-42"

CONCEPT_FACTORS = {
    "object_color": (0, 6),
    "object_size": (6, 8),
    "camera_height": (8, 11),
    "background_color": (11, 14),
}

TASK_CLASSES = list(range(6))


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def calculate_metrics(y_true, y_pred, classes):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError("Labels und Predictions haben unterschiedliche Längen.")

    cm = confusion_matrix(
        y_true, y_pred, labels=classes
    )

    recalls = recall_score(
        y_true,
        y_pred,
        labels=classes,
        average=None,
        zero_division=0,
    )

    support = cm.sum(axis=1)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true, y_pred,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": [
            {
                "class": int(cls),
                "recall": float(recall),
                "support": int(n),
            }
            for cls, recall, n in zip(classes, recalls, support)
        ],
        "confusion_matrix": cm.tolist(),
    }


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)




OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = {}
summary_rows = []
class_rows = []

for model_name, path in PREDICTIONS.items():

    if not path.is_file():
        print(f"[SKIP] Datei fehlt: {path}")
        continue

    print(f"\n=== {model_name} ===")

    # Nur eigene, vertrauenswürdige Prediction-Dateien laden.
    data = torch.load(str(path), map_location="cpu")

    y_true = to_numpy(data["y"]).reshape(-1)
    y_scores = to_numpy(data["y_preds"])

    if y_scores.ndim != 2 or y_scores.shape[1] != 6:
        raise ValueError(
            f"Unerwartete Task-Form: {y_scores.shape}"
        )

    y_pred = y_scores.argmax(axis=1)

    task_metrics = calculate_metrics(
        y_true, y_pred, TASK_CLASSES
    )

    model_result = {
        "prediction_file": str(path),
        "n_test": len(y_true),
        "task": task_metrics,
        "concepts": {},
    }

    def add_result(category, factor, metrics):
        summary_rows.append({
            "model": model_name,
            "category": category,
            "factor": factor,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
        })

        for item in metrics["per_class"]:
            class_rows.append({
                "model": model_name,
                "category": category,
                "factor": factor,
                "class": item["class"],
                "recall": item["recall"],
                "support": item["support"],
            })

        matrix_path = (
            OUTPUT_DIR /
            f"confusion_{model_name}_{factor}.csv"
        )

        matrix = np.asarray(metrics["confusion_matrix"])
        np.savetxt(
            matrix_path,
            matrix,
            delimiter=",",
            fmt="%d",
        )

    add_result("task", "object_shape", task_metrics)

    print(
        f"Task: Accuracy={task_metrics['accuracy']:.4%}, "
        f"Balanced Accuracy={task_metrics['balanced_accuracy']:.4%}, "
        f"Macro-F1={task_metrics['macro_f1']:.4f}"
    )

    # Vanilla hat keinen expliziten Concept Head.
    if "c_preds" in data and "c" in data:

        c_true = to_numpy(data["c"])
        c_scores = to_numpy(data["c_preds"])

        if c_scores.ndim == 3 and c_scores.shape[-1] == 1:
            c_scores = c_scores[..., 0]

        if c_true.shape != (len(y_true), 14):
            raise ValueError(
                f"Unerwartete Concept-Label-Form: {c_true.shape}"
            )

        if c_scores.shape != c_true.shape:
            raise ValueError(
                f"Concept-Output passt nicht zu Labels: "
                f"{c_scores.shape} vs. {c_true.shape}"
            )

        for factor, (start, end) in CONCEPT_FACTORS.items():

            true_factor = c_true[:, start:end]
            pred_factor = c_scores[:, start:end]

            # Prüfen, ob die Labels pro Faktor one-hot sind.
            if not np.all(true_factor.sum(axis=1) == 1):
                raise ValueError(
                    f"Concept-Labels für {factor} sind "
                    "nicht korrekt one-hot kodiert."
                )

            factor_metrics = calculate_metrics(
                true_factor.argmax(axis=1),
                pred_factor.argmax(axis=1),
                list(range(end - start)),
            )

            model_result["concepts"][factor] = factor_metrics

            add_result("concept", factor, factor_metrics)

            print(
                f"{factor}: "
                f"Accuracy={factor_metrics['accuracy']:.4%}, "
                f"Macro-F1={factor_metrics['macro_f1']:.4f}"
            )

    results[model_name] = model_result



with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(results, f, indent=2)

save_csv(
    OUTPUT_DIR / "metrics_summary.csv",
    summary_rows,
    [
        "model",
        "category",
        "factor",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ],
)

save_csv(
    OUTPUT_DIR / "metrics_per_class.csv",
    class_rows,
    [
        "model",
        "category",
        "factor",
        "class",
        "recall",
        "support",
    ],
)

print(f"\Results saved to: {OUTPUT_DIR}")