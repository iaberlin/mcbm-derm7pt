import csv
import json
from pathlib import Path
import numpy as np


def prepare_value(value):
    if isinstance(value, dict):
        value = {str(k): prepare_value(v) for k, v in value.items()}
        return value

    if isinstance(value, (list, tuple)):
        return [prepare_value(v) for v in value]

    if isinstance(value, np.generic):
        return value.item()

    return value


def save_job_evaluation(row, job_id):
    master_path = Path("evaluation/evaluation-master.csv")
    output_dir = Path("evaluation/jobs")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(master_path, "r", newline="") as f:
        columns = next(csv.reader(f))


    # if set(row) - set(columns):
    #     raise ValueError("Unbekannte Spalten in den Evaluationsergebnissen")

    for key, value in row.items():

        if key in ["task_recall_per_class", "task_support_per_class"]:
            value = [[k, v] for k, v in value.items()]
            row[key] = json.dumps(prepare_value(value))

        elif isinstance(value, (dict, list, tuple)):
            row[key] = json.dumps(prepare_value(value))

        else:
            row[key] = prepare_value(value)

    output_path = output_dir / f"evaluation-{job_id}.csv"

    with open(output_path, "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)

    print(f"evaluation saved to: {output_path}")
