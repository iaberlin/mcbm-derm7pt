import os
import uuid

from src.helpers.save_eval import save_job_evaluation
from src.datasets import get_loader
from src.helpers.metrics import calc_all_metrics, calc_concept_metrics
from .train import TrainExperiment


class TestExperiment(TrainExperiment):

    experiment_name = "test"
    wandb_offline = False

    def __init__(self, **kwargs) -> None:

        super().__init__(continue_training=True, **kwargs)

    def _prepare_labels_preds(self, loader):
        test_set = super()._prepare_labels_preds(loader)

        if loader is self.test_loader:
            self.test_set = test_set

        return test_set

    #==========Run==========
    def run(self):
        self.save = False
        losses_test = self.test_epoch()
        metrics_eval = self.evaluate(self.ini_epoch)


        y_true = self.test_set["y"]
        y_pred = self.test_set["y_preds"].argmax(dim=1)

        task_metrics = calc_all_metrics(y_true, y_pred)

        print("\nTask Evaluation:")
        print(task_metrics)

        concept_metrics = {}

        if getattr(self.model, "has_concepts", False):
            c_true = self.test_set["c"]
            c_preds = self.test_set["c_preds"]

            concept_metrics = calc_concept_metrics(c_true, c_preds)

            print("\nConcept Evaluation:")
            print(concept_metrics)


        job_id = os.environ.get("SLURM_JOB_ID")

        if job_id is None:
            job_id = f"local-{uuid.uuid4().hex[:8]}"

        run_id = (
            f"{self.cfg['data']['dataset']}-"
            f"{self.cfg['model']['model_type']}-"
            f"{self.cfg['model']['encoder']['arch']}-"
            f"{self.seed}-{job_id}"
        )

        # Ergebnisse für die CSV zusammenstellen
        row = {
            "run_id": run_id,
            "job_id": job_id,
            "dataset": self.cfg["data"]["dataset"],
            "dataset": self.cfg["data"]["dataset"],
            "model": self.cfg["model"]["model_type"],
            "backbone": self.cfg["model"]["encoder"]["arch"],
            "train_seed": self.seed,
            "split_seed": self.split_seed,
            "config": self.config_file,

            "task_accuracy": task_metrics["accuracy"],
            "task_balanced_accuracy": task_metrics["balanced_accuracy"],
            "task_macro_f1": task_metrics["macro_f1"],
            "task_recall_per_class": task_metrics["recall_per_class"],
            "task_support_per_class": task_metrics["support_per_class"],
            "task_confusion_matrix": task_metrics["confusion_matrix"],

            **concept_metrics,

            "checkpoint_path": os.path.join(self.model_dir, f"epoch_{self.ini_epoch}.pt"),

        }

        save_job_evaluation(row, run_id)


        log_dict = {**losses_test, **metrics_eval}
        self.wandb_run.log(log_dict)
        print("Epoch {}".format(self.ini_epoch))
        print('\n'.join(["{}:\t{}".format(k, v) for k, v in log_dict.items()])+'\n')