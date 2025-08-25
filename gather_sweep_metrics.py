import glob
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

import wandb
from utils.metrics import get_all_fold_metrics, get_all_metrics, get_mcc_threhsold

LEVEL_CLASSES = yaml.safe_load(open("metadata/level_classes.yaml"))

AGG_METHODS = [
    "MaxPool",
    "MeanPool",
    "LightAttentionPool",
    "MultiHeadAttentionPool",
]

PARAMETERS = [
    "exp_name",
    "category_level",
    "metadata_file",
    "clip_len",
    "agg_method",
    "loss",
    "mlp_dropout",
]
PPI_PARAMETERS = PARAMETERS + [
    "ppi_database",
    "ppi_test",
    "graph_layer",
    "num_neighbors",
    "n_graph_layers",
]


def collect_experiment_results(run_exp_folder, categories):
    all_folds_thresholds = []
    all_folds_pred_dfs = []
    for ho_fold in range(5):
        fold_run_exp_folder = f"{run_exp_folder}/fold_{ho_fold}"

        assert os.path.exists(
            f"{fold_run_exp_folder}/fold_{ho_fold}_val_predictions.csv"
        ), "Val file doesn't exist"

        val_preds_df = pd.read_csv(
            f"{fold_run_exp_folder}/fold_{ho_fold}_val_predictions.csv"
        )
        val_fold_true = val_preds_df[[f"{cat}_true" for cat in categories]].values
        val_fold_pred = F.sigmoid(
            torch.from_numpy(val_preds_df[[f"{cat}_pred" for cat in categories]].values)
        ).numpy()

        thresholds = get_mcc_threhsold(val_fold_true, val_fold_pred)
        all_folds_thresholds.append(thresholds)

        test_preds_df = pd.read_csv(
            f"{fold_run_exp_folder}/fold_{ho_fold}_test_predictions.csv", index_col=0
        )
        test_preds_df = test_preds_df.drop(columns="seq").groupby(["id"]).mean()
        test_preds_df = test_preds_df.rename(
            columns={col: f"{col}_fold_{ho_fold}" for col in test_preds_df.columns}
        )

        if ho_fold == 0:
            all_folds_pred_dfs = test_preds_df
        else:
            all_folds_pred_dfs = pd.merge(
                all_folds_pred_dfs, test_preds_df, left_index=True, right_index=True
            )

    all_folds_true = []
    all_folds_preds = []
    for ho_fold in range(5):
        true_cols = [f"{cat}_true_fold_{ho_fold}" for cat in categories]
        all_folds_true.append(all_folds_pred_dfs[true_cols].values)

        pred_columns = [f"{cat}_pred_fold_{ho_fold}" for cat in categories]
        all_folds_preds.append(
            F.sigmoid(torch.from_numpy(all_folds_pred_dfs[pred_columns].values)).numpy()
        )
    assert np.any(
        [np.all(array == all_folds_true[0]) for array in all_folds_true[1:]] != True
    ), "true values don't match"

    all_fold_metrics = get_all_fold_metrics(
        all_folds_true[0], all_folds_preds, all_folds_thresholds, categories
    )

    overall_metrics = {k: v for k, v in all_fold_metrics.items() if "perclass" not in k}
    overall_metrics_df = pd.DataFrame.from_dict(overall_metrics, orient="index").T
    overall_metrics_df.to_csv(
        f"{run_exp_folder}/all_folds_overall_metrics.csv", index=False
    )

    perclass_metrics = {k: v for k, v in all_fold_metrics.items() if "perclass" in k}
    perclass_metrics_df = pd.DataFrame.from_dict(perclass_metrics)
    perclass_metrics_df = perclass_metrics_df.rename(
        columns={k: k.replace("_perclass", "") for k in perclass_metrics_df.columns}
    )
    perclass_metrics_df.to_csv(
        f"{run_exp_folder}/all_folds_perclass_metrics.csv", index=False
    )
    return overall_metrics


def get_sweep_metrics(exp_folder, save_folder, runs):
    for i, run in tqdm(enumerate(runs), total=len(runs)):
        if run.State != "finished":
            print(f"{run.id} didn't finish. skipping...")
            continue
        if run.config["agg_method"] not in AGG_METHODS:
            continue

        run_config = run.config

        run_exp_folder = (
            exp_folder
            + "/"
            + run_config["exp_name"]
            + "_"
            + os.path.basename(run_config["metadata_file"]).split(".")[0]
            + "/"
            + run.id
        )

        run_config = {k: v for k, v in run_config.items() if k in PARAMETERS}
        run_config["metadata_file"] = os.path.basename(
            run_config["metadata_file"]
        ).split(".")[0]
        run_config["run_id"] = run.id

        categories = LEVEL_CLASSES[run_config["category_level"]]
        n_categories = len(categories)

        run_overall_metric = collect_experiment_results(run_exp_folder, categories)

        run_config.update(run_overall_metric)
        run_metric_df = pd.DataFrame.from_dict(run_config, orient="index").T

        if i == 0:
            all_runs_metric_df = run_metric_df
        else:
            all_runs_metric_df = pd.concat(
                [all_runs_metric_df, run_metric_df], ignore_index=True
            )

        print(all_runs_metric_df.tail())

        all_runs_metric_df.to_csv(f"{save_folder}/all_sweep_metrics.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on sweep configs.")
    parser.add_argument('--exp_folder', type=str, required=True, help='Path to experiment folder')
    parser.add_argument('--sweep_save_folder', type=str, required=True, help='Path to save results (metrics/configs)')
    args = parser.parse_args()
    exp_folder = args.exp_folder
    sweep_save_folder = args.sweep_save_folder
    
    #exp_folder = "/scratch/groups/emmalu/seq2loc/sweep_experiments"
    #sweep_save_folder = "/scratch/groups/emmalu/seq2loc/sweep_analysis"

    api = wandb.Api()
    entity = api.default_entity
    runs = api.runs(f"{entity}/seq2loc_sweep")
    get_sweep_metrics(exp_folder, sweep_save_folder, runs)
