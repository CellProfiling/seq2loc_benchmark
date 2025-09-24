import argparse
import os
import random
import warnings

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from joblib import Parallel, delayed
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from data.collate_fn import collate_fn, test_collate_fn
from data.dataset import EmbeddingDataset
from models.feat2loc_model import get_agg_model
from models.mlp import MLPClassifier
from utils.metrics import get_all_fold_metrics, get_all_metrics, get_mcc_threhsold

warnings.filterwarnings("ignore", category=UserWarning)


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

LEVEL_CLASSES = yaml.safe_load(open("datasets/final/hierarchical_label_set.yaml"))


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def get_embedding_dim(exp_name):
    if exp_name == "ProtT5":
        return 1024
    elif exp_name == "ProtBert":
        return 1024
    elif exp_name == "ESM2":
        return 2560
    elif exp_name == "ESM3":
        return 1536
    elif "SubCell" in exp_name:
        return 1536
    elif "DINO" in exp_name:
        return 768
    else:
        raise ValueError("Invalid exp_name")


def load_model_weights(model, model_path, prefix):
    model_weights = torch.load(model_path, map_location="cpu", weights_only=False)
    filtered_weights = {
        k[len(prefix) + 1 :]: v
        for k, v in model_weights["state_dict"].items()
        if k.startswith(prefix)
    }
    op = model.load_state_dict(filtered_weights)
    print(op, flush=True)
    return model


def main(data_folder, exp_folder, embedding_folder, config):
    exp_folder = (
        exp_folder
        + "/"
        + config["exp_name"]
        + "_"
        + config["metadata_file"]
        + "/"
        + config["run_id"]
    )

    embedding_dim = get_embedding_dim(config["exp_name"])
    categories = LEVEL_CLASSES[config["category_level"]]
    n_categories = len(categories)

    all_folds_thresholds = []
    all_folds_preds = []
    for ho_fold in range(5):
        fold_exp_folder = f"{exp_folder}/fold_{ho_fold}"
        if os.path.exists(f"{fold_exp_folder}/fold_{ho_fold}_val_predictions.csv"):
            print(f"Loading predictions for fold {ho_fold}", flush=True)
            val_preds_df = pd.read_csv(
                f"{fold_exp_folder}/fold_{ho_fold}_val_predictions.csv"
            )
            val_fold_true = val_preds_df[[f"{cat}_true" for cat in categories]].values
            val_fold_pred = F.sigmoid(
                torch.from_numpy(
                    val_preds_df[[f"{cat}_pred" for cat in categories]].values
                )
            ).numpy()
            fold_thresholds = get_mcc_threhsold(val_fold_true, val_fold_pred)
            all_folds_thresholds.append(fold_thresholds)

            test_preds_df = pd.read_csv(
                f"{fold_exp_folder}/fold_{ho_fold}_test_predictions.csv"
            )
            test_fold_true = test_preds_df[[f"{cat}_true" for cat in categories]].values
            test_fold_pred = F.sigmoid(
                torch.from_numpy(
                    test_preds_df[[f"{cat}_pred" for cat in categories]].values
                )
            ).numpy()
            all_folds_preds.append(test_fold_pred)
        else:
            exp_embedding_folder = (
                f"{embedding_folder}/{config['exp_name']}-4k.h5"
                if config["exp_name"] != "ESM3"
                else f"{embedding_folder}/{config['exp_name']}-3k.h5"
            )
            print(f"Running inference for fold {ho_fold}", flush=True)
            fold_model_path = f"{fold_exp_folder}/models/best_model_acc.ckpt"

            valid_dataset = EmbeddingDataset(
                exp_embedding_folder,
                f"{data_folder}/{config['metadata_file']}.csv",
                config["category_level"],
                folds=[ho_fold],
                clip_len=config["clip_len"],
                # test_mode=True,
            )
            valid_loader = DataLoader(
                valid_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
            )

            test_dataset = EmbeddingDataset(
                exp_embedding_folder,
                f"{data_folder}/hou_testset.csv",
                config["category_level"],
                clip_len=config["clip_len"],
                test_mode=True,
            )
            test_loader = DataLoader(
                test_dataset, batch_size=64, shuffle=False, collate_fn=test_collate_fn
            )

            agg_model = get_agg_model(
                config["agg_method"], embedding_dim, config["clip_len"]
            )
            agg_model = load_model_weights(agg_model, fold_model_path, "model")

            mlp_embedding_dim = (
                embedding_dim * 2
                if config["agg_method"] == "LightAttentionPool"
                else embedding_dim
            )
            mlp_config = {
                "input_dim": mlp_embedding_dim,
                "num_classes": n_categories,
                "hidden_dim": 512,
                "num_hidden_layers": 2,
                "dropout": 0.2,
            }
            mlp_model = MLPClassifier(**mlp_config)
            mlp_model = load_model_weights(mlp_model, fold_model_path, "mlp_classifier")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            mlp_model.to(device)
            with torch.no_grad():
                agg_model.eval()
                mlp_model.eval()

                fold_val_true = []
                fold_val_pred = []
                for i, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    embeddings, masks, locations = batch
                    pooled_embedding, attention = agg_model(
                        embeddings.to(device), masks.to(device)
                    )
                    preds = mlp_model(pooled_embedding)
                    preds = torch.sigmoid(preds)

                    fold_val_true.append(locations.cpu().numpy())
                    fold_val_pred.append(preds.cpu().numpy())

                fold_val_true = np.concatenate(fold_val_true, axis=0)
                fold_val_pred = np.concatenate(fold_val_pred, axis=0)

                val_preds_df = pd.DataFrame()
                for i, cat in enumerate(categories):
                    val_preds_df[f"{cat}_true"] = fold_val_true[:, i]
                    val_preds_df[f"{cat}_pred"] = fold_val_pred[:, i]

                val_preds_df.to_csv(
                    f"{fold_exp_folder}/val_fold_{ho_fold}_predictions.csv", index=False
                )

                fold_thresholds = get_mcc_threhsold(fold_val_true, fold_val_pred)
                all_folds_thresholds.append(fold_thresholds)

                test_fold_id = []
                test_fold_seq = []
                test_fold_true = []
                test_fold_pred = []
                test_fold_attn = []
                for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
                    id, seq, embedding, mask, locations = batch

                    pooled_embedding, attention = agg_model(
                        embedding.to(device), mask.to(device)
                    )
                    preds = mlp_model(pooled_embedding)
                    preds = torch.sigmoid(preds)

                    test_fold_id.extend(id)
                    test_fold_seq.extend(seq)

                    test_fold_true.append(locations)
                    test_fold_pred.append(preds)
                    test_fold_attn.append(attention)

                test_fold_true = torch.cat(test_fold_true, dim=0).cpu().numpy()
                test_fold_pred = torch.cat(test_fold_pred, dim=0).cpu().numpy()
                test_fold_attn = torch.cat(test_fold_attn, dim=0)

                all_folds_preds.append(test_fold_pred)

                test_preds_df = pd.DataFrame(
                    {"id": test_fold_id, "sequence": test_fold_seq}
                )
                for i, cat in enumerate(categories):
                    test_preds_df[f"{cat}_true"] = test_fold_true[:, i]
                    test_preds_df[f"{cat}_pred"] = test_fold_pred[:, i]

                test_preds_df.to_csv(
                    f"{fold_exp_folder}/test_fold_{ho_fold}_predictions.csv",
                    index=False,
                )

            torch.save(test_fold_attn, f"{fold_exp_folder}/fold_{ho_fold}_attention.pt")

        test_fold_pred_bin = (test_fold_pred > fold_thresholds).astype(np.int16)
        fold_metrics = get_all_metrics(
            test_fold_true, test_fold_pred, test_fold_pred_bin, categories=categories
        )

        overall_metrics = {k: v for k, v in fold_metrics.items() if "perclass" not in k}
        overall_metrics_df = pd.DataFrame.from_dict(overall_metrics, orient="index").T
        overall_metrics_df.to_csv(
            f"{exp_folder}/fold_{ho_fold}_overall_metrics.csv", index=False
        )

        perclass_metrics = {k: v for k, v in fold_metrics.items() if "perclass" in k}
        perclass_metrics_df = pd.DataFrame.from_dict(perclass_metrics)
        perclass_metrics_df["category"] = perclass_metrics["category_perclass"]
        perclass_metrics_df.to_csv(
            f"{exp_folder}/fold_{ho_fold}_perclass_metrics.csv", index=False
        )

        np.save(f"{exp_folder}/all_thresholds.npy", np.array(all_folds_thresholds))

    all_fold_metrics = get_all_fold_metrics(
        test_fold_true, all_folds_preds, all_folds_thresholds, categories
    )

    overall_metrics = {k: v for k, v in all_fold_metrics.items() if "perclass" not in k}
    overall_metrics_df = pd.DataFrame.from_dict(overall_metrics, orient="index").T
    overall_metrics_df.to_csv(
        f"{exp_folder}/all_folds_overall_metrics.csv", index=False
    )

    perclass_metrics = {k: v for k, v in all_fold_metrics.items() if "perclass" in k}
    perclass_metrics_df = pd.DataFrame.from_dict(perclass_metrics)
    perclass_metrics_df = perclass_metrics_df.rename(
        columns={k: k.replace("_perclass", "") for k in perclass_metrics_df.columns}
    )
    perclass_metrics_df.to_csv(
        f"{exp_folder}/all_folds_perclass_metrics.csv", index=False
    )
    return overall_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on sweep configs.")
    parser.add_argument('--data_folder', type=str, required=True, help='Path to data folder')
    parser.add_argument('--exp_folder', type=str, required=True, help='Path to experiment folder')
    parser.add_argument('--embedding_folder', type=str, required=True, help='Path to embedding folder')
    parser.add_argument('--save_folder', type=str, required=True, help='Path to save results (metrics/configs)')

    args = parser.parse_args()

    data_folder = args.data_folder
    exp_folder = args.exp_folder
    embedding_folder = args.embedding_folder
    save_folder = args.save_folder

    if os.path.exists(f"{save_folder}/sweep_config.csv"):
        df = pd.read_csv(f"{save_folder}/sweep_config.csv")
    else:
        api = wandb.Api(timeout=29)
        entity = api.default_entity
        runs = api.runs(f"{entity}/seq2loc_sweep")

        all_run_config = []
        for i, run in tqdm(enumerate(runs), total=len(runs)):
            if run.State != "finished":
                print(f"{run.id} didn't finish. skipping...")
                continue
            if run.config["agg_method"] not in AGG_METHODS:
                continue

            run_config = run.config

            run_config = {k: v for k, v in run_config.items() if k in PARAMETERS}
            run_config["metadata_file"] = os.path.basename(
                run_config["metadata_file"]
            ).split(".")[0]
            run_config["run_id"] = run.id
            all_run_config.append(run_config)

        df = pd.DataFrame.from_dict(all_run_config)
        df.to_csv(f"{save_folder}/sweep_config.csv", index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    all_metrics = []

    def process_row(row):
        config = row[PARAMETERS + ["run_id"]].to_dict()
        print(f"Running configuration:", {k: v for k, v in config.items()}, flush=True)
        row_metrics = main(data_folder, exp_folder, embedding_folder, config)
        config.update(row_metrics)
        return pd.DataFrame.from_dict(config, orient="index").T

    num_jobs = -1  # Number of parallel jobs

    results = Parallel(n_jobs=num_jobs)(
        delayed(process_row)(row) for _, row in tqdm(df.iterrows(), total=len(df))
    )

    all_metrics_df = pd.concat(results, ignore_index=True)
    print(all_metrics_df, flush=True)

    all_metrics_df.to_csv(f"{save_folder}/overall_metrics.csv", index=False)
