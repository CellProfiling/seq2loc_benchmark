import argparse
import copy
import glob
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import yaml

from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

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
    "graph_layer",
    "graph_dropout",
    "n_graph_layers",
    "num_neighbors",
    "ppi_database",
    "ppi_test",
]
METRICS = [
    "macro_ap",
    "micro_ap",
    "acc",
    "acc_samples",
    "f1_macro",
    "f1_micro",
    "jaccard_macro",
    "jaccard_micro",
    "rocauc_macro",
    "rocauc_micro",
    "mlrap",
    "coverage_error",
    "num_labels",
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


def load_all_models(
    exp_folder, agg_method, graph_layer, embedding_dim, n_graph_layers, mlp_hidden_dim
):
    agg_model = get_agg_model(agg_method, embedding_dim)
    agg_model_path = f"{exp_folder}/models/best_model_acc.ckpt"
    agg_model = load_model_weights(agg_model, agg_model_path, "model")

    graph_model = get_graph_model(
        graph_layer,
        embedding_dim,
        num_layers=n_graph_layers,
        dropout=0.1,
    )
    graph_model = load_model_weights(graph_model, agg_model_path, "graph_model")

    mlp_embedding_dim = (
        embedding_dim * 2 if agg_method == "LightAttentionPool" else embedding_dim
    )
    mlp_config = {
        "input_dim": mlp_embedding_dim,
        "num_classes": len(LEVEL_CLASSES["subcellular"]),
        "hidden_dim": mlp_hidden_dim,
        "num_hidden_layers": 2,
        "dropout": 0.1,
    }
    mlp_model = MLPClassifier(**mlp_config)
    mlp_model = load_model_weights(mlp_model, agg_model_path, "mlp_classifier")

    return agg_model, graph_model, mlp_model


def forward_graph_model(
    agg_model, graph_model, mlp_model, embeddings, masks, edge_index, locations
):
    pooled_embedding, attention = agg_model(embeddings, masks)
    graph_embedding = graph_model(pooled_embedding, edge_index)
    preds = mlp_model(graph_embedding)
    preds = torch.sigmoid(preds)
    return preds, attention


def process_loader(loader, embedding_loader, agg_model, graph_model, mlp_model, device):
    all_ids = []
    all_seqs = []
    all_true = []
    all_pred = []
    all_attn = []
    for batch in tqdm(loader, total=len(loader)):
        prot_idxs = batch.x
        locations = batch.y
        edge_index = batch.edge_index
        prot_ids, seqs, embeddings, masks = (
            embedding_loader.load_batch_embedding_and_mask(
                prot_idxs, return_metadata=True
            )
        )
        preds, attention = forward_graph_model(
            agg_model,
            graph_model,
            mlp_model,
            embeddings.to(device),
            masks.to(device),
            edge_index.to(device),
            locations.to(device),
        )
        all_ids.extend(prot_ids)
        all_seqs.extend(seqs)
        all_true.append(locations.cpu().numpy())
        all_pred.append(preds.cpu().numpy())
        all_attn.append(attention.cpu().numpy())

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    all_attn = np.concatenate(all_attn, axis=0)
    return all_ids, all_seqs, all_true, all_pred, all_attn


def main(exp_folder, embedding_folder, config):
    print(f"Running inference for {exp_folder} with config: {config}", flush=True)

    inference_neighbour_configs = ["train_ppi", "test_ppi", "all_ppi", "no_ppi"]

    all_overall_metrics = []
    for inference_neighbour_config in inference_neighbour_configs:
        if not os.path.exists(
            f"{exp_folder}/inference_{inference_neighbour_config}_all_folds_overall_metrics.csv"
        ):
            from torch_geometric.loader import NeighborLoader
            from data.ppi_dataset import (
                LEVEL_CLASSES,
                EmbeddingLoader,
                get_ppi_test_dataset,
            )
            from models.graph_feat2loc_model import get_agg_model, get_graph_model

            embedding_dim = get_embedding_dim(config["exp_name"])
            embedding_loader = EmbeddingLoader(
                embeddings_file=config["embeddings_file"],
                metadata_files=[config["metadata_file"], config["testset_file"]],
                clip_len=config["clip_len"],
            )
            # print(embedding_loader.keys2idx.keys())
            categories = LEVEL_CLASSES[config["category_level"]]
            n_categories = len(categories)

            all_folds_thresholds = []
            all_folds_preds = []
            for ho_fold in range(5):
                fold_exp_folder = f"{exp_folder}/fold_{ho_fold}"

                train_folds = [i for i in range(5) if i != ho_fold]
                val_fold = [ho_fold]
                print(
                    f"Running inference for fold {ho_fold} and inference config {inference_neighbour_config}",
                    flush=True,
                )
                fold_model_path = f"{fold_exp_folder}/models/best_model_acc.ckpt"

                fold_exp_inf_folder = (
                    f"{fold_exp_folder}/inference_{inference_neighbour_config}"
                )
                if not os.path.exists(fold_exp_inf_folder):
                    os.makedirs(fold_exp_inf_folder)

                metadata = pd.read_csv(config["metadata_file"])
                val_data = metadata[metadata["fold"].isin(val_fold)].reset_index(
                    drop=True
                )

                if inference_neighbour_config == "train_ppi":
                    link_data = metadata[
                        metadata["fold"].isin(train_folds)
                    ].reset_index(drop=True)
                elif inference_neighbour_config == "test_ppi":
                    link_data = copy.deepcopy(val_data)
                elif inference_neighbour_config == "all_ppi":
                    link_data = pd.concat([metadata, val_data], ignore_index=True)
                else:
                    link_data = None

                val_dataset = get_ppi_test_dataset(
                    data_df=val_data,
                    link_df=link_data,
                    ppi=config["ppi_database"],
                    category_level=config["category_level"],
                    isoforms2idx=embedding_loader.keys2idx,
                )
                val_loader = NeighborLoader(
                    val_dataset,
                    num_neighbors=config["num_neighbors"],
                    batch_size=2,
                    input_nodes=None,
                    shuffle=False,
                    num_workers=8,
                    persistent_workers=True,
                )

                test_data = pd.read_csv(config["testset_file"])
                test_dataset = get_ppi_test_dataset(
                    data_df=test_data,
                    link_df=link_data,
                    ppi=config["ppi_database"],
                    category_level=config["category_level"],
                    isoforms2idx=embedding_loader.keys2idx,
                )
                test_loader = NeighborLoader(
                    test_dataset,
                    num_neighbors=config["num_neighbors"],
                    batch_size=2,
                    input_nodes=None,
                    shuffle=False,
                    num_workers=8,
                    persistent_workers=True,
                )

                agg_model = get_agg_model(
                    config["agg_method"], embedding_dim, config["clip_len"]
                )
                agg_model = load_model_weights(agg_model, fold_model_path, "model")

                graph_model = get_graph_model(
                    config["graph_layer"],
                    embedding_dim,
                    num_layers=config["n_graph_layers"],
                    dropout=config["graph_dropout"],
                )
                graph_model = load_model_weights(
                    graph_model, fold_model_path, "graph_model"
                )

                mlp_embedding_dim = (
                    embedding_dim * 2
                    if config["agg_method"] == "LightAttentionPool"
                    else embedding_dim
                )
                mlp_config = {
                    "input_dim": config["embedding_dim"],
                    "num_classes": n_categories,
                    "hidden_dim": config["mlp_hidden_dim"],
                    "num_hidden_layers": config["mlp_num_hidden_layers"],
                    "dropout": config["mlp_dropout"],
                }
                mlp_model = MLPClassifier(**mlp_config)
                mlp_model = load_model_weights(
                    mlp_model, fold_model_path, "mlp_classifier"
                )

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                agg_model.to(device)
                graph_model.to(device)
                mlp_model.to(device)

                with torch.no_grad():
                    agg_model.eval()
                    graph_model.eval()
                    mlp_model.eval()

                    (
                        val_fold_ids,
                        val_fold_seqs,
                        fold_val_true,
                        fold_val_pred,
                        fold_val_attn,
                    ) = process_loader(
                        val_loader,
                        embedding_loader,
                        agg_model,
                        graph_model,
                        mlp_model,
                        device,
                    )

                only_val_idxs = [
                    i
                    for i in range(len(val_fold_ids))
                    if val_fold_ids[i] in val_data["uniprot_id"].values
                ]

                val_fold_ids = [val_fold_ids[i] for i in only_val_idxs]
                val_fold_seqs = [val_fold_seqs[i] for i in only_val_idxs]
                fold_val_attn = fold_val_attn[only_val_idxs]

                fold_val_true = fold_val_true[only_val_idxs]
                fold_val_pred = fold_val_pred[only_val_idxs]

                fold_thresholds = get_mcc_threhsold(fold_val_true, fold_val_pred)
                all_folds_thresholds.append(fold_thresholds)

                val_preds_df = pd.DataFrame(
                    {"id": val_fold_ids, "sequence": val_fold_seqs}
                )
                for i, cat in enumerate(categories):
                    val_preds_df[f"{cat}_true"] = fold_val_true[:, i]
                    val_preds_df[f"{cat}_pred"] = fold_val_pred[:, i]

                val_preds_df = (
                    val_preds_df.groupby(["id", "sequence"]).mean().reset_index()
                )
                val_preds_df.to_csv(
                    f"{fold_exp_inf_folder}/val_fold_{ho_fold}_predictions.csv",
                    index=False,
                )

                fold_val_attn_df = pd.DataFrame(
                    fold_val_attn,
                    columns=[f"attention_{i}" for i in range(fold_val_attn.shape[1])],
                )
                fold_val_attn_df["id"] = val_fold_ids
                fold_val_attn_df["sequence"] = val_fold_seqs
                fold_val_attn_df = (
                    fold_val_attn_df.groupby(["id", "sequence"]).mean().reset_index()
                )
                fold_val_attn_df.to_csv(
                    f"{fold_exp_inf_folder}/fold_{ho_fold}_val_attention.csv",
                    index=False,
                )

                with torch.no_grad():
                    agg_model.eval()
                    graph_model.eval()
                    mlp_model.eval()

                    (
                        test_fold_ids,
                        test_fold_seqs,
                        fold_test_true,
                        fold_test_pred,
                        fold_test_attn,
                    ) = process_loader(
                        test_loader,
                        embedding_loader,
                        agg_model,
                        graph_model,
                        mlp_model,
                        device,
                    )
                only_test_idxs = [
                    i
                    for i in range(len(test_fold_ids))
                    if test_fold_ids[i] in test_data["uniprot_id"].values
                ]
                test_fold_ids = [test_fold_ids[i] for i in only_test_idxs]
                test_fold_seqs = [test_fold_seqs[i] for i in only_test_idxs]
                fold_test_attn = fold_test_attn[only_test_idxs]
                fold_test_true = fold_test_true[only_test_idxs]
                fold_test_pred = fold_test_pred[only_test_idxs]

                test_preds_df = pd.DataFrame(
                    {"id": test_fold_ids, "sequence": test_fold_seqs}
                )
                for i, cat in enumerate(categories):
                    test_preds_df[f"{cat}_true"] = fold_test_true[:, i]
                    test_preds_df[f"{cat}_pred"] = fold_test_pred[:, i]
                test_preds_df = (
                    test_preds_df.groupby(["id", "sequence"]).mean().reset_index()
                )

                fold_test_true = test_preds_df[
                    [f"{cat}_true" for cat in categories]
                ].values
                fold_test_pred = test_preds_df[
                    [f"{cat}_pred" for cat in categories]
                ].values
                all_folds_preds.append(fold_test_pred)

                test_preds_df.to_csv(
                    f"{fold_exp_inf_folder}/test_fold_{ho_fold}_predictions.csv",
                    index=False,
                )

                fold_test_attn_df = pd.DataFrame(
                    fold_test_attn,
                    columns=[f"attention_{i}" for i in range(fold_test_attn.shape[1])],
                )
                fold_test_attn_df["id"] = test_fold_ids
                fold_test_attn_df["sequence"] = test_fold_seqs
                fold_test_attn_df = (
                    fold_test_attn_df.groupby(["id", "sequence"]).mean().reset_index()
                )
                fold_test_attn_df.to_csv(
                    f"{fold_exp_inf_folder}/fold_{ho_fold}_test_attention.csv",
                    index=False,
                )

                test_fold_pred_bin = (fold_test_pred > fold_thresholds).astype(np.int16)
                fold_metrics = get_all_metrics(
                    fold_test_true,
                    fold_test_pred,
                    test_fold_pred_bin,
                    categories=categories,
                )

                overall_metrics = {
                    k: v for k, v in fold_metrics.items() if "perclass" not in k
                }
                overall_metrics_df = pd.DataFrame.from_dict(
                    overall_metrics, orient="index"
                ).T
                overall_metrics_df.to_csv(
                    f"{fold_exp_inf_folder}/fold_{ho_fold}_inference_{inference_neighbour_config}_overall_metrics.csv",
                    index=False,
                )

                perclass_metrics = {
                    k: v for k, v in fold_metrics.items() if "perclass" in k
                }
                perclass_metrics_df = pd.DataFrame.from_dict(perclass_metrics)
                perclass_metrics_df["category"] = perclass_metrics["category_perclass"]
                perclass_metrics_df.to_csv(
                    f"{fold_exp_inf_folder}/fold_{ho_fold}_inference_{inference_neighbour_config}_perclass_metrics.csv",
                    index=False,
                )

                np.save(
                    f"{fold_exp_inf_folder}/fold_{ho_fold}_inference_{inference_neighbour_config}_all_thresholds.npy",
                    np.array(all_folds_thresholds),
                )

            all_fold_metrics = get_all_fold_metrics(
                fold_test_true, all_folds_preds, all_folds_thresholds, categories
            )

            overall_metrics = {
                k: v for k, v in all_fold_metrics.items() if "perclass" not in k
            }
            overall_metrics_df = pd.DataFrame.from_dict(
                overall_metrics, orient="index"
            ).T
            overall_metrics_df.to_csv(
                f"{exp_folder}/inference_{inference_neighbour_config}_all_folds_overall_metrics.csv",
                index=False,
            )

            perclass_metrics = {
                k: v for k, v in all_fold_metrics.items() if "perclass" in k
            }
            perclass_metrics_df = pd.DataFrame.from_dict(perclass_metrics)
            perclass_metrics_df = perclass_metrics_df.rename(
                columns={
                    k: k.replace("_perclass", "") for k in perclass_metrics_df.columns
                }
            )
            perclass_metrics_df.to_csv(
                f"{exp_folder}/inference_{inference_neighbour_config}_all_folds_perclass_metrics.csv",
                index=False,
            )
        else:
            print(
                f"Skipping {inference_neighbour_config} for {exp_folder} as already exists",
                flush=True,
            )
            overall_metrics_df = pd.read_csv(
                f"{exp_folder}/inference_{inference_neighbour_config}_all_folds_overall_metrics.csv"
            )
        overall_metrics_df["inference_neighbour_config"] = inference_neighbour_config
        all_overall_metrics.append(overall_metrics_df)
    all_overall_metrics = pd.concat(all_overall_metrics, ignore_index=True)
    return all_overall_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on sweep configs.")
    parser.add_argument('--exp_folder', type=str, required=True, help='Path to experiment folder')
    parser.add_argument('--embedding_folder', type=str, required=True, help='Path to embedding folder')
    parser.add_argument('--save_folder', type=str, required=True, help='Path to save results (metrics/configs)')

    args = parser.parse_args()

    exp_folder = args.exp_folder
    embedding_folder = args.embedding_folder
    save_folder = args.save_folder

    #save_folder = "/scratch/groups/emmalu/seq2loc/ppi_analysis"
    #exp_folder = "/scratch/groups/emmalu/seq2loc/ppi_experiments/"
    #embedding_folder = "/scratch/groups/emmalu/seq2loc/embeddings/"

    all_exp_folders = sorted(glob.glob(f"{exp_folder}/*"))
    all_metrics = []

    for exp_folder in all_exp_folders:
        if not os.path.exists(f"{exp_folder}/config.yaml"):
            print(f"Skipping {exp_folder} as config.yaml not found", flush=True)
            continue
        if not "hpa_uniprot_combined_trainset" in exp_folder:
            continue
        print(f"Processing {exp_folder}")
        config = yaml.safe_load(open(f"{exp_folder}/config.yaml"))
        try:
            exp_metrics = main(exp_folder, embedding_folder, config)
            for parameter in PARAMETERS:
                exp_metrics[parameter] = (
                    # ",".join(map(str, config[parameter]))
                    config[parameter][0]
                    if type(config[parameter]) == list
                    else config[parameter]
                )
            exp_metrics["run_id"] = exp_folder.split("hpa_uniprot_combined_trainset_")[
                1
            ]
            all_metrics.append(exp_metrics)
        except Exception as e:
            print(f"Error processing {exp_folder}: {e}", flush=True)
            continue

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_metrics_df.to_csv(f"{save_folder}/overall_metrics.csv", index=False)
