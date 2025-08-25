import argparse
import os
import random
import shutil

import lightning as L
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import DummyLogger
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd

import wandb
from data.ppi_dataset import EmbeddingLoader, get_ppi_dataset, LEVEL_CLASSES
from models.graph_feat2loc_model import GraphFeat2LocModel
from models.focal_loss import SigmoidFocalLoss
from models.feature_collection_cb import FeatureCollectionCallback
from torch_geometric.loader import NeighborLoader


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main(config=None):
    run = wandb.init(project="seq2loc_graph", config=config)

    exp_folder = (
        config["exp_folder"]
        + "/"
        + config["exp_name"]
        + "_"
        + os.path.basename(config["metadata_file"]).split(".")[0]
    )

    exp_folder = f"{exp_folder}_{run.id}"

    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    else:
        shutil.rmtree(exp_folder)
        os.makedirs(exp_folder)

    yaml.dump(config, open(f"{exp_folder}/config.yaml", "w"), default_flow_style=False)

    wandb_logger = WandbLogger(
        project="seq2loc_graph", name=config["exp_name"], config=config, dir=exp_folder
    )

    metrics = [
        "accuracy",
        "f1_score",
        "macro_ap",
        "micro_ap",
        "coverage_error",
        "mlrap",
    ]
    all_valid_metrics = pd.DataFrame(columns=metrics)
    all_test_metrics = pd.DataFrame(columns=metrics)

    embedding_loader = EmbeddingLoader(
        embeddings_file=config["embeddings_file"],
        metadata_files=[config["metadata_file"], config["testset_file"]],
        clip_len=config["clip_len"],
    )
    categories = LEVEL_CLASSES[config["category_level"]]

    for ho_fold in range(5):
        fold_exp_folder = f"{exp_folder}/fold_{ho_fold}"
        if not os.path.exists(fold_exp_folder):
            os.makedirs(fold_exp_folder)

        train_folds = [i for i in range(5) if i != ho_fold]
        val_fold = [ho_fold]

        train_dataset = get_ppi_dataset(
            config["metadata_file"],
            embedding_loader.keys2idx,
            config["ppi_database"],
            config["category_level"],
            train_folds,
            include_links=True,
        )

        train_loader = NeighborLoader(
            train_dataset,
            num_neighbors=config["num_neighbors"],
            batch_size=config["batch_size"],
            input_nodes=None,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            drop_last=True,
        )

        val_dataset = get_ppi_dataset(
            config["metadata_file"],
            embedding_loader.keys2idx,
            config["ppi_database"],
            config["category_level"],
            val_fold,
            include_links=True if config["ppi_test"] else False,
        )

        val_loader = NeighborLoader(
            val_dataset,
            num_neighbors=[64 for i in range(len(config["num_neighbors"]))],
            batch_size=1,
            input_nodes=None,
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
        )

        test_dataset = get_ppi_dataset(
            config["testset_file"],
            embedding_loader.keys2idx,
            config["ppi_database"],
            config["category_level"],
            None,
            include_links=True if config["ppi_test"] else False,
        )
        test_loader = NeighborLoader(
            test_dataset,
            num_neighbors=[64 for i in range(len(config["num_neighbors"]))],
            batch_size=1,
            input_nodes=None,
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
        )

        if config["loss"] == "BCEWithLogitsLoss":
            criterion = nn.BCEWithLogitsLoss()
        elif config["loss"] == "SigmoidFocalLoss":
            criterion = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

        mlp_config = {
            "input_dim": config["embedding_dim"],
            "num_classes": train_dataset.y.shape[-1],
            "hidden_dim": config["mlp_hidden_dim"],
            "num_hidden_layers": config["mlp_num_hidden_layers"],
            "dropout": config["mlp_dropout"],
        }

        model = GraphFeat2LocModel(
            embedding_loader=embedding_loader,
            model_name=config["agg_method"],
            graph_model_name=config["graph_layer"],
            n_graph_layers=config["n_graph_layers"],
            graph_dropout=config["graph_dropout"],
            clip_len=config["clip_len"],
            loss=criterion,
            mlp_config=mlp_config,
            batches_per_epoch=len(train_loader),
            fold_idx=ho_fold + 1,
            optimizer=config["optimizer"],
            init_lr=config["init_lr"],
            max_epochs=config["max_epochs"],
        )
        model_ckpt_cb = ModelCheckpoint(
            dirpath=f"{fold_exp_folder}/models/",
            filename="best_model_acc",
            monitor=f"valid/fold_{ho_fold + 1}_macro_ap",
            verbose=True,
            save_last=True,
            save_top_k=1,
            mode="max",
            enable_version_counter=False,
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stopping = EarlyStopping(
            monitor=f"valid/fold_{ho_fold + 1}_macro_ap",
            patience=10,
            mode="max",
        )
        feat_cb = FeatureCollectionCallback()

        trainer = L.Trainer(
            default_root_dir=exp_folder,
            accelerator="gpu",
            num_nodes=1,
            devices="auto",
            check_val_every_n_epoch=config["valid_every"],
            max_epochs=model.max_epochs,
            logger=wandb_logger,
            log_every_n_steps=10,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            callbacks=[model_ckpt_cb, lr_monitor, early_stopping, feat_cb],
            num_sanity_val_steps=0,
        )
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        fo_metrics = trainer.test(model, dataloaders=val_loader, ckpt_path="best")
        fo_metrics = {
            k.replace(f"valid/fold_{ho_fold + 1}_", ""): v
            for fo_metric in fo_metrics
            for k, v in fo_metric.items()
        }
        all_valid_metrics.loc[f"fold_{ho_fold + 1}"] = pd.Series(fo_metrics)

        feat_dict = model.features
        val_pred_df = pd.DataFrame({"id": feat_dict["ids"], "seq": feat_dict["seqs"]})
        for i, cat in enumerate(categories):
            val_pred_df[f"{cat}_true"] = feat_dict["targets"][:, i]
            val_pred_df[f"{cat}_pred"] = feat_dict["logits"][:, i]
        val_pred_df.to_csv(
            f"{fold_exp_folder}/fold_{ho_fold}_val_predictions.csv", index=False
        )
        torch.save(
            feat_dict["attentions"],
            f"{fold_exp_folder}/fold_{ho_fold}_val_attention.pt",
        )

        test_metrics = trainer.test(model, dataloaders=test_loader, ckpt_path="best")
        test_metrics = {
            k.replace(f"valid/fold_{ho_fold + 1}_", ""): v
            for test_metric in test_metrics
            for k, v in test_metric.items()
        }
        all_test_metrics.loc[f"fold_{ho_fold + 1}"] = pd.Series(test_metrics)

        feat_dict = model.features
        test_pred_df = pd.DataFrame({"id": feat_dict["ids"], "seq": feat_dict["seqs"]})
        for i, cat in enumerate(categories):
            test_pred_df[f"{cat}_true"] = feat_dict["targets"][:, i]
            test_pred_df[f"{cat}_pred"] = feat_dict["logits"][:, i]
        test_pred_df.to_csv(
            f"{fold_exp_folder}/fold_{ho_fold}_test_predictions.csv", index=False
        )
        torch.save(
            feat_dict["attentions"],
            f"{fold_exp_folder}/fold_{ho_fold}_test_attention.pt",
        )

    all_valid_metrics.to_csv(f"{exp_folder}/overall_valid_metrics.csv", index=True)
    all_test_metrics.to_csv(f"{exp_folder}/overall_test_metrics.csv", index=True)

    wandb_logger.log_metrics(
        {"overall_valid/fo_metrics": wandb.Table(dataframe=all_valid_metrics)}
    )
    wandb_logger.log_metrics(
        {"overall_test/fo_metrics": wandb.Table(dataframe=all_test_metrics)}
    )

    for metric in metrics:
        wandb_logger.log_metrics(
            {f"overall_valid/{metric}": all_valid_metrics[metric].mean()}
        )
        wandb_logger.log_metrics(
            {f"overall_test/{metric}": all_test_metrics[metric].mean()}
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="")

    argparser.add_argument(
        "-c", "--config", help="yaml file sweep params", default=None, type=str
    )
    argparser.add_argument(
        "-r", "--random_seed", help="random_seed", default=42, type=int
    )

    args = argparser.parse_args()  # ["-c", "configs/graph_configs/config_prott5.yaml"])

    # Set seed
    set_random_seed(args.random_seed)

    config_path = args.config
    config = load_config(config_path)
    main(config)
