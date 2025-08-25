import argparse
import os
import random

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
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd

import wandb
from data.collate_fn import collate_fn, test_collate_fn
from data.dataset import EmbeddingDataset
from models.feat2loc_model import Feat2LocModel
from models.focal_loss import SigmoidFocalLoss
from models.feature_collection_cb import FeatureCollectionCallback


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


def main(sweep_config=None, sweep_id=None):
    api = wandb.Api()
    entity = api.default_entity
    runs = api.runs(f"{entity}/seq2loc_sweep")
    crashed_runs = [run for run in runs if run.State in ["crashed", "failed"]]
    print(f"found {len(crashed_runs)} crashed runs. deleting....")
    for run in crashed_runs:
        run.delete()

    if sweep_id is None:
        sweep_id = wandb.sweep(sweep_config, project="seq2loc_sweep")

    wandb.agent(sweep_id, function=train)


def train(config=None):
    run = wandb.init(config=config)
    config = wandb.config

    exp_folder = (
        config["exp_folder"]
        + "/"
        + config["exp_name"]
        + "_"
        + os.path.basename(config["metadata_file"]).split(".")[0]
    )

    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    exp_folder = f"{exp_folder}/{run.id}"
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    yaml.dump(config, open(f"{exp_folder}/config.yaml", "w"), default_flow_style=False)

    wandb_logger = WandbLogger(
        project="seq2loc_sweep",
        name=config["exp_name"],
        config=config,
        dir=exp_folder,
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

    for ho_fold in range(5):
        fold_exp_folder = f"{exp_folder}/fold_{ho_fold}"
        if not os.path.exists(fold_exp_folder):
            os.makedirs(fold_exp_folder)

        train_folds = [i for i in range(5) if i != ho_fold]
        val_fold = [ho_fold]

        train_dataset = EmbeddingDataset(
            config["embeddings_file"],
            config["metadata_file"],
            config["category_level"],
            train_folds,
            clip_len=config["clip_len"],
            random_clip=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

        val_dataset = EmbeddingDataset(
            config["embeddings_file"],
            config["metadata_file"],
            config["category_level"],
            val_fold,
            clip_len=config["clip_len"],
            random_clip=False,
            test_mode=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False,
            collate_fn=test_collate_fn,
        )

        test_dataset = EmbeddingDataset(
            config["embeddings_file"],
            config["testset_file"],
            config["category_level"],
            None,
            clip_len=config["clip_len"],
            random_clip=False,
            test_mode=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False,
            collate_fn=test_collate_fn,
        )

        if config["loss"] == "BCEWithLogitsLoss":
            criterion = nn.BCEWithLogitsLoss()
        elif config["loss"] == "SigmoidFocalLoss":
            criterion = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

        mlp_config = {
            "input_dim": config["embedding_dim"],
            "num_classes": train_dataset.n_categories,
            "hidden_dim": config["mlp_hidden_dim"],
            "num_hidden_layers": config["mlp_num_hidden_layers"],
            "dropout": config["mlp_dropout"],
        }

        model = Feat2LocModel(
            model_name=config["agg_method"],
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
        for i, cat in enumerate(val_dataset.categories):
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
        for i, cat in enumerate(test_dataset.categories):
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

    run.log({"overall_valid/fo_metrics": wandb.Table(dataframe=all_valid_metrics)})
    run.log({"overall_test/fo_metrics": wandb.Table(dataframe=all_test_metrics)})

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
        "-c", "--sweep_config", help="yaml file sweep params", default=None, type=str
    )
    argparser.add_argument(
        "-sw_id",
        "--sweep_id",
        help="sweep id to run",
        default=None,
        type=str,
        nargs="?",
    )
    argparser.add_argument(
        "-r", "--random_seed", help="random_seed", default=42, type=int
    )
    args = argparser.parse_args()

    # Set seed
    set_random_seed(args.random_seed)

    sweep_config_path = args.sweep_config
    sweep_id = args.sweep_id

    if sweep_config_path is None and sweep_id is None:
        raise ValueError("Either sweep_config or sweep_id must be provided")

    if sweep_config_path is not None:
        sweep_config = load_config(sweep_config_path)
    else:
        sweep_config = None

    main(sweep_config, sweep_id)
