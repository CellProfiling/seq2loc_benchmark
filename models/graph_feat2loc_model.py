from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lightning as L
import torch
from torch import nn, optim
from torchmetrics import MetricCollection, classification

from models import aggregation as agg
from models.mlp import MLPClassifier
from data.ppi_dataset import EmbeddingLoader
import torch_geometric.nn as graph_models


def get_agg_model(model_name: str, input_dim: int, clip_len: int) -> nn.Module:
    if model_name == "MaxPool":
        model = agg.MaxPool()
    elif model_name == "MeanPool":
        model = agg.MeanPool()
    elif model_name == "LightAttentionPool":
        model = agg.LightAttentionPool(input_dim=input_dim)
    elif model_name == "MultiHeadAttentionPool":
        model = agg.MultiHeadAttentionPool(input_dim=input_dim)
    elif model_name == "TransformerPool":
        model = agg.TransformerPool(input_dim=input_dim, max_len=clip_len)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


def get_graph_model(
    model_name: str, input_dim: int, num_layers: int, dropout: float
) -> nn.Module:
    if model_name == "graphsage":
        model = graph_models.GraphSAGE(
            in_channels=input_dim,
            hidden_channels=input_dim,
            out_channels=input_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_name == "gcn":
        model = graph_models.GCN(
            in_channels=input_dim,
            hidden_channels=input_dim,
            out_channels=input_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


class GraphFeat2LocModel(L.LightningModule):
    def __init__(
        self,
        embedding_loader: EmbeddingLoader,
        model_name: str,
        graph_model_name: str,
        n_graph_layers: int,
        graph_dropout: float,
        clip_len: int,
        loss: nn.Module,
        mlp_config: Dict[str, Any],
        batches_per_epoch: int,
        fold_idx: int,
        optimizer: str,
        init_lr: float = 1.0e-4,
        max_epochs: int = 100,
    ):
        super().__init__()

        self.embedding_loader = embedding_loader

        self.model = get_agg_model(
            model_name=model_name, input_dim=mlp_config["input_dim"], clip_len=clip_len
        )
        if model_name == "LightAttentionPool":
            mlp_config["input_dim"] *= 2

        self.graph_model = get_graph_model(
            model_name=graph_model_name,
            input_dim=mlp_config["input_dim"],
            num_layers=n_graph_layers,
            dropout=graph_dropout,
        )

        mlp_config["input_dim"] = self.graph_model.out_channels
        self.mlp_classifier = MLPClassifier(**mlp_config)

        num_categories = mlp_config["num_classes"]

        self.loss = loss

        self.batches_per_epoch = batches_per_epoch
        self.fold_idx = fold_idx

        self.optimizer_name = optimizer
        self.lr = init_lr
        self.weight_decay = 0.05
        self.betas = [0.9, 0.95]
        self.max_epochs = max_epochs

        self.train_metrics = MetricCollection(
            {
                "accuracy": classification.MultilabelAccuracy(
                    num_labels=num_categories
                ),
                "f1_score": classification.MultilabelF1Score(num_labels=num_categories),
                "macro_ap": classification.MultilabelAveragePrecision(
                    num_labels=num_categories, average="macro"
                ),
                "micro_ap": classification.MultilabelAveragePrecision(
                    num_labels=num_categories, average="micro"
                ),
                "coverage_error": classification.MultilabelCoverageError(
                    num_labels=num_categories
                ),
                "mlrap": classification.MultilabelRankingAveragePrecision(
                    num_labels=num_categories
                ),
            },
            prefix=f"train/fold_{fold_idx}_",
        )
        self.valid_metrics = self.train_metrics.clone(prefix=f"valid/fold_{fold_idx}_")

    def configure_optimizers(self):
        params = (
            list(self.model.parameters())
            + list(self.graph_model.parameters())
            + list(self.mlp_classifier.parameters())
        )
        if self.optimizer_name == "AdamW":
            optimizer = optim.AdamW(
                params,
                lr=self.lr,
            )
        elif self.optimizer_name == "Adam":
            optimizer = optim.Adam(
                params,
                lr=self.lr,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=2, verbose=True
                ),
                "monitor": f"valid/fold_{self.fold_idx}_macro_ap",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.training:
            prot_idxs = batch.x
            embeddings, masks = self.embedding_loader.load_batch_embedding_and_mask(
                prot_idxs
            )
            return embeddings, masks, batch.edge_index, batch.y
        else:
            prot_idxs = batch.x
            prot_ids, seqs, embeddings, masks = (
                self.embedding_loader.load_batch_embedding_and_mask(
                    prot_idxs, return_metadata=True
                )
            )
            return prot_ids, seqs, embeddings, masks, batch.edge_index, batch.y

    def forward(self, embeddings, masks, edge_index, return_attn=False):
        pooled_embedding, attention = self.model(embeddings, masks)

        graph_embedding = self.graph_model(pooled_embedding, edge_index)

        logits = self.mlp_classifier(graph_embedding)

        if return_attn:
            return logits, attention
        else:
            return logits

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        embeddings, masks, edge_index, locations = batch

        logits = self.forward(embeddings, masks, edge_index)

        loss = self.loss(logits, locations)
        self.log(f"train/fold_{self.fold_idx}_loss", loss)

        batch_metrics = self.train_metrics(torch.sigmoid(logits), locations.long())
        self.log_dict(batch_metrics, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def on_validation_epoch_start(self):
        self.valid_metrics.reset()

    def validation_step(self, batch, batch_idx):
        ids, seqs, embeddings, masks, edge_index, locations = batch

        logits, attention = self.forward(
            embeddings, masks, edge_index, return_attn=True
        )

        loss = self.loss(logits, locations)
        self.log(f"valid/fold_{self.fold_idx}_loss", loss)

        self.valid_metrics.update(torch.sigmoid(logits), locations.long())
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute())
        self.valid_metrics.reset()

    def on_test_epoch_start(self):
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        ids, seqs, embeddings, masks, edge_index, locations = batch

        logits, attention = self.forward(
            embeddings, masks, edge_index, return_attn=True
        )

        loss = self.loss(logits, locations)

        self.valid_metrics.update(logits, locations.long())
        return {
            "loss": loss,
            "ids": ids,
            "seqs": seqs,
            "logits": logits,
            "locations": locations,
            "attention": attention,
        }

    def on_test_epoch_end(self):
        self.log_dict(self.valid_metrics.compute())
        self.valid_metrics.reset()
