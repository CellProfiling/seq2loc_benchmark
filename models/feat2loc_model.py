from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lightning as L
import torch
from torch import nn, optim
from torchmetrics import MetricCollection, classification

from models import aggregation as agg
from models.mlp import MLPClassifier


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


class Feat2LocModel(L.LightningModule):
    def __init__(
        self,
        model_name: Dict[str, Any],
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

        self.model = get_agg_model(
            model_name=model_name, input_dim=mlp_config["input_dim"], clip_len=clip_len
        )
        num_categories = mlp_config["num_classes"]

        if model_name == "LightAttentionPool":
            mlp_config["input_dim"] *= 2
        self.mlp_classifier = MLPClassifier(**mlp_config)

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
        params = list(self.model.parameters()) + list(self.mlp_classifier.parameters())
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
                "frequency": 2,
            },
        }

    def forward(self, embeddings, masks, return_attn=False):
        pooled_embedding, attention = self.model(embeddings, masks)
        logits = self.mlp_classifier(pooled_embedding)

        if return_attn:
            return logits, attention
        else:
            return logits

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        embeddings, masks, locations = batch

        logits = self.forward(embeddings, masks)

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
        ids, seqs, embeddings, masks, locations = batch

        logits, attention = self.forward(embeddings, masks, return_attn=True)

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
        ids, seqs, embeddings, masks, locations = batch

        logits, attention = self.forward(embeddings, masks, return_attn=True)

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
