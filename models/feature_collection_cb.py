from typing import Any, Mapping, Union
import torch
from torch import Tensor
import lightning as L
from lightning.pytorch.callbacks import Callback


class FeatureCollectionCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        pl_module.features = None

        self.ids = []
        self.seqs = []
        self.attentions = []
        self.logits = []
        self.targets = []

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Union[Tensor, Mapping[str, Any], None],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        self.ids.extend(outputs["ids"])
        self.seqs.extend(outputs["seqs"])
        self.attentions.append(outputs["attention"].detach().cpu())
        self.logits.append(outputs["logits"].detach().cpu())
        self.targets.append(outputs["locations"].detach().cpu())

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        pl_module.features = {
            "ids": self.ids,
            "seqs": self.seqs,
            "attentions": (
                torch.cat(self.attentions, dim=0) if self.attentions else None
            ),
            "logits": torch.cat(self.logits, dim=0) if self.logits else None,
            "targets": torch.cat(self.targets, dim=0) if self.targets else None,
        }
