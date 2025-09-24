import random
import os
import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

# Get the directory where this file is located
base_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(base_dir, "..", "datasets", "final", "hierarchical_label_set.yaml")

with open(yaml_path) as f:
    LEVEL_CLASSES = yaml.safe_load(f)

#LEVEL_CLASSES = yaml.safe_load(open("metadata/level_classes.yaml"))


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        embeddings_file,
        metadata_file,
        category_level,
        folds=None,
        clip_len=1024,
        random_clip=False,
        test_mode=False,
        id_col="uniprot_id",
    ):
        """
        Args:
            classes: List of location classes, predefined order should be maintined in yaml file
            embeddings_file: h5 file with embedding from protein-LM
            metadata_file: csv with gene_id and location annotations
            loc_col: column in metadata_file with location annotations
            folds: Subset of [0,1,2,3,4] to run on
            clip_len: Max size of sequence after clipping
        """
        super().__init__()

        self.embeddings = h5py.File(embeddings_file, "r")

        uniprot_ids = [key for key in self.embeddings.keys()]
        metadata = pd.read_csv(metadata_file)
        metadata = metadata[metadata[id_col].isin(uniprot_ids)].reset_index(drop=True)
        if folds is not None:
            metadata = metadata[metadata["fold"].isin(folds)].reset_index(drop=True)
        self.metadata = metadata

        self.id_col = id_col

        self.categories = LEVEL_CLASSES[category_level]
        self.n_categories = len(self.categories)

        all_locs = list(self.metadata[category_level].astype(str).str.split(";"))
        labels_onehot = np.array(
            [[1 if cat in x else 0 for cat in self.categories] for x in all_locs]
        )
        assert np.all(
            np.array([len(x) for x in all_locs]) == labels_onehot.sum(axis=1)
        ), f"Error in one-hot encoding for {self.metadata.iloc[np.where(np.array([len(x) for x in all_locs]) != labels_onehot.sum(axis=1))[0]]["uniprot_id"].values}"

        self.labels = labels_onehot
        self.metadata[self.categories] = labels_onehot

        self.clip_len = clip_len
        self.random_clip = random_clip
        self.test_mode = test_mode

    def __len__(self):
        return len(self.metadata)

    def get_clipped_embedding(self, row):
        isoform_id = row[self.id_col]
        seq = row.get("sequence")

        embedding = np.array(self.embeddings.get(isoform_id))
        embedding = torch.tensor(embedding, dtype=torch.float32)
        prot_len, embedding_dim = embedding.shape
        if prot_len < self.clip_len:
            embedding = torch.cat(
                (embedding, torch.zeros(self.clip_len - prot_len, embedding_dim))
            )
            mask = torch.cat(
                (torch.ones(prot_len), torch.zeros(self.clip_len - prot_len))
            )
        else:
            if self.random_clip and not self.test_mode:
                if random.random() < 0.5:
                    embedding = embedding[: self.clip_len]
                else:
                    idxs = torch.cat(
                        (
                            torch.arange(self.clip_len // 2),
                            torch.arange(prot_len - (self.clip_len // 2), prot_len),
                        )
                    )
                    embedding = embedding[idxs]
            else:
                idxs = torch.cat(
                    (
                        torch.arange(self.clip_len // 2),
                        torch.arange(prot_len - (self.clip_len // 2), prot_len),
                    )
                )
                embedding = embedding[idxs]
            mask = torch.ones(self.clip_len)

        mask = mask.type(torch.float32)
        return embedding, mask, seq

    def __getitem__(self, index):
        metadata_row = self.metadata.loc[index]

        isoform_id = metadata_row[self.id_col]
        embedding, mask, seq = self.get_clipped_embedding(metadata_row)

        locations = torch.tensor(
            metadata_row[self.categories].astype(float).values, dtype=torch.float32
        )
        if self.test_mode:
            return isoform_id, seq, embedding, mask, locations
        else:
            return embedding, mask, locations
