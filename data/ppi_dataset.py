import copy
import random

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Data, Dataset, InMemoryDataset, OnDiskDataset
from torch_geometric.utils import add_self_loops
from tqdm import tqdm

LEVEL_CLASSES = yaml.safe_load(open("metadata/level_classes.yaml"))


def load_and_map_string_ppi(threshold=800):
    ppi_data = pd.read_csv("ppi_data/9606.protein.links.v12.0.onlyAB.csv")
    ppi_data["protein1"] = ppi_data["protein1"].apply(lambda x: x.replace("9606.", ""))
    ppi_data["protein2"] = ppi_data["protein2"].apply(lambda x: x.replace("9606.", ""))

    ensp2uniprot = pd.read_csv("ppi_data/ensp2uniprot.tsv", sep="\t")
    ensp2uniprot = ensp2uniprot[ensp2uniprot["Reviewed"] == "reviewed"].reset_index(
        drop=True
    )
    ensp2uniprot = ensp2uniprot.groupby("From")["Entry"].apply(list).reset_index()
    ppi_data["protein1"] = ppi_data["protein1"].map(
        ensp2uniprot.set_index("From")["Entry"]
    )
    ppi_data["protein2"] = ppi_data["protein2"].map(
        ensp2uniprot.set_index("From")["Entry"]
    )
    ppi_data = ppi_data.dropna().reset_index(drop=True)
    ppi_data = ppi_data.explode("protein1").reset_index(drop=True)
    ppi_data = ppi_data.explode("protein2").reset_index(drop=True)
    ppi_data = ppi_data[ppi_data["combined_score"] > threshold].reset_index(drop=True)
    ppi_data = ppi_data[["protein1", "protein2"]]
    ppi_data.to_csv("ppi_data/string_ppi_uniprot.csv", index=False)
    return ppi_data


def load_and_map_bioplex_ppi():
    ppi_data = pd.read_csv("ppi_data/BioPlex_293T_Network_10K_Dec_2019.tsv", sep="\t")
    ppi_data = ppi_data.rename(columns={"UniprotA": "protein1", "UniprotB": "protein2"})
    ppi_data = ppi_data[["protein1", "protein2"]]
    ppi_data["protein1"] = ppi_data["protein1"].apply(lambda x: x.split("-")[0])
    ppi_data["protein2"] = ppi_data["protein2"].apply(lambda x: x.split("-")[0])
    ppi_data = ppi_data[ppi_data["protein1"] != "UNKNOWN"]
    ppi_data = ppi_data[ppi_data["protein2"] != "UNKNOWN"]
    ppi_data.to_csv("ppi_data/bioplex_ppi_uniprot.csv", index=False)
    return ppi_data


STRING_PPI_DATA = pd.read_csv("ppi_data/string_ppi_uniprot.csv")
BIO_PPI_DATA = pd.read_csv("ppi_data/bioplex_ppi_uniprot.csv")


class EmbeddingLoader:
    def __init__(self, embeddings_file, metadata_files, clip_len=1024):
        self.embeddings = h5py.File(embeddings_file, "r")

        keys = list(self.embeddings.keys())
        self.keys2idx = {key: idx for idx, key in enumerate(keys)}
        self.idx2keys = {idx: key for idx, key in enumerate(keys)}

        self.embedding_size = np.array(self.embeddings[keys[0]]).shape[1]

        if type(metadata_files) == str:
            metadata_files = [metadata_files]
        metadata = []
        for metadata_file in metadata_files:
            metadata.append(pd.read_csv(metadata_file))
        metadata = pd.concat(metadata).reset_index(drop=True)

        self.key2seq = {
            row["uniprot_id"]: row["sequence"] for _, row in metadata.iterrows()
        }

        self.clip_len = clip_len

    def get_clipped_embedding(self, isoform_id):
        embedding_np = np.array(self.embeddings.get(isoform_id))

        seq = self.key2seq[isoform_id]

        prot_len, embedding_dim = embedding_np.shape

        if prot_len < self.clip_len:
            embedding = torch.zeros((self.clip_len, embedding_dim), dtype=torch.float32)
            embedding[:prot_len] = torch.from_numpy(embedding_np)

            mask = torch.zeros(self.clip_len, dtype=torch.float32)
            mask[:prot_len] = 1.0
        else:
            indices = np.concatenate(
                [
                    np.arange(self.clip_len // 2),
                    np.arange(prot_len - (self.clip_len // 2), prot_len),
                ]
            )

            embedding = torch.tensor(embedding_np[indices], dtype=torch.float32)
            mask = torch.ones(self.clip_len, dtype=torch.float32)
            seq = "".join([seq[i] for i in indices])

        return embedding, mask, seq

    def load_batch_embedding_and_mask(self, isoform_idxs, return_metadata=False):
        batch_size = len(isoform_idxs)

        embeddings = torch.zeros(
            (batch_size, self.clip_len, self.embedding_size),
            dtype=torch.float32,
            device=isoform_idxs.device,
        )
        masks = torch.zeros(
            (batch_size, self.clip_len), dtype=torch.float32, device=isoform_idxs.device
        )

        isoform_ids = []
        seqs = []
        for i, isoform_idx in enumerate(isoform_idxs):
            isoform_id = self.idx2keys[isoform_idx.item()]
            isoform_ids.append(isoform_id)
            embedding, mask, seq = self.get_clipped_embedding(isoform_id)
            seqs.append(seq)
            embeddings[i] = embedding
            masks[i] = mask

        if return_metadata:
            return isoform_ids, seqs, embeddings, masks
        else:
            return embeddings, masks


def get_ppi_dataset(
    metadata_file,
    isoforms2idx,
    ppi,
    category_level,
    folds=None,
    include_links=False,
):
    metadata = pd.read_csv(metadata_file)
    if folds is not None:
        metadata = metadata[metadata["fold"].isin(folds)].reset_index(drop=True)

    metadata["embedding_idx"] = metadata["uniprot_id"].apply(lambda x: isoforms2idx[x])
    x = torch.tensor(metadata["embedding_idx"].values, dtype=torch.long)

    categories = LEVEL_CLASSES[category_level]
    all_locs = list(metadata[category_level].astype(str).str.split(";"))
    labels_onehot = np.array(
        [[1 if cat in x else 0 for cat in categories] for x in all_locs]
    )
    assert np.all(np.array([len(x) for x in all_locs]) == labels_onehot.sum(axis=1))
    y = torch.tensor(labels_onehot, dtype=torch.float32)

    graph_node2idx = {
        isoform: idx for idx, isoform in enumerate(metadata["uniprot_id"])
    }
    if not include_links:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        ppi_data = STRING_PPI_DATA if ppi == "string" else BIO_PPI_DATA
        ppi_data = ppi_data[
            ppi_data["protein1"].isin(metadata["uniprot_id"])
            & ppi_data["protein2"].isin(metadata["uniprot_id"])
        ].reset_index(drop=True)
        edge_index = np.array(
            [
                ppi_data["protein1"].map(lambda x: graph_node2idx[x]).values,
                ppi_data["protein2"].map(lambda x: graph_node2idx[x]).values,
            ]
        )
        edge_index = torch.from_numpy(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(x))

    data = Data(x=x, edge_index=edge_index, y=y)
    data.validate()
    return data


def get_ppi_test_dataset(
    data_df,
    link_df,
    ppi,
    category_level,
    isoforms2idx,
):
    if link_df is None or data_df.equals(link_df):
        comb_data_df = copy.deepcopy(data_df)
        print("Link dataframe is equal to data dataframe. No need to process links.")
    else:
        comb_data_df = pd.concat([data_df, link_df]).reset_index(drop=True)

    comb_data_df["embedding_idx"] = comb_data_df["uniprot_id"].apply(
        lambda x: isoforms2idx[x]
    )
    x = torch.tensor(comb_data_df["embedding_idx"].values, dtype=torch.long)

    categories = LEVEL_CLASSES[category_level]
    all_locs = list(comb_data_df[category_level].astype(str).str.split(";"))
    labels_onehot = np.array(
        [[1 if cat in x else 0 for cat in categories] for x in all_locs]
    )
    assert np.all(np.array([len(x) for x in all_locs]) == labels_onehot.sum(axis=1))
    y = torch.tensor(labels_onehot, dtype=torch.float32)

    graph_node2idx = {
        isoform: idx for idx, isoform in enumerate(comb_data_df["uniprot_id"])
    }

    if link_df is not None:
        ppi_data = STRING_PPI_DATA if ppi == "string" else BIO_PPI_DATA
        ppi_data = ppi_data[
            ppi_data["protein1"].isin(comb_data_df["uniprot_id"])
            & ppi_data["protein2"].isin(comb_data_df["uniprot_id"])
        ].reset_index(drop=True)
        edge_index = np.array(
            [
                ppi_data["protein1"].map(lambda x: graph_node2idx[x]).values,
                ppi_data["protein2"].map(lambda x: graph_node2idx[x]).values,
            ]
        )
        edge_index = torch.from_numpy(edge_index)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    edge_index, _ = add_self_loops(edge_index, num_nodes=len(x))

    data = Data(x=x, edge_index=edge_index, y=y)
    data.validate()
    print("Test data:", data)
    return data
