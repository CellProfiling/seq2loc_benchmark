import argparse
import gc
import os
import re
import shutil

import h5py
import torch
from Bio import SeqIO
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from huggingface_hub import login
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    EsmModel,
    T5EncoderModel,
    T5Tokenizer,
)


def prottransSeqConvert(sequence):
    # ProtTrans Models don't use some rare amino acids and have space between residues
    # https://huggingface.co/Rostlab/prot_t5_xl_uniref50
    sequence = re.sub(r"[UZOB]", "X", sequence)
    sequence = [" ".join(sequence)]
    return sequence


def get_ESM3_embedding(sequence, model):
    #https://github.com/evolutionaryscale/esm/issues/2
    protein = ESMProtein(sequence=(sequence))
    protein_tensor = model.encode(protein)
    output = model.forward_and_sample(
        protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
    )
    embedding = output.per_residue_embedding[1:-1, :]
    return embedding.cpu()


def get_ESM2_embedding(sequence, model, tokenizer):
    #Example for extracting embedding: https://github.com/evolutionaryscale/esm/issues/2
    ids = tokenizer.batch_encode_plus(
        [sequence], add_special_tokens=False, padding=True
    )
    input_ids = torch.tensor(ids["input_ids"]).to(device)
    embedding = model(input_ids=input_ids)
    embedding = embedding.last_hidden_state[0, :]
    return embedding.cpu()

def get_ESM1_embedding(sequence, model, tokenizer):
    #https://huggingface.co/docs/transformers/en/model_doc/esm
    ids = tokenizer.batch_encode_plus(
        [sequence], add_special_tokens=False, padding=True
    )
    input_ids = torch.tensor(ids["input_ids"]).to(device)
    embedding = model(input_ids=input_ids)
    embedding = embedding.last_hidden_state[0, :]
    return embedding.cpu()


def get_ProtT5_embedding(sequence, model, tokenizer):
    #https://huggingface.co/docs/transformers/en/model_doc/t5
    #https://www.kaggle.com/code/tilii7/prott5-xl-embedding
    sequence = prottransSeqConvert(sequence)
    ids = tokenizer.batch_encode_plus(sequence, add_special_tokens=False, padding=True)
    input_ids = torch.tensor(ids["input_ids"]).to(device)
    embedding = model(input_ids=input_ids)
    embedding = embedding.last_hidden_state[0, :]
    return embedding.cpu()


def get_ProtBert_embedding(sequence, model, tokenizer):
    #https://huggingface.co/Rostlab/prot_bert
    sequence = prottransSeqConvert(sequence)
    input_ids = tokenizer(
        sequence, return_tensors="pt", add_special_tokens=False, padding=True).to(device)
    embedding = model(**input_ids)
    embedding = embedding.last_hidden_state[0, :]
    return embedding.cpu()


def get_embedding(sequence, model, tokenizer=None, modelname="ESM3", clip_len=4000):
    if len(sequence) > clip_len:
        sequence = sequence[:clip_len//2] + sequence[-clip_len//2:]
    if modelname == "ESM3":
        return get_ESM3_embedding(sequence, model)
    elif modelname == "ESM2":
        return get_ESM2_embedding(sequence, model, tokenizer)
    elif modelname == "ESM1":
        if len(sequence) > 1022: #input length is limited for ESM1b
         sequence = sequence[:511] + sequence[-511:]
        return get_ESM1_embedding(sequence, model, tokenizer)
    elif modelname == "ProtT5":
        return get_ProtT5_embedding(sequence, model, tokenizer)
    elif modelname == "ProtBert":
        return get_ProtBert_embedding(sequence, model, tokenizer)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="")

    argparser.add_argument(
        "-f", "--fasta", help="path to fasta with transcript ids and sequences"
    )
    argparser.add_argument("-m", "--model", help="embedding model")
    argparser.add_argument("-p", "--model_path", help="path to model")
    argparser.add_argument("-o", "--output_path", help="path to output ")
    argparser.add_argument("-t", "--token", help="token for huggingface ESM3")
    argparser.add_argument("-c", "--clip_len", type=int, default=4000, help="token for huggingface ESM3")

    args = argparser.parse_args()

    fastafile = args.fasta
    modelname = args.model
    model_path = args.model_path + "/" + modelname
    output_path = args.output_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if "." not in output_path:#output_path is directory
        print("output_path is directory")
        output_path = output_path + "/" + modelname
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_file = output_path + "/embeddings.h5"
    else:#output_path is file
        output_file = output_path
    #shutil.rmtree(output_file, ignore_errors=True)
    token = args.token
    clip_len = args.clip_len

    login(token)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # LOAD MODEL
    if modelname == "ESM3":
        tokenizer = None
        model = ESM3.from_pretrained(ESM3_OPEN_SMALL)

    elif modelname == "ESM2":
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/esm2_t36_3B_UR50D", cache_dir=model_path
        )
        model = EsmModel.from_pretrained(
            "facebook/esm2_t36_3B_UR50D", cache_dir=model_path
        )
    elif modelname == "ESM1":
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/esm1b_t33_650M_UR50S", cache_dir=model_path
        )
        model = EsmModel.from_pretrained(
            "facebook/esm1b_t33_650M_UR50S", cache_dir=model_path
        )
    elif modelname == "ProtT5":
        tokenizer = T5Tokenizer.from_pretrained(
            "Rostlab/prot_t5_xl_uniref50", do_lower_case=False, cache_dir=model_path
        )
        model = T5EncoderModel.from_pretrained(
            "Rostlab/prot_t5_xl_uniref50", cache_dir=model_path
        )
        gc.collect()
    elif modelname == "ProtBert":
        tokenizer = BertTokenizer.from_pretrained(
            "Rostlab/prot_bert", do_lower_case=False, cache_dir=model_path
        )
        model = BertModel.from_pretrained("Rostlab/prot_bert", cache_dir=model_path)
    else:
        raise Exception("Improper Model Name")

    model = model.to(device)
    model = model.eval()
    
    with torch.no_grad():
        with h5py.File(output_file, "w") as hf:
            total_seqs = len(list(SeqIO.parse(fastafile, "fasta")))
            for seq_record in tqdm(SeqIO.parse(fastafile, "fasta"), total=total_seqs):
                gene_id = str(seq_record.description)
                gene_id = gene_id.strip()
                isoform = str(seq_record.seq)
                embedding = get_embedding(
                    isoform, model, tokenizer, modelname=modelname, clip_len=clip_len
                )
                hf.create_dataset(gene_id, data=embedding)
