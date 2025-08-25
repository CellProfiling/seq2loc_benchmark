import argparse
import pandas as pd
import numpy as np
import joblib

# For Ensembl/ENSG IDs: use gget.seq
def ensembl_ids_to_seqs(gene_csv, id_col="ensembl_ids"):
    import gget
    gene_list = gene_csv[id_col].to_list()
    uniprot_ids = []
    seqs = []
    for gene_id in gene_list:
        try:
            output = gget.seq(gene_id, translate=True, isoforms=False)
            uniprot_id = output[0].split()[2]
            seq = output[1]
        except Exception as e:
            print(e)
            uniprot_id = pd.NA
            seq = pd.NA
        uniprot_ids.append(uniprot_id)
        seqs.append(seq)
    return uniprot_ids, seqs

# For UniProt IDs: use requests + biopython
def uni_id_to_seq(uni_id):
    import requests as r
    from Bio import SeqIO
    from io import StringIO
    baseUrl = "http://www.uniprot.org/uniprot/"
    currentUrl = baseUrl + uni_id + ".fasta"
    response = r.post(currentUrl)
    cData = ''.join(response.text)
    Seq = StringIO(cData)
    pSeq = list(SeqIO.parse(Seq, 'fasta'))
    try:
        seq = str(pSeq[0].seq)
    except Exception as e:
        print(f"Error retrieving sequence for {uni_id}: {e}")
        seq = pd.NA
    return seq

def get_uniprot_seqs(gene_csv, id_col="Uniprot"):
    gene_list = gene_csv[id_col].dropna().unique().tolist()
    seqs = [uni_id_to_seq(uni_id) for uni_id in gene_list]
    # Map back to original DataFrame indices
    seq_dict = dict(zip(gene_list, seqs))
    mapped_seqs = gene_csv[id_col].map(seq_dict)
    return gene_list, mapped_seqs.tolist()

def choose_seq_func(id_col):
    id_col_lower = id_col.lower()
    if ("ensembl" in id_col_lower) or ("ensg" in id_col_lower):
        return ensembl_ids_to_seqs, "ensembl"
    elif "uniprot" in id_col_lower:
        return get_uniprot_seqs, "uniprot"
    else:
        print(f"Warning: id_col \"{id_col}\" not recognized. Defaulting to ensembl_ids_to_seqs.")
        return get_canonical_seq, "ensembl"

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Add uniprot canonical sequences (and uniprot ids if needed) to a metadata CSV based on ensmble-gene/uniprot-protein IDs.")
    argparser.add_argument("--csv", required=True, help="Path to csv with gene/protein IDs")
    argparser.add_argument("--n_cores", type=int, default=1, help="Number of CPUs for parallel processing")
    argparser.add_argument("--id_col", default="ensembl_ids", help="Column in CSV to use as gene/protein ID (e.g. ensembl_ids, uniprot_id, ensg_id, Uniprot)")
    args = argparser.parse_args()
    csv = args.csv
    n_cores = args.n_cores
    id_col = args.id_col

    df = pd.read_csv(csv)

    # Add columns if missing
    if "Sequence" not in df.columns:
        df["Sequence"] = pd.NA
    if "uniprot_id" not in df.columns and ("ensembl" in id_col.lower() or "ensg" in id_col.lower()):
        df["uniprot_id"] = pd.NA

    seq_func, mode = choose_seq_func(id_col)

    # Only fill in missing sequences
    temp = df[df.Sequence.isna()]
    idx = temp.index.to_list()
    split_dfs = np.array_split(temp, n_cores)

    if mode == "ensembl":
        output = joblib.Parallel(n_jobs=n_cores)(
            joblib.delayed(seq_func)(split_df, id_col=id_col) for split_df in split_dfs
        )
        uniprot_ids = [o[0] for o in output]
        seqs = [o[1] for o in output]
        uniprot_ids = sum(uniprot_ids, [])
        seqs = sum(seqs, [])
        df.loc[idx, "uniprot_id"] = uniprot_ids
        df.loc[idx, "Sequence"] = seqs
    elif mode == "uniprot":
        # For UniProt, get_uniprot_seqs returns all unique IDs and a mapped sequence list
        output = joblib.Parallel(n_jobs=n_cores)(
            joblib.delayed(seq_func)(split_df, id_col=id_col) for split_df in split_dfs
        )
        # Merge all mapped sequence lists (second element of each output)
        seqs = sum([o[1] for o in output], [])
        # Fill with mapped sequence data
        df.loc[idx, "Sequence"] = seqs

    df.to_csv(csv, index=False)