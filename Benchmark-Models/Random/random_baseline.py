def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.abspath('../..'))
from utils.metrics import *
import yaml
import argparse
import re


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(train_csv, test_csv, yaml_classes, save_dir, k=10000, single=False, multi=False):
    
    os.makedirs(save_dir, exist_ok=True)

    CATEGORIES_YAML = load_config(yaml_classes)
    trainset = pd.read_csv(train_csv)
    hou_testset = pd.read_csv(test_csv)

    implicitly_multi = [
        "actin-filaments",
        "intermediate-filaments",
        "centrosome",
        "microtubules",
        "endosomes",
        "lysosomes",
        "peroxisomes",
        "lipid-droplets"
    ]
    pattern = "|".join(map(re.escape, implicitly_multi))

    single_testset = hou_testset[~hou_testset.level3.str.contains(";")]
    single_testset.loc[single_testset.level2.str.contains(";"), "level2"] = pd.NA
    single_testset.loc[
        (single_testset.level1.str.contains(";")) &
        ~((single_testset.level1.str.contains(pattern, na=False)) & (single_testset['level1'].str.count(";") == 1)), 
        "level1"] = pd.NA

    multi_testset = hou_testset[~hou_testset.uniprot_id.isin(single_testset.uniprot_id)]
    multi_testset.loc[~multi_testset.level2.str.contains(";"), "level2"] = pd.NA
    multi_testset.loc[~multi_testset.level1.str.contains(";"), "level1"] = pd.NA

    assert not (single and multi)
    if single:
        hou_testset = single_testset
        tag="_single"
    elif multi:
        hou_testset = multi_testset
        tag="_multi"
    else:
        tag=""

    avg_dfs = []
    for level in [1,2,3]:
        categories = CATEGORIES_YAML[f"level{level}"]
        hou_testset_level = hou_testset[hou_testset[f"level{level}"].notna()]
        testset_size=hou_testset_level.shape[0]
        train_targets = []
        for locs in trainset[f"level{level}"].str.split(";").to_list():
            train_targets.append([1 if loc in locs else 0 for loc in categories])
        train_targets = np.array(train_targets)
        probs = train_targets.mean(axis=0)


        y_true = []
        for locs in hou_testset_level[f"level{level}"].str.split(";").to_list():
            y_true.append([1 if loc in locs else 0 for loc in categories])
        y_true = np.array(y_true)

        num_classes = len(probs)
        rand = np.random.binomial(1, p=probs, size=(testset_size*k, num_classes)).reshape(k, testset_size, num_classes)


        #Get rid classes that are empty in testset (ie int-fils + plastid)
        idxs = np.where(y_true.sum(axis=0) != 0)[0]
        y_true = y_true[:, idxs]
        rand = rand[:, :, idxs]


        #Random Iterations

        metrics_perclass_ = np.zeros((y_true.shape[1], 7))
        metrics_avg_ = np.zeros((1,12))
        for pred in rand:
            metrics = get_all_metrics(y_true, pred, None, categories, continuous=False)
            metrics_perclass = np.array([
                        metrics["mcc_perclass"], 
                        metrics["acc_perclass"], 
                        metrics["recall_perclass"], 
                        metrics["precision_perclass"], 
                        metrics["f1_perclass"],
                        metrics["jaccard_perclass"],
                        metrics["rocauc_perclass"]
                        ]).T
            metrics_avg = np.array([
                    metrics["macro_ap"],
                    metrics["micro_ap"],
                    metrics["acc"],
                    metrics["f1_macro"],
                    metrics["f1_micro"],
                    metrics["jaccard_macro"],
                    metrics["jaccard_micro"],
                    metrics["rocauc_macro"],
                    metrics["rocauc_micro"],
                    metrics["mlrap"],
                    metrics["coverage_error"],
                    metrics["num_labels"]
                    ]).T
            metrics_perclass_+=metrics_perclass
            metrics_avg_+=metrics_avg
        metrics_perclass_ /= k
        metrics_avg_ /= k

        cols = [
                "mcc_perclass",
                "acc_perclass",
                "recall_perclass",
                "precision_perclass",
                "f1_perclass",
                "jaccard_perclass",
                "rocauc_perclass"
                ]
        metrics_perclass_ = pd.DataFrame(metrics_perclass_, columns=cols)
        metrics_perclass_["labels"] = np.array(categories)[idxs]
        metrics_perclass_ = metrics_perclass_.to_csv(f"{save_dir}/random_perclass_metrics_level{level}{tag}.csv")

        cols = [
                "macro_ap",
                "micro_ap",
                "acc",
                "f1_macro",
                "f1_micro",
                "jaccard_macro",
                "jaccard_micro",
                "rocauc_macro",
                "rocauc_micro",
                "mlrap",
                "cov_error",
                "num_labels"
                ]
        metrics_avg_ = pd.DataFrame(metrics_avg_, columns=cols)
        metrics_avg_["level"] = level
        avg_dfs.append(metrics_avg_)

    avg_df = pd.concat(avg_dfs)
    avg_df.to_csv(f"{save_dir}/random_avg_metrics{tag}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-k","--num_iterations", 
        default=10000,
        type=int
    )

    parser.add_argument(
        "-train","--train_csv", 
        default="/hai/scratch/zwefers/seq2loc/metadata/combined_hpa_uniprot_trainset.csv",
        type=str
    )

    parser.add_argument(
        "-test","--test_csv", 
        default="/hai/scratch/zwefers/seq2loc/metadata/hou_testset.csv",
        type=str
    )

    parser.add_argument(
        "-y","--yaml_classes", 
        default="/hai/scratch/zwefers/seq2loc/metadata/location_levels.yaml",
        type=str
    )

    parser.add_argument(
        "-s","--save_dir", 
        default="./",
        type=str
    )

    parser.add_argument(
        "--single", 
        action='store_true'
    )
    parser.add_argument(
        "--multi", 
        action='store_true'
    )

    args = parser.parse_args()


    k= args.num_iterations
    train_csv = args.train_csv
    test_csv = args.test_csv
    yaml_classes = args.yaml_classes
    save_dir = args.save_dir
    single = args.single
    multi = args.multi

    main(train_csv, test_csv, yaml_classes, save_dir, k=k, single=single, multi=multi)
