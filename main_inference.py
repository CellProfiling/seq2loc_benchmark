import copy
import os
import shutil
import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yaml
from statsmodels.formula.api import ols

PARAMETERS = [
    "exp_name",
    "category_level",
    "metadata_file",
    "clip_len",
    "agg_method",
    "loss",
    "mlp_dropout",
    "graph_layer",
    "graph_dropout",
    "n_graph_layers",
    "num_neighbors",
    "ppi_database",
    "ppi_test",
]
METRICS = [
    "macro_ap",
    "micro_ap",
    "acc",
    "acc_samples",
    "f1_macro",
    "f1_micro",
    "jaccard_macro",
    "jaccard_micro",
    "rocauc_macro",
    "rocauc_micro",
    "mlrap",
    "coverage_error",
    "num_labels",
]

LEVEL_CLASSES = yaml.safe_load(open("metadata/level_classes.yaml"))
plt.rcParams["pdf.fonttype"] = 42

INFERENCE_CONFIG_DICT = {
    "train_ppi": "Training Set",
    "test_ppi": "Test Set",
    "no_ppi": "No PPI",
}


def main(exp_folder):
    all_overall_metrics = []
    for inference_neighbour_config in INFERENCE_CONFIG_DICT.keys():
        overall_metrics_df = pd.read_csv(
            f"{exp_folder}/inference_{inference_neighbour_config}_all_folds_overall_metrics.csv"
        )
        overall_metrics_df["Inference PPI"] = INFERENCE_CONFIG_DICT[
            inference_neighbour_config
        ]
        all_overall_metrics.append(overall_metrics_df)
    all_overall_metrics = pd.concat(all_overall_metrics, ignore_index=True)
    return all_overall_metrics


def generate_metrics_plots(all_metrics_df, best_prott5_df, save_folder):
    metric_save_folder = f"{save_folder}/metrics"
    os.makedirs(metric_save_folder, exist_ok=True)
    sns.set_theme(style="ticks")
    for metric in METRICS:
        g = sns.catplot(
            data=all_metrics_df[
                (all_metrics_df["ppi_test"] == False)
                & (all_metrics_df["category_level"] == "level1")
            ],
            x="num_neighbors",
            y=metric,
            hue="Inference PPI",
            col="exp_name",
            kind="bar",
            height=4,
            aspect=1,
        )
        for ax in g.axes.flat:
            ax.hlines(
                y=best_prott5_df[metric].values[0],
                xmin=ax.get_xlim()[0],
                xmax=ax.get_xlim()[1],
                color="red",
                linestyle="--",
                label=f"Best ProtT5 {best_prott5_df[metric].values[0]:.3f}",
            )

        plt.savefig(f"{metric_save_folder}/{metric}.png", bbox_inches="tight", dpi=300)
        plt.close()


def compare_perclass_comparison(all_metrics_df, all_metrics_graph_df, save_folder):
    best_metrics_df = (
        all_metrics_df[
            (all_sweep_df["metadata_file"] == "hpa_uniprot_combined_trainset")
        ]
        .groupby(["category_level"])
        .apply(lambda x: x.loc[x["macro_ap"].idxmax()], include_groups=False)
        .reset_index(drop=True)
    )
    best_metric_graph_df = (
        all_metrics_graph_df.groupby(["category_level"])
        .apply(lambda x: x.loc[x["macro_ap"].idxmax()], include_groups=False)
        .reset_index(drop=True)
    )

    best_prott5_perclass_df = pd.read_csv(
        "/scratch/groups/emmalu/seq2loc/sweep_experiments"
        + "/"
        + best_prott5_df["exp_name"][0]
        + "_"
        + os.path.basename(best_prott5_df["metadata_file"][0]).split(".")[0]
        + "/"
        + best_prott5_df["run_id"][0]
        + "/"
        + "/all_folds_perclass_metrics.csv",
    )
    best_prott5_perclass_df["Model"] = "ProtT5"

    best_prott5_graph_df = (
        all_metrics_df[
            (all_metrics_df["exp_name"] == "ProtT5")
            & (all_metrics_df["category_level"] == "level1")
        ]
        .sort_values(["macro_ap"], ascending=False)[0:1]
        .reset_index(drop=True)
    )
    best_prott5_graph_perclass_df = pd.read_csv(
        "/scratch/groups/emmalu/seq2loc/ppi_experiments/"
        + best_prott5_graph_df["exp_name"][0]
        + "_"
        + os.path.basename(best_prott5_graph_df["metadata_file"][0]).split(".")[0]
        + "_"
        + best_prott5_graph_df["run_id"][0]
        + "/inference_"
        + [
            key
            for key in INFERENCE_CONFIG_DICT.keys()
            if best_prott5_graph_df["Inference PPI"][0] in INFERENCE_CONFIG_DICT[key]
        ][0]
        + "_all_folds_perclass_metrics.csv",
    )
    best_prott5_graph_perclass_df["Model"] = "ProtT5 Graph"

    merged_perclass_df = pd.concat(
        [best_prott5_perclass_df, best_prott5_graph_perclass_df],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=merged_perclass_df,
        x="category",
        y="f1",
        hue="Model",
        ax=ax,
        # palette=["blue", "orange"],
    )
    plt.xticks(rotation=90)
    plt.savefig(
        f"{save_folder}/comparison_perclass_f1.png", bbox_inches="tight", dpi=300
    )
    plt.close()

    merged_perclass_df["precision"] = -merged_perclass_df["precision"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=merged_perclass_df,
        x="category",
        y="precision",
        hue="Model",
        ax=ax,
        # palette=["blue", "orange"],
    )
    sns.barplot(
        data=merged_perclass_df,
        x="category",
        y="recall",
        hue="Model",
        ax=ax,
        # palette=["blue", "orange"],
    )
    plt.xticks(rotation=450)
    plt.savefig(
        f"{save_folder}/comparison_perclass_precision_recall.png",
        bbox_inches="tight",
        dpi=300,
    )

    return best_prott5_perclass_df, best_prott5_graph_perclass_df


if __name__ == "__main__":
    all_sweep_df = pd.read_csv(
        "/scratch/groups/emmalu/seq2loc/sweep_analysis/overall_metrics.csv"
    )
    best_prott5_df = (
        all_sweep_df[
            (all_sweep_df["exp_name"] == "ProtT5")
            & (all_sweep_df["metadata_file"] == "hpa_uniprot_combined_trainset")
            & (all_sweep_df["category_level"] == "level1")
        ]
        .sort_values(["macro_ap"], ascending=False)[0:1]
        .reset_index(drop=True)
    )

    save_folder = "/scratch/groups/emmalu/seq2loc/ppi_analysis"

    exp_folder = "/scratch/groups/emmalu/seq2loc/ppi_experiments/"
    embedding_folder = "/scratch/groups/emmalu/seq2loc/embeddings/"

    all_exp_folders = sorted(glob.glob(f"{exp_folder}/*"))
    all_metrics = []

    for exp_folder in all_exp_folders:
        if not os.path.exists(f"{exp_folder}/config.yaml"):
            print(f"Skipping {exp_folder} as config.yaml not found", flush=True)
            continue
        if not "hpa_uniprot_combined_trainset" in exp_folder:
            continue
        print(f"Processing {exp_folder}")
        config = yaml.safe_load(open(f"{exp_folder}/config.yaml"))
        try:
            exp_metrics = main(exp_folder)
            for parameter in PARAMETERS:
                exp_metrics[parameter] = (
                    config[parameter][0]
                    if type(config[parameter]) == list
                    else config[parameter]
                )
            exp_metrics["run_id"] = exp_folder.split("hpa_uniprot_combined_trainset_")[
                1
            ]
            all_metrics.append(exp_metrics)
        except:
            pass

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_metrics_df = all_metrics_df[
        all_metrics_df["ppi_database"] == "string"
    ].reset_index(drop=True)
    all_metrics_df = all_metrics_df.sort_values(
        by=["exp_name", "category_level", "num_neighbors", "ppi_database"]
    ).reset_index(drop=True)
    all_metrics_df.to_csv(f"{save_folder}/overall_metrics.csv", index=False)

    generate_metrics_plots(all_metrics_df, best_prott5_df, save_folder)

    compare_perclass_comparison(all_sweep_df, all_metrics_df, save_folder)
