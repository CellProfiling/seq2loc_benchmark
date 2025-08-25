import copy
import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yaml
from statsmodels.formula.api import ols

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

PARAMETERS = {
    "exp_name": ["ProtBert", "ProtT5", "ESM2", "ESM3"],
    "category_level": ["level1", "level2", "level3"],
    "metadata_file": [
        "hpa_trainset",
        "uniprot_trainset",
        "hpa_uniprot_combined_trainset",
        "hpa_uniprot_combined_human_trainset",
    ],
    "clip_len": [512, 1024, 2048],
    "mlp_dropout": [0, 0.25, 0.5],
    "agg_method": [
        "MaxPool",
        "MeanPool",
        "LightAttentionPool",
        "MultiHeadAttentionPool",
    ],
    "loss": ["BCEWithLogitsLoss", "SigmoidFocalLoss"],
}
LEVEL_CLASSES = yaml.safe_load(open("metadata/level_classes.yaml"))
plt.rcParams["pdf.fonttype"] = 42


def generate_sweep_plots(metric_df, sweep_path, save_path="valid"):
    sweep_save_path = f"{sweep_path}/plots"
    if not os.path.exists(sweep_save_path):
        os.makedirs(sweep_save_path)

    save_path = f"{sweep_save_path}/{save_path}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for loss in ["SigmoidFocalLoss", "BCEWithLogitsLoss"]:
        metric_df_loss = metric_df[metric_df["loss"] == loss]

        loss_save_path = f"{save_path}/{loss}"
        if not os.path.exists(save_path):
            os.makedirs(loss_save_path)

        for metadata_file in metric_df_loss["metadata_file"].unique():
            metric_df_loss_metadata = metric_df_loss[
                metric_df_loss["metadata_file"] == metadata_file
            ]
            metric_df_loss_metadata.loc[:, "agg_method"] = pd.Categorical(
                metric_df_loss_metadata["agg_method"],
                categories=AGG_METHODS,
                ordered=True,
            )
            metric_df_loss_metadata.loc[:, "exp_name"] = pd.Categorical(
                metric_df_loss_metadata["exp_name"],
                categories=["ProtT5", "ProtBert", "ESM2", "ESM3"],
                ordered=True,
            )


def get_anova_table(valid_metrics, parameters, metrics):
    anova_tables = []
    for metric in metrics:
        for param in parameters:
            formula = f"{metric} ~ C({param})"
            model = ols(formula, data=valid_metrics).fit()
            anova_table = sm.stats.anova_lm(model, typ=2, robust="hc3")
            anova_table = anova_table.rename(
                columns={
                    "df": "Degrees of Freedom",
                    "F": "F-statistic",
                    "PR(>F)": "p-value",
                }
            )
            anova_table = anova_table.drop(columns=["sum_sq"])
            anova_table = anova_table.drop(index="Residual")
            anova_table["metric"] = metric
            anova_tables.append(anova_table)
    anova_tables = pd.concat(anova_tables)
    return anova_tables


def effect_plot(df, var, save_path):
    effect_save_folder = f"{save_path}/{var}"
    shutil.rmtree(effect_save_folder)
    os.makedirs(effect_save_folder, exist_ok=True)

    df[var] = pd.Categorical(
        df[var],
        categories=PARAMETERS[var],
        ordered=True,
    )
    for category_level in ["level1", "level2", "level3"]:
        g = sns.jointplot(
            x="micro_ap",
            y="macro_ap",
            hue=var,
            data=df[df["category_level"] == category_level],
            # kind="kde",
            palette="tab10",
            s=10,
            alpha=0.5,
        )
        g.plot_joint(
            sns.kdeplot,
            hue=var,
            palette="tab10",
            levels=4,
            # thresh=0.1,
            fill=False,
            alpha=0.5,
            linewidths=3,
        )
        g.savefig(
            f"{effect_save_folder}/macro_vs_micro_ap_cat_{category_level}.pdf",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    pivot_df = df.pivot_table(
        index=set(PARAMETERS.keys()).difference({var}), columns=var, values=METRICS
    )

    comp_params = copy.deepcopy(PARAMETERS[var])
    baseline = comp_params.pop(0)

    for param in comp_params:
        pivot_df[pd.MultiIndex.from_product([METRICS, [param]])] = (
            pivot_df[pd.MultiIndex.from_product([METRICS, [param]])].values
            - pivot_df[pd.MultiIndex.from_product([METRICS, [baseline]])].values
        )
    pivot_df[pd.MultiIndex.from_product([METRICS, [baseline]])] = 0

    # for metric in METRICS:
    #     g = sns.pairplot(
    #         data=pivot_df[metric].reset_index(),
    #         kind="scatter",
    #         # markers="x",
    #         hue="category_level",
    #         vars=PARAMETERS[var],
    #     )
    #     # g.map_lower(sns.violinplot)
    #     plt.savefig(
    #         f"{effect_save_folder}/{metric}_pairplot.pdf",
    #         dpi=200,
    #         bbox_inches="tight",
    #     )
    #     plt.close()

    all_difference_df = []
    for param in comp_params:
        difference_df = (
            pivot_df.xs(param, level=var, axis=1)
            - pivot_df.xs(baseline, level=var, axis=1)
        ).reset_index()
        difference_df["Method"] = param
        # print(difference_df.head())
        all_difference_df.append(difference_df)
    all_difference_df = pd.concat(all_difference_df, ignore_index=True)

    for category_level in ["level1", "level2", "level3"]:
        g = sns.jointplot(
            x="micro_ap",
            y="macro_ap",
            hue="Method",
            data=all_difference_df[
                all_difference_df["category_level"] == category_level
            ],
            # kind="kde",
            palette="tab10",
            s=5,
            alpha=0.75,
        )
        g.plot_joint(
            sns.kdeplot,
            hue="Method",
            palette="tab10",
            levels=4,
            # thresh=0.1,
            fill=False,
            alpha=0.5,
        )
        g.savefig(
            f"{effect_save_folder}/macro_vs_micro_ap_cat_rel_{category_level}.pdf",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    for metric in METRICS:
        g = sns.FacetGrid(
            all_difference_df,
            col="category_level",
            hue="Method",
            sharex=True,
            sharey=False,
        ).map_dataframe(
            sns.violinplot,
            x="Method",
            y=metric,
            data=all_difference_df,  # , errorbar="sd"
        )
        g.map_dataframe(
            sns.stripplot,
            x="Method",
            y=metric,
            data=all_difference_df,
            dodge=True,
            jitter=False,
            # hue="Method",
            # palette="tab10",
            color="black",
            size=3,
            alpha=0.7,
        )
        g.set_xticklabels(rotation=90)
        g.add_legend()
        g.savefig(
            f"{effect_save_folder}/{metric}.pdf",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()


def sweep_analysis(metric_path, save_path):
    sweep_df = pd.read_csv(metric_path)

    effect_plot(sweep_df, "exp_name", save_path)
    effect_plot(sweep_df, "metadata_file", save_path)
    effect_plot(sweep_df, "clip_len", save_path)
    effect_plot(sweep_df, "mlp_dropout", save_path)
    effect_plot(sweep_df, "agg_method", save_path)
    effect_plot(sweep_df, "loss", save_path)

    best_level_save_folder = f"{save_path}/best_level"
    os.makedirs(best_level_save_folder, exist_ok=True)

    best_exp_level_save_folder = f"{save_path}/best_exp_level"
    os.makedirs(best_exp_level_save_folder, exist_ok=True)

    for metric in METRICS:
        best_exp_configs = (
            sweep_df.groupby(["exp_name", "category_level"])
            .apply(
                lambda x: (
                    x.nlargest(3, [metric])
                    if metric != "coverage_error"
                    else x.nsmallest(3, [metric])
                ),
                include_groups=False,
            )
            .reset_index(drop=True)
        )
        best_exp_configs.to_csv(
            f"{best_exp_level_save_folder}/{metric}.csv", index=False
        )

        best_level_configs = (
            sweep_df.groupby(["category_level"]).apply(
                lambda x: (
                    x.nlargest(3, [metric])
                    if metric != "coverage_error"
                    else x.nsmallest(3, [metric])
                ),
                include_groups=False,
            )
        ).reset_index()
        best_level_configs.to_csv(f"{best_level_save_folder}/{metric}.csv", index=False)


if __name__ == "__main__":
    sweep_metric_path = (
        f"/scratch/groups/emmalu/seq2loc/sweep_analysis/overall_metrics.csv"
    )
    sweep_save_path = "/scratch/groups/emmalu/seq2loc/sweep_analysis"

    sweep_analysis(sweep_metric_path, sweep_save_path)

    ppi_metric_path = f"/scratch/groups/emmalu/seq2loc/ppi_analysis/all_ppi_metrics.csv"
    ppi_save_path = "/scratch/groups/emmalu/seq2loc/ppi_analysis"

    ppi_df = pd.read_csv(ppi_metric_path)
