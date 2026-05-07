import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from compare_fits import (
    format_for_plotting,
    get_fit_df,
    plot_errors_facet,
    label_qualitative_dynamics,
)
from utils import get_parameter_names, label_data_type


def plot_accuracy(data_dir, df, data_type):
    df = df[["Experiment", "Model", "Dynamic"]].copy().drop_duplicates()
    df = df.sort_values(by=["Model", "Experiment"])
    models = df[df["Model"] != "Ground Truth"]["Model"].unique()
    labels = sorted(df["Dynamic"].unique())
    num_experiments = len(df["Experiment"].unique())

    confusion_matrices = []
    for model in models:
        mat = confusion_matrix(
            df[df["Model"] == "Ground Truth"]["Dynamic"],
            df[df["Model"] == model]["Dynamic"],
            labels=labels,
        )
        confusion_matrices.append(mat)

    # heatmaps
    fig, ax = plt.subplots(
        ncols=len(models) + 1,
        gridspec_kw=dict(width_ratios=[1] * len(models) + [0.1]),
        figsize=(4 * len(models), 4),
    )
    accuracies = []
    for i in range(len(models)):
        acc = np.trace(confusion_matrices[i]) / num_experiments
        accuracies.append({"Model": models[i], "Accuracy": acc})
        sns.heatmap(
            confusion_matrices[i],
            annot=True,
            xticklabels=labels,
            yticklabels=labels if i == 0 else [],
            vmin=0,
            vmax=num_experiments // 2,
            cbar=False,
            ax=ax[i],
        )
        ax[i].set(
            xlabel=models[i],
            ylabel="Ground Truth",
            title=f"Accuracy: {acc:5.3f}",
        )
    fig.colorbar(ax[1].collections[0], cax=ax[-1])
    ax[-1].set(ylabel="Number of Experiments")
    fig.suptitle(f"Qualitative Interaction Classification Accuracy for {data_type} Data")
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{data_dir}/accuracy_confmat_{data_type}.png", bbox_inches="tight", dpi=200)
    plt.close()

    # barplot
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(pd.DataFrame(accuracies), x="Model", y="Accuracy", color="#9a0eea", ax=ax)
    ax.tick_params("x", rotation=45)
    ax.set(title=f"Qualitative Interaction Classification Accuracy\nfor {data_type} Data")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{data_dir}/accuracy_{data_type}.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_freq_dependence_fits(save_loc, df, data_type):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(
        df[df["Model"] == "Game Assay"].sort_values(by="Dynamic"),
        x="Dynamic",
        y="Frequency Dependence Error",
        ax=ax,
    )
    ax.set(
        title=f"Game Assay Fit Error on {data_type} Data",
        ylabel="Growth Rate by Fraction Sensitive\nFit Error",
    )
    ax.tick_params("x", rotation=45)
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/{data_type}_assay_freqdepend_fits.png", bbox_inches="tight", dpi=200)
    plt.close()

    # print(
    #     df[["Model", "Dynamic", "Frequency Dependence Error"]]
    #     .drop_duplicates()
    #     .groupby(["Model", "Dynamic"])
    #     .agg(["mean", "sem"])
    # )


def qualitative_results(save_loc, df, data_type):
    df = label_qualitative_dynamics(df)
    df = format_for_plotting(df)
    plot_accuracy(save_loc, df, data_type)

    df_gt = df[df["Model"] == "Ground Truth"][["Experiment", "Dynamic"]]
    df = df[df["Model"] != "Ground Truth"].merge(df_gt, on="Experiment", suffixes=("", " True"))
    plot_freq_dependence_fits(save_loc, df, data_type)


def quantitative_results(save_loc, df):
    # Pivot dataframe from wide to long
    for param in get_parameter_names():
        if param not in df:
            df[param] = np.nan
    df = pd.melt(
        df,
        id_vars=["Model", "Experiment"],
        value_vars=get_parameter_names(),
        var_name="Parameter",
        value_name="Value",
    )
    df = df.reset_index(drop=True)

    # Get absolute differences
    for param in get_parameter_names():
        for experiment in df["Experiment"].unique():
            filt = (df["Experiment"] == experiment) & (df["Parameter"] == param)
            gt_value = df[(df["Model"] == "Ground Truth") & filt]["Value"].values[0]
            df.loc[filt, "Difference"] = np.abs(gt_value - df["Value"])
    df = df.dropna(axis=0)

    # Plot differences
    df_diff = df[df["Model"] != "Ground Truth"]
    plot_errors_facet(save_loc, df_diff, sns.boxplot, "Model", "Difference", None, "Parameter")


def main():
    # Read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str)
    args = parser.parse_args()

    # Get fits and ground truth
    fit_df = get_fit_df(args.data_dir)
    gt_df = pd.read_csv(f"{args.data_dir}/ground_truth.csv")
    gt_df["Model"] = "Ground Truth"
    df = pd.concat([fit_df, gt_df])
    data_type = label_data_type(args.data_dir)

    # Save results
    qualitative_results(args.data_dir, df, data_type)
    quantitative_results(args.data_dir, df)


if __name__ == "__main__":
    main()
