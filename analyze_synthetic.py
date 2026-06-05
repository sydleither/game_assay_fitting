import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils import (
    format_for_plotting,
    get_fit_df,
    label_data_type,
    label_qualitative_dynamics,
)


def plot_confusion_matrices(save_loc, df, data_type):
    def fmt(x):
        s = f"{x:.2f}"
        if s == "0.00":
            return "0"
        if s.startswith("0."):
            return s[1:]
        if s.endswith(".00"):
            return s[:-3]
        return s

    models = df["Model"].unique()
    labels = sorted(df["Dynamic True"].unique())
    fig, ax = plt.subplots(
        ncols=len(models) + 1,
        gridspec_kw=dict(width_ratios=[1] * len(models) + [0.1]),
        figsize=(5 * len(models), 5),
    )

    for i, model in enumerate(models):
        matrices = []
        for rep in df["Replicate"].unique():
            df_mod = df[(df["Model"] == model) & (df["Replicate"] == rep)]
            matrices.append(
                confusion_matrix(
                    df_mod["Dynamic True"],
                    df_mod["Dynamic"],
                    labels=labels,
                    normalize="true",
                )
            )
        matrices = np.stack(matrices)
        matrix_mean = np.mean(matrices, axis=0)

        matrix_ci_low = np.zeros_like(matrix_mean)
        matrix_ci_high = np.zeros_like(matrix_mean)
        for r in range(matrix_mean.shape[0]):
            for c in range(matrix_mean.shape[1]):
                cell_samples = matrices[:, r, c]
                res = stats.bootstrap(
                    (cell_samples,),
                    np.mean,
                    confidence_level=0.95,
                    n_resamples=1000,
                    method="percentile",
                )
                matrix_ci_low[r, c] = res.confidence_interval.low
                matrix_ci_high[r, c] = res.confidence_interval.high

        annot = [
            [f"{m:.2f}\n({fmt(lo)}, {fmt(hi)})" for m, lo, hi in zip(row_m, row_lo, row_hi)]
            for row_m, row_lo, row_hi in zip(matrix_mean, matrix_ci_low, matrix_ci_high)
        ]
        sns.heatmap(
            matrix_mean,
            annot=annot,
            fmt="",
            cmap=sns.cubehelix_palette(as_cmap=True),
            xticklabels=labels,
            yticklabels=labels if i == 0 else [],
            vmin=0,
            vmax=1,
            cbar=False,
            ax=ax[i],
        )
        acc = np.diagonal(matrices, axis1=1, axis2=2).mean(axis=1)
        acc_mean = np.mean(acc)
        res_acc = stats.bootstrap(
            (acc,),
            np.mean,
            confidence_level=0.95,
            n_resamples=1000,
            method="percentile",
        )
        acc_lo = res_acc.confidence_interval.low
        acc_hi = res_acc.confidence_interval.high
        ax[i].set(
            xlabel=model,
            ylabel="Ground Truth",
            title=f"Accuracy: {acc_mean:.2f} ({acc_lo:.2f}, {acc_hi:.2f})",
        )

    fig.colorbar(ax[1].collections[0], cax=ax[-1])
    ax[-1].set(ylabel="Proportion of True Classification")
    fig.suptitle(f"Normalized Confusion Matrices for {data_type} Data")
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/confusion_{data_type}.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_accuracy(save_loc, df, data_type):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(
        df.drop_duplicates(subset=["Replicate", "Model"]),
        x="Model",
        y="Accuracy",
        color="#8da0cb",
        ax=ax,
    )
    ax.tick_params("x", rotation=45)
    ax.set(title=f"Qualitative Interaction Classification Accuracy\non {data_type} Data")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/accuracy_{data_type}.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_true_class_balance(save_loc, df, data_type):
    fig, ax = plt.subplots(figsize=(4, 4))
    counts = (
        df[df["Model"] == "Ground Truth"].groupby(["Dynamic", "Replicate"]).size().reset_index()
    )
    sns.barplot(counts, x="Dynamic", y=0, color="#b3b3b3", ax=ax)
    ax.tick_params("x", rotation=45)
    ax.set(title=f"Class Balance of Generated\n{data_type} Data")
    ax.set(ylabel="Count Per-Replicate")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/balance_{data_type}.png", bbox_inches="tight", dpi=200)
    plt.close()


def main():
    # Read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str)
    args = parser.parse_args()
    save_loc = args.data_dir
    data_type = label_data_type(args.data_dir)

    # Get fits and ground truth
    df = []
    for rep in os.listdir(save_loc):
        if os.path.isfile(f"{save_loc}/{rep}"):
            continue
        df_gt = pd.read_csv(f"{save_loc}/{rep}/ground_truth.csv")
        df_gt["Model"] = "Ground Truth"
        df_rep = pd.concat([get_fit_df(f"{save_loc}/{rep}"), df_gt])
        df_rep["Replicate"] = rep
        df.append(df_rep)
    df = pd.concat(df)
    df = label_qualitative_dynamics(df)
    df = format_for_plotting(df)

    # Label ground truth per-row
    df_gt = df[df["Model"] == "Ground Truth"][["Replicate", "Experiment", "Dynamic"]]
    df_gt = df[df["Model"] != "Ground Truth"].merge(
        df_gt, on=["Replicate", "Experiment"], suffixes=("", " True")
    )
    df_gt = df_gt.drop_duplicates(subset=["Replicate", "Experiment", "Model"])
    df_gt["Correct"] = df_gt["Dynamic"] == df_gt["Dynamic True"]
    df_gt["Accuracy"] = df_gt.groupby(["Replicate", "Model"])["Correct"].transform("mean")
    df_gt = df_gt.sort_values(by=["Model", "Replicate", "Experiment"])

    # Plot
    plot_true_class_balance(save_loc, df, data_type)
    plot_accuracy(save_loc, df_gt, data_type)
    plot_confusion_matrices(save_loc, df_gt, data_type)


if __name__ == "__main__":
    main()
