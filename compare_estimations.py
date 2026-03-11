import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from compare_fits import get_fit_df, plot_errors, plot_errors_facet, label_qualitative_dynamics
from utils import get_parameter_names


def plot_qualitative(data_dir, df):
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

    fig, ax = plt.subplots(
        ncols=len(models) + 1,
        gridspec_kw=dict(width_ratios=[1] * len(models) + [0.1]),
        figsize=(4 * len(models), 4),
    )
    for i in range(len(models)):
        sns.heatmap(
            confusion_matrices[i],
            annot=True,
            xticklabels=labels,
            yticklabels=labels if i == 0 else [],
            vmin=0,
            vmax=num_experiments,
            cbar=False,
            ax=ax[i],
        )
        ax[i].set(
            xlabel="Ground Truth",
            ylabel=models[i],
            title=f"Accuracy: {np.trace(confusion_matrices[i])/num_experiments:5.3f}",
        )
    fig.colorbar(ax[1].collections[0], cax=ax[-1])
    ax[-1].set(ylabel="Number of Experiments")
    fig.suptitle("Accuracy of each Model")
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{data_dir}/accuracy_confmat.png", bbox_inches="tight", dpi=200)
    plt.close()


def qualitative_results(save_loc, df):
    df = label_qualitative_dynamics(df)
    plot_qualitative(save_loc, df)


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
    plot_errors_facet(save_loc, df_diff, sns.barplot, "Model", "Difference", None, "Parameter")
    plot_errors_facet(save_loc, df, sns.barplot, "Model", "Value", None, "Parameter")
    plot_errors_facet(save_loc, df, sns.scatterplot, "Model", "Value", None, "Parameter")
    plot_errors_facet(save_loc, df, sns.scatterplot, "Model", "Value", "Experiment", "Parameter")


def main():
    # Read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str, default="data/spatial_egt/formatted")
    args = parser.parse_args()

    # Get fits and ground truth
    fit_df = get_fit_df(args.data_dir)
    gt_df = pd.read_csv(f"{args.data_dir}/ground_truth.csv")

    # Combine fit and ground truth dataframes
    gt_df["Model"] = "Ground Truth"
    df = pd.concat([fit_df, gt_df])

    # Save results
    quantitative_results(args.data_dir, df)
    qualitative_results(args.data_dir, df)


if __name__ == "__main__":
    main()
