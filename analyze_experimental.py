import argparse

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import entropy

from utils import (
    get_colors,
    get_fit_df,
    label_qualitative_dynamics,
    format_for_plotting,
)


def plot_parameter_ranges(data_dir, df, data_type="Experimental"):
    df_param = pd.melt(
        df,
        id_vars=["Model", "Experiment"],
        value_vars=[x for x in df.columns if x[0] == "$"],
        var_name="Parameter",
        value_name="Value",
    )
    df_param = df_param.reset_index(drop=True).dropna().drop_duplicates()
    df_param.loc[df_param["Parameter"].str.contains("A"), "Model"] = (
        "Lotka-Volterra Interaction Matrix"
    )
    df_param.loc[df_param["Parameter"].str.contains("r"), "Model"] = (
        "Lotka-Volterra Intrinsic Growths"
    )
    df_param = df_param.sort_values(by=["Model", "Experiment", "Parameter"], ascending=False)

    facet_grid = sns.catplot(
        df_param,
        col="Model",
        x="Parameter",
        y="Value",
        kind="box",
        color="#FFFFFF",
        sharex=False,
        sharey=False,
        height=4,
        aspect=1,
    )
    facet_grid.set_titles("{col_name}")
    facet_grid.figure.suptitle(f"Parameter Ranges of Models Fit on {data_type} Data")
    facet_grid.figure.patch.set_alpha(0.0)
    facet_grid.tight_layout()
    facet_grid.savefig(
        f"{data_dir}/parameters_{data_type}.png", bbox_inches="tight", dpi=200
    )
    plt.close()


def plot_dynamics(save_loc, df, data_type="Experimental"):
    dynamics = list(get_colors().keys())
    n_colors = len(dynamics)
    palette_list = [get_colors()[d] for d in dynamics]
    custom_cmap = mcolors.ListedColormap(palette_list)
    dynamics_to_int = {s: i for i, s in enumerate(dynamics)}

    df = df[["Experiment", "Model", "Dynamic"]].drop_duplicates()
    df["Experiment"] = df["Experiment"].map({s: i for i, s in enumerate(df["Experiment"].unique())})
    df = df.pivot(index="Experiment", columns="Model", values="Dynamic").replace(dynamics_to_int)

    fig, ax = plt.subplots(figsize=(6, 4))
    res = sns.heatmap(
        df,
        cmap=custom_cmap,
        ax=ax,
        vmin=-0.5,
        vmax=n_colors - 0.5,
        cbar=True,
        cbar_kws={"ticks": range(n_colors), "label": "Dynamics"},
    )
    colorbar = res.collections[0].colorbar
    colorbar.set_ticklabels(dynamics)
    colorbar.set_label("Dynamics")
    ax.set_title(f"Qualitative Interaction Classification of {data_type} Data")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(
        f"{save_loc}/classifications_{data_type}.png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close()


def plot_entropy(save_loc, df, data_type="Experimental"):
    df = df[["Experiment", "Model", "Dynamic"]].drop_duplicates()
    entropies = (
        df.groupby("Model")["Dynamic"].value_counts(normalize=True).groupby(level=0).apply(entropy)
    )
    df = df.merge(entropies, on="Model")
    df = df.rename({"proportion": "Entropy"}, axis=1)
    df = df.drop_duplicates("Model")
    print(df)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(df, x="Model", y="Entropy", color="#fc8d62", ax=ax)
    ax.set_title(f"Entropy of Qualitative Interaction Classifications\nfor {data_type} Data")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/entropy_{data_type}.png", bbox_inches="tight", dpi=200)
    plt.close()


def main():
    # Read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str, default="data/experimental/0")
    args = parser.parse_args()

    # Read in fitting data
    df = get_fit_df(args.data_dir)
    df["Replicate"] = 0
    df = label_qualitative_dynamics(df)
    df = format_for_plotting(df)
    df = df.sort_values(by=["Model", "Experiment"])

    # Plot results
    plot_entropy(args.data_dir, df)
    plot_dynamics(args.data_dir, df)
    plot_parameter_ranges(args.data_dir, df)


if __name__ == "__main__":
    main()
