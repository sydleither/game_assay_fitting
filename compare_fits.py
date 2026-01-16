import argparse
import os
from warnings import filterwarnings

import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import euclidean
import seaborn as sns

from utils import get_cell_types

filterwarnings("ignore")


def read_and_format_ode(data_dir, exp_name, file_name, sensitive_type):
    model = file_name.split("_")[-2].title()
    df = pd.read_csv(f"{data_dir}/{exp_name}/{file_name}")
    df = df.drop(["Time", "Count"], axis=1).drop_duplicates()
    df = df.rename({f"Fraction_{sensitive_type}": "Fraction Sensitive"}, axis=1)
    df["Model"] = model
    if model == "Replicator":
        df["Advantage Sensitive"] = df["p_SR"] - df["p_RR"]
        df["Advantage Resistant"] = df["p_RS"] - df["p_SS"]
    return df.reset_index(drop=True)


def read_and_format_game_assay(data_dir, exp_name, sensitive_type):
    # Read in growth rate and payoff data
    gr_path = f"{data_dir}/{exp_name}/{exp_name}_growth_rate_df_processed.csv"
    payoff_path = f"{data_dir}/{exp_name}/{exp_name}_game_params_df_processed.csv"
    if not os.path.exists(gr_path) or not os.path.exists(payoff_path):
        return
    growth_rate_df = pd.read_csv(gr_path)
    payoff_df = pd.read_csv(payoff_path)
    # Transform payoff dataframe from wide format to long
    payoff_df = payoff_df.drop(["error"], axis=1)
    payoff_df = pd.wide_to_long(
        payoff_df, stubnames=["Type", "error"], i="DrugConcentration", j="n"
    )
    payoff_df = payoff_df.reset_index()
    payoff_df = payoff_df.rename(
        {
            "Type": "CellType",
            "error": "Frequency Dependence Error",
            "p11": "p_SS",
            "p12": "p_SR",
            "p21": "p_RS",
            "p22": "p_RR",
            "Advantage_0": "Advantage Sensitive",
            "Advantage_1": "Advantage Resistant",
        },
        axis=1,
    )
    payoff_df = payoff_df.drop(["n", "c12", "c21", "r1", "r2"], axis=1)
    # Format growth rate dataframe
    growth_rate_df = growth_rate_df.rename(
        {
            "GrowthRate_error": "Error",
            f"Fraction_{sensitive_type}": "Fraction Sensitive",
        },
        axis=1,
    )
    growth_rate_df = growth_rate_df[
        [
            "PlateId",
            "WellId",
            "DrugConcentration",
            "CellType",
            "Fraction Sensitive",
            "GrowthRate",
            "GrowthRate_lowerBound",
            "GrowthRate_higherBound",
            "Intercept",
            "GrowthRate_window_start",
            "GrowthRate_window_end",
            "Error",
        ]
    ]
    # Combine dataframes
    df = growth_rate_df.merge(payoff_df, on=["DrugConcentration", "CellType"])
    df["Model"] = "Game Assay"
    return df.reset_index(drop=True)


def plot_gamespaces(save_loc, df, hue):
    df = df.drop_duplicates(subset=["Model", "Experiment"])
    df = df.dropna(subset=["Frequency Dependence Error"], axis=0)
    error_hue = "error" in hue.lower()

    if error_hue:
        cmap = sns.color_palette("crest", as_cmap=True)
        norm = plt.Normalize(vmin=df[hue].min(), vmax=df[hue].max())
    else:
        cmap = "Set2"
        norm = None

    facet = sns.FacetGrid(df, col="Model", height=4, aspect=1)
    facet.map_dataframe(
        sns.scatterplot,
        x="Advantage Resistant",
        y="Advantage Sensitive",
        hue=hue,
        palette=cmap,
        hue_norm=norm,
        legend=not error_hue,
    )

    if error_hue:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        facet.figure.subplots_adjust(right=1.165)
        cbar = facet.figure.colorbar(sm, ax=facet.axes, orientation="vertical")
        cbar.set_label(hue)
        cbar.outline.set_visible(False)
    else:
        facet.add_legend()

    facet.map(plt.axhline, y=0, color="gray")
    facet.map(plt.axvline, x=0, color="gray")
    facet.figure.patch.set_alpha(0.0)
    facet.savefig(f"{save_loc}/gamespace_{hue}.png", dpi=200)
    plt.close()


def plot_overlaid_gamespace(save_loc, df):
    df = df.drop_duplicates(subset=["Model", "Experiment"])
    df = df.dropna(subset=["Frequency Dependence Error"], axis=0)
    if len(df["Model"].unique()) > 2:
        print("Too many payoff-based models to overlay.")
        return
    experiments = df["Experiment"].unique()
    models = sorted(df["Model"].unique())

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    # Plot game space
    sns.scatterplot(
        df,
        x="Advantage Resistant",
        y="Advantage Sensitive",
        hue="Experiment",
        hue_order=experiments,
        style="Model",
        style_order=models,
        legend=False,
        ax=ax[0],
    )
    ax[0].axhline(y=0, color="gray")
    ax[0].axvline(x=0, color="gray")
    # Plot error distribution
    df["Coordinates"] = df[["Advantage Resistant", "Advantage Sensitive"]].values.tolist()
    df = df.pivot(index="Experiment", columns="Model", values="Coordinates").reset_index()
    df["Error"] = df.apply(lambda x: euclidean(x["Game Assay"], x["Replicator"]), axis=1)
    sns.barplot(
        df,
        x="Experiment",
        y="Error",
        hue="Experiment",
        hue_order=experiments,
        legend=False,
        ax=ax[1],
    )
    ax[1].set(ylabel="Euclidean Distance")
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].axhline(y=df["Error"].mean(), color="gray", ls="--")
    # Save figure
    fig.suptitle("Game Assay vs Replicator Game Space")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/gamespace_overlaid.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_errors(save_loc, df, y):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(df, x="Model", y=y, ax=ax)
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/ode_bar_{y}.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_errors_per_experiment(save_loc, df, y):
    fig, ax = plt.subplots(figsize=(5, 8))
    sns.barplot(df, x="Experiment", y=y, hue="Model", ax=ax)
    ax.tick_params("x", rotation=90)
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/ode_bar_{y}_exp.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_errors_scatter(save_loc, df, x, y, hue):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.scatterplot(df, x=x, y=y, hue=hue, alpha=0.5, ax=ax)
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/ode_scatter_{x}_{y}_{hue}.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_errors_box(save_loc, df, x, y, hue):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.boxplot(df, x=x, y=y, hue=hue, ax=ax)
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/ode_box_{x}_{y}_{hue}.png", bbox_inches="tight", dpi=200)
    plt.close()


def main():
    # Read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str, default="data/experimental")
    parser.add_argument("-exp", "--exp_name", type=str, default=None)
    args = parser.parse_args()

    # Read in fitting data
    df = []
    for exp_name in os.listdir(args.data_dir):
        if os.path.isfile(f"{args.data_dir}/{exp_name}") or exp_name == "layout_files":
            continue
        sensitive_type, resistant_type = get_cell_types(exp_name)
        # Read in ODE fit data
        df_ode = []
        for file_name in os.listdir(f"{args.data_dir}/{exp_name}"):
            if file_name.split("_")[-1] == "fit.csv":
                df_ode.append(
                    read_and_format_ode(args.data_dir, exp_name, file_name, sensitive_type)
                )
        if len(df_ode) == 0:
            continue
        df_ode = pd.concat(df_ode)
        # Read in game assay fit data
        df_assay = read_and_format_game_assay(args.data_dir, exp_name, sensitive_type)
        # Merge game assay and ODE dataframes
        df_comb = pd.concat([df_ode, df_assay])
        df_comb["Experiment"] = exp_name
        df_comb["Resistant Type"] = resistant_type
        df.append(df_comb)
    df = pd.concat(df)

    # Formatting
    df = df[df["DrugConcentration"] == 0.0]
    mean_error = df[["Model", "Experiment", "Error"]].groupby(["Model", "Experiment"]).mean()
    mean_error.reset_index()
    mean_error = mean_error.rename({"Error": "Mean Count Error"}, axis=1)
    df = df.merge(mean_error, on=["Model", "Experiment"])
    df["Growth Rate Window Size"] = df["GrowthRate_window_end"] - df["GrowthRate_window_start"]

    # Plot generic errors
    plot_errors(args.data_dir, df, "Error")
    plot_errors_per_experiment(args.data_dir, df, "Error")
    plot_errors_scatter(args.data_dir, df, "Fraction Sensitive", "Error", "Model")

    # Plot replicator vs game assay
    df = df[df["Model"].isin(["Game Assay", "Replicator"])]
    df.to_csv("gr_window_subset.csv", index=False)
    plot_errors_scatter(args.data_dir, df, "GrowthRate_window_start", "Error", "Model")
    plot_errors_box(args.data_dir, df, "Growth Rate Window Size", "Error", "Model")
    plot_errors_per_experiment(args.data_dir, df, "Frequency Dependence Error")
    plot_errors(args.data_dir, df, "Frequency Dependence Error")
    plot_gamespaces(args.data_dir, df, "Resistant Type")
    plot_gamespaces(args.data_dir, df, "Frequency Dependence Error")
    plot_gamespaces(args.data_dir, df, "Mean Count Error")
    plot_gamespaces(args.data_dir, df, "Experiment")
    plot_overlaid_gamespace(args.data_dir, df)


if __name__ == "__main__":
    main()
