import argparse
from itertools import combinations
import os
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils import get_cell_types, get_parameter_names

filterwarnings("ignore")


##################
# Data Wrangling #
##################
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
    if "BIC" in growth_rate_df: #TODO
        growth_rate_df = growth_rate_df.drop("BIC", axis=1)
    growth_rate_df = growth_rate_df.rename(
        {
            "GrowthRate_error": "Error",
            "GrowthRate_BIC": "BIC",
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
            "BIC"
        ]
    ]
    # Combine dataframes
    df = growth_rate_df.merge(payoff_df, on=["DrugConcentration", "CellType"])
    df["Model"] = "Game Assay"
    return df.reset_index(drop=True)


def get_fit_df(data_dir):
    df = []
    for exp_name in os.listdir(data_dir):
        if os.path.isfile(f"{data_dir}/{exp_name}") or exp_name == "layout_files":
            continue
        sensitive_type, resistant_type = get_cell_types(exp_name)
        # Read in ODE fit data
        df_ode = []
        for file_name in os.listdir(f"{data_dir}/{exp_name}"):
            if file_name.split("_")[-1] == "fit.csv":
                df_ode.append(read_and_format_ode(data_dir, exp_name, file_name, sensitive_type))
        if len(df_ode) == 0:
            continue
        df_ode = pd.concat(df_ode, ignore_index=True)
        # Read in game assay fit data
        df_assay = read_and_format_game_assay(data_dir, exp_name, sensitive_type)
        # Merge game assay and ODE dataframes
        df_comb = pd.concat([df_ode, df_assay], ignore_index=True)
        df_comb["Experiment"] = exp_name
        df_comb["Resistant Type"] = resistant_type
        df.append(df_comb)
    df = pd.concat(df)
    df = df[df["DrugConcentration"] == 0.0]
    return df


def classify_game(a, b, c, d):
    if np.any(np.isnan([a, b, c, d])):
        return np.nan
    if a > c and b > d:
        return "Sensitive Wins"
    if c > a and b > d:
        return "Coexistence"
    if a > c and d > b:
        return "Bistability"
    if c > a and d > b:
        return "Resistant Wins"
    return "Unknown"


def classify_lv_dynamic(r_S, r_R, a_SS, a_SR, a_RS, a_RR):
    if np.any(np.isnan([r_S, r_R, a_SS, a_SR, a_RS, a_RR])):
        return np.nan
    denom = a_SS * a_RR - a_SR * a_RS
    mix = ((a_SR * r_R - a_RR * r_S) / denom, (a_RS * r_S - a_SS * r_R) / denom)
    if mix[0] < 0 or mix[1] < 0:
        mix_stable = np.nan
    else:
        mix_stable = denom > 0
    all_0_stable = r_R - a_RS * (r_S / a_SS) < 0
    all_1_stable = r_S - a_SR * (r_R / a_RR) < 0
    if np.isnan(mix_stable):
        if all_0_stable:
            return "Sensitive Wins"
        if all_1_stable:
            return "Resistant Wins"
        return "Extinction"
    else:
        if mix_stable:
            return "Coexistence"
        if not mix_stable:
            return "Bistability"


def label_qualitative_dynamics(df, keys=["Model", "Experiment"]):
    # Reduce dataframe
    for param in get_parameter_names():
        if param not in df:
            df[param] = np.nan
    df_q = df[keys + get_parameter_names()].drop_duplicates()

    # Get game quadrant of game-theoretic models
    df_q["Replicator Dynamic"] = df_q.apply(
        lambda x: classify_game(x["p_SS"], x["p_SR"], x["p_RS"], x["p_RR"]), axis=1
    )

    # Get fixed point dynamics of lotka-volterra models
    df_q["Lotka-Volterra Dynamic"] = df_q.apply(
        lambda x: classify_lv_dynamic(
            x["r_S"], x["r_R"], x["a_SS"], x["a_SR"], x["a_RS"], x["a_RR"]
        ),
        axis=1,
    )

    # Combine LV and EGT long-term dynamics columns
    df_q["Dynamic"] = df_q["Lotka-Volterra Dynamic"].fillna(df_q["Replicator Dynamic"])
    df_q = df_q[keys + ["Dynamic"]]
    return df.merge(df_q, on=keys)


def format_for_plotting(df):
    # Formatting exponential growth window
    if "Growth Rate Window" in df.columns:
        df.loc[df["Growth Rate Window"] == "per_exp", "Growth Rate Window"] = "Per-Experiment"
        df.loc[df["Growth Rate Window"] == "per_well", "Growth Rate Window"] = "Per-Replicate"
        df.loc[df["Growth Rate Window"] == "per_cell", "Growth Rate Window"] = "Per-Cell-Type"
        df.loc[df["Growth Rate Window"] == "none", "Growth Rate Window"] = "None"
        df = df.rename({"Growth Rate Window": "Exponential Growth Window Strategy"}, axis=1)
    # Formatting parameter names
    new_param_names = {}
    for param in get_parameter_names():
        new_param = param
        if "p_" in param or "a_" in param:
            new_param = param.upper()
        new_param = fr"${new_param[0]}_{{{new_param[2:]}}}$"
        new_param_names[param] = new_param
    df = df.rename(new_param_names, axis=1)
    # Renaming replicator to exponential growth
    df.loc[df["Model"] == "Replicator", "Model"] = "Exponential Growth ODE"
    df.loc[df["Model"] == "Lotka-Volterra", "Model"] = "Lotka-Volterra ODE"
    return df


#########
# Plots #
#########
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


def plot_errors_facet(save_loc, df, sns_plot, x, y, hue, facet):
    facet_grid = sns.FacetGrid(
        df,
        col=facet,
        hue=hue,
        hue_order=sorted(df[hue].unique()) if hue else None,
        sharex=False,
        sharey=False,
        height=4,
        aspect=1,
    )
    facet_grid.map_dataframe(sns_plot, x=x, y=y)
    if hue:
        facet_grid.add_legend()
    if x == "Experiment":
        facet_grid.set_xticklabels([])
    facet_grid.figure.patch.set_alpha(0.0)
    facet_grid.tight_layout()
    facet_grid.savefig(f"{save_loc}/ode_{x}_{y}_{hue}_{facet}.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_errors(save_loc, df, sns_plot, x, y, hue):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns_plot(df, x=x, y=y, hue=hue, ax=ax)
    title = f"{hue} {y} by {x}" if hue else f"{y} by {x}"
    ax.set(title=title)
    if x == "Experiment":
        ax.tick_params("x", rotation=90)
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/ode_{x}_{y}_{hue}.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_qualitative(data_dir, df, focal_col="Model", sub_col=None):
    df = df[["Experiment", focal_col, "Dynamic"]].copy().drop_duplicates()
    df = df.sort_values(by=[focal_col, "Experiment"])
    models = df[focal_col].unique()
    model_combos = list(combinations(models, 2))
    labels = sorted(df["Dynamic"].unique())
    num_experiments = len(df["Experiment"].unique())

    confusion_matrices = []
    for model1, model2 in model_combos:
        mat = confusion_matrix(
            df[df[focal_col] == model1]["Dynamic"],
            df[df[focal_col] == model2]["Dynamic"],
            labels=labels,
        )
        confusion_matrices.append(mat)

    fig, ax = plt.subplots(
        ncols=len(model_combos) + 1,
        gridspec_kw=dict(width_ratios=[1] * len(model_combos) + [0.1]),
        figsize=(4 * len(model_combos), 4),
    )
    for i in range(len(model_combos)):
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
            xlabel=model_combos[i][1],
            ylabel=model_combos[i][0],
            title=f"Agreement: {np.trace(confusion_matrices[i]) / num_experiments:5.3f}",
        )
    fig.colorbar(ax[0].collections[0], cax=ax[-1])
    ax[-1].set(ylabel="Number of Experiments")
    if not sub_col:
        fig.suptitle(f"Qualitative Agreement between {focal_col}s")
        figname = f"qualitative_agreement_{focal_col}.png"
    else:
        fig.suptitle(f"Qualitative Agreement between {focal_col}s for {sub_col}")
        figname = f"qualitative_agreement_{focal_col}_{sub_col}.png"
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{data_dir}/{figname}", bbox_inches="tight", dpi=200)
    plt.close()


########
# Main #
########
def main():
    # Read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str, default="data/experimental")
    args = parser.parse_args()

    # Read in fitting data
    df = get_fit_df(args.data_dir)

    # Formatting
    mean_error = df[["Model", "Experiment", "Error"]].groupby(["Model", "Experiment"]).mean()
    mean_error.reset_index()
    mean_error = mean_error.rename({"Error": "Mean Count Error"}, axis=1)
    df = df.merge(mean_error, on=["Model", "Experiment"])

    # Qualitative results
    df = label_qualitative_dynamics(df)
    plot_qualitative(args.data_dir, df)

    # Plot generic fitting errors
    plot_errors(args.data_dir, df, sns.barplot, "Model", "Error", None)
    plot_errors_facet(args.data_dir, df, sns.barplot, "Experiment", "Error", "Experiment", "Model")

    # Plot replicator vs game assay fitting errors
    df = df[df["Model"].isin(["Game Assay", "Replicator"])]
    plot_errors_facet(
        args.data_dir,
        df,
        sns.scatterplot,
        "Error",
        "Frequency Dependence Error",
        "Experiment",
        "Model",
    )
    plot_errors(args.data_dir, df, sns.barplot, "Model", "Frequency Dependence Error", None)
    plot_errors_facet(
        args.data_dir,
        df,
        sns.barplot,
        "Experiment",
        "Frequency Dependence Error",
        "Experiment",
        "Model",
    )
    plot_gamespaces(args.data_dir, df, "Resistant Type")
    plot_gamespaces(args.data_dir, df, "Frequency Dependence Error")
    plot_gamespaces(args.data_dir, df, "Mean Count Error")
    plot_gamespaces(args.data_dir, df, "Experiment")
    plot_overlaid_gamespace(args.data_dir, df)


if __name__ == "__main__":
    main()
