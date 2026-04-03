import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score

from compare_fits import get_fit_df, label_qualitative_dynamics, plot_errors, plot_errors_facet
from fit_ode import fit
from game_assay.game_analysis import calculate_counts, calculate_growth_rates, calculate_payoffs
from game_assay.game_analysis_utils import optimize_growth_rate_window
from run_game_assay import plot_fits, plot_freqdepend_fit
from utils import get_cell_types, get_parameter_names


def plot_qualitative(data_dir, df, model):
    df_model = df[(df["Model"] == model) | (df["Model"] == "Ground Truth")]
    df_q = df_model[["Growth Rate Window", "Experiment", "Dynamic"]].drop_duplicates()
    df_q = df_q.sort_values(by=["Growth Rate Window", "Experiment"])
    gr_windows = df_model["Growth Rate Window"].unique()

    agreements = []
    for i in range(len(gr_windows)):
        for j in range(i + 1, len(gr_windows)):
            acc = accuracy_score(
                df_q[df_q["Growth Rate Window"] == gr_windows[i]]["Dynamic"],
                df_q[df_q["Growth Rate Window"] == gr_windows[j]]["Dynamic"],
            )
            agreements.append(
                {
                    "Growth Rate Window 1": gr_windows[i],
                    "Growth Rate Window 2": gr_windows[j],
                    "Agreement": acc,
                }
            )

    df_agr = pd.DataFrame(agreements)
    df_agr2 = df_agr.copy()
    df_agr2["temp"] = df_agr2["Growth Rate Window 1"]
    df_agr2["Growth Rate Window 1"] = df_agr2["Growth Rate Window 2"]
    df_agr2["Growth Rate Window 2"] = df_agr2["temp"]
    df_agr = pd.concat([df_agr, df_agr2])
    df_acc = df_agr[df_agr["Growth Rate Window 1"] == "Ground Truth"].copy()
    df_acc = df_acc.rename(
        {"Growth Rate Window 2": "Growth Rate Window", "Agreement": "Accuracy"}, axis=1
    )
    df_agr = df_agr[
        (df_agr["Growth Rate Window 1"] != "Ground Truth")
        & (df_agr["Growth Rate Window 2"] != "Ground Truth")
    ]
    df_agr = df_agr.pivot(
        index="Growth Rate Window 1", columns="Growth Rate Window 2", values="Agreement"
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(df_agr, annot=True, vmin=0, vmax=1, ax=ax)
    ax.set_title(f"Agreement Between Growth Rate Windows for {model}")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{data_dir}/agreement_{model}.png", bbox_inches="tight", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(df_acc, x="Growth Rate Window", y="Accuracy", ax=ax)
    ax.set_title(f"Accuracy of Growth Rate Windows for {model}")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{data_dir}/accuracy_{model}.png", bbox_inches="tight", dpi=200)
    plt.close()


def get_ground_truth(in_data_dir, df):
    if not os.path.exists(f"{in_data_dir}/ground_truth.csv"):
        return df
    df_gt = pd.read_csv(f"{in_data_dir}/ground_truth.csv")
    df_gt["Growth Rate Window"] = "Ground Truth"
    df_gt["Model"] = "Ground Truth"
    return pd.concat([df, df_gt])


def get_growth_rates(data_dir, in_dir):
    # Read in all the data files
    df = []
    for gr_window in os.listdir(data_dir):
        if os.path.isfile(f"{data_dir}/{gr_window}"):
            continue
        df_window = get_fit_df(f"{data_dir}/{gr_window}")
        df_window["Growth Rate Window"] = gr_window.title()
        df.append(df_window)
    df = pd.concat(df)

    # Normalize error by growth rate window size
    df["Window Size"] = (df["GrowthRate_window_end"] - df["GrowthRate_window_start"]) / 4
    # df["Error"] = df["Error"] / df["Window Size"]

    # Final formatting and return
    df = get_ground_truth(in_dir, df)
    df = label_qualitative_dynamics(df, ["Growth Rate Window", "Model", "Experiment"])
    return df.reset_index()


def save_growth_rate(in_data_dir, out_data_dir, exp_name, window):
    save_loc = f"{out_data_dir}/{window}/{exp_name}"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    sensitive_type, resistant_type = get_cell_types(exp_name)
    cell_types = [sensitive_type, resistant_type]
    counts_df = calculate_counts(in_data_dir, exp_name)

    gr_window = None
    if window == "none":
        gr_window = (counts_df["Time"].min(), counts_df["Time"].max())
    elif window == "per_exp":
        counts_df = optimize_growth_rate_window(counts_df)

    growth_rate_df = calculate_growth_rates(
        f"{out_data_dir}/{window}",
        exp_name,
        counts_df,
        growth_rate_window=gr_window,
        cell_type_list=cell_types,
    )
    cell_colors = {sensitive_type: "#4C956C", resistant_type: "#EF7C8E"}
    plot_fits(save_loc, exp_name, counts_df, growth_rate_df, cell_types, cell_colors)

    payoff_df = calculate_payoffs(
        f"{out_data_dir}/{window}",
        exp_name,
        growth_rate_df,
        cell_types,
        f"Fraction_{sensitive_type}",
    )
    plot_freqdepend_fit(save_loc, exp_name, growth_rate_df, payoff_df, cell_colors, cell_types)

    fit(
        f"{out_data_dir}/{window}",
        exp_name,
        "replicator",
        counts_df,
        growth_rate_df,
        save_figs=True,
    )
    fit(
        f"{out_data_dir}/{window}",
        exp_name,
        "lotka-volterra",
        counts_df,
        growth_rate_df,
        save_figs=True,
    )


def main():
    # Read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str, default="data")
    parser.add_argument("-in", "--in_dir", type=str, default="experimental")
    parser.add_argument("-out", "--out_dir", type=str, default="gr_experimental")
    parser.add_argument("-w", "--window", type=str, choices=["none", "per_exp", "per_well"])
    parser.add_argument("-plot", "--plot", type=int, default=0)
    args = parser.parse_args()

    # Check that input data path exists
    if not os.path.exists(f"{args.data_dir}/{args.in_dir}"):
        raise ValueError("Path to existing data does not exist.")

    # Create new path for output data
    save_loc = f"{args.data_dir}/{args.out_dir}"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    # Save growth rate dataframes for each experiment
    if args.plot == 0:
        for exp_name in os.listdir(f"{args.data_dir}/{args.in_dir}"):
            if (
                os.path.isfile(f"{args.data_dir}/{args.in_dir}/{exp_name}")
                or exp_name == "layout_files"
            ):
                continue
            print(exp_name)
            save_growth_rate(
                f"{args.data_dir}/{args.in_dir}",
                save_loc,
                exp_name,
                args.window,
            )

    # Save results
    if args.plot == 1:
        df = get_growth_rates(save_loc, f"{args.data_dir}/{args.in_dir}")
        df.loc[df["Error"] > 0.1, "Error"] = 0.1 #TODO remove

        if "Ground Truth" not in df["Model"].unique():
            gt = []
            for exp in df["Experiment"].unique():
                gt.append(
                    {
                        "Experiment": exp,
                        "Model": "Ground Truth",
                        "Growth Rate Window": "Ground Truth",
                        "Dynamic": "Sensitive Wins",
                    }
                )
            df = pd.concat([df, pd.DataFrame(gt)])

        plot_qualitative(save_loc, df, "Game Assay")
        plot_qualitative(save_loc, df, "Replicator")
        plot_qualitative(save_loc, df, "Lotka-Volterra")

        df = df[(df["Growth Rate Window"] != "Ground Truth") & (df["Model"] != "Ground Truth")]
        df_freqdepend = df[df["Model"] != "Lotka-Volterra"][
            ["Growth Rate Window", "Model", "Experiment", "Frequency Dependence Error"]
        ].drop_duplicates()
        plot_errors(save_loc, df, sns.barplot, "Model", "Error", "Growth Rate Window")
        plot_errors(
            save_loc,
            df_freqdepend,
            sns.barplot,
            "Model",
            "Frequency Dependence Error",
            "Growth Rate Window",
        )
        plot_errors(save_loc, df, sns.barplot, "Growth Rate Window", "Error", "Model")
        plot_errors(
            save_loc,
            df_freqdepend,
            sns.barplot,
            "Growth Rate Window",
            "Frequency Dependence Error",
            "Model",
        )
        plot_errors(save_loc, df, sns.barplot, "Growth Rate Window", "Error", None)
        plot_errors(
            save_loc,
            df_freqdepend,
            sns.barplot,
            "Growth Rate Window",
            "Frequency Dependence Error",
            None,
        )

        df = pd.melt(
            df,
            id_vars=["Model", "Experiment"],
            value_vars=get_parameter_names(),
            var_name="Parameter",
            value_name="Value",
        )
        df = df.reset_index(drop=True)
        plot_errors_facet(save_loc, df, sns.scatterplot, "Model", "Value", None, "Parameter")


if __name__ == "__main__":
    main()
