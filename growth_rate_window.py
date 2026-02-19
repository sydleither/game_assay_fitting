import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import comb
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

from compare_fits import (
    get_fit_df,
    label_qualitative_dynamics,
    plot_errors
)
from fit_ode import fit
from game_assay.game_analysis import (
    calculate_counts,
    calculate_growth_rates,
    calculate_payoffs,
)
from game_assay.game_analysis_utils import optimize_growth_rate_window, optimize_growth_rate_window2
from utils import abm_parameter_map, get_cell_types, get_growth_rate_window


def plot_qualitative(data_dir, df, model):
    df_model = df[(df["Model"] == model) | (df["Model"] == "Ground Truth")]
    df_q = df_model[["Growth Rate Window", "Experiment", "Dynamic"]].drop_duplicates()
    gr_windows = df_model["Growth Rate Window"].unique()
    num_axes = comb(len(gr_windows), 2, exact=True)

    accuracies = []
    fig, ax = plt.subplots(1, num_axes, figsize=(4 * num_axes, 4))
    ax_num = 0
    for i in range(len(gr_windows)):
        for j in range(i + 1, len(gr_windows)):
            labels = sorted(
                df_q[df_q["Growth Rate Window"].isin([gr_windows[i], gr_windows[j]])][
                    "Dynamic"
                ].unique()
            )
            conf_mat = confusion_matrix(
                df_q[df_q["Growth Rate Window"] == gr_windows[i]]["Dynamic"],
                df_q[df_q["Growth Rate Window"] == gr_windows[j]]["Dynamic"],
                labels=labels,
            )
            acc = accuracy_score(
                df_q[df_q["Growth Rate Window"] == gr_windows[i]]["Dynamic"],
                df_q[df_q["Growth Rate Window"] == gr_windows[j]]["Dynamic"],
            )
            df_conf_mat = pd.DataFrame(conf_mat, columns=labels, index=labels)
            df_conf_mat.columns.name = gr_windows[j]
            df_conf_mat.index.name = gr_windows[i]
            sns.heatmap(df_conf_mat, annot=True, ax=ax[ax_num])
            ax[ax_num].set(title=f"{gr_windows[i]} vs {gr_windows[j]}\nAgreement: {acc:5.3f}")
            ax_num += 1
            if gr_windows[i] == "Ground Truth":
                accuracies.append({"Growth Rate Window": gr_windows[j], "Accuracy": acc})
            elif gr_windows[j] == "Ground Truth":
                accuracies.append({"Growth Rate Window": gr_windows[i], "Accuracy": acc})
    fig.suptitle("Confusion Matrices for Growth Rate Window Dynamics")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{data_dir}/conf_mat_{model}.png", bbox_inches="tight", dpi=200)
    plt.close()

    if len(accuracies) > 0:
        df_acc = pd.DataFrame(accuracies)
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.barplot(df_acc, x="Growth Rate Window", y="Accuracy", ax=ax)
        fig.patch.set_alpha(0.0)
        fig.tight_layout()
        fig.savefig(f"{data_dir}/accuracy_{model}.png", bbox_inches="tight", dpi=200)
        plt.close()


def get_ground_truth(in_data_dir, df):
    if not os.path.exists(f"{in_data_dir}/ground_truth.csv"):
        return df
    df_gt = pd.read_csv(f"{in_data_dir}/ground_truth.csv")
    df_gt = df_gt.rename(columns=abm_parameter_map())
    params = [x for x in abm_parameter_map().values() if x in df.columns]
    df_gt = df_gt[["Experiment"] + params]
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
    df["Error"] = df["Error"] / df["Window Size"]

    # Final formatting and return
    df = get_ground_truth(f"{data_dir}/{in_dir}", df)
    df = label_qualitative_dynamics(df, ["Growth Rate Window", "Model", "Experiment"])
    df.loc[df["Growth Rate Window"] == "Extra_Dynamic", "Growth Rate Window"] = "Dynamic2"
    return df.reset_index()


def save_growth_rate(in_data_dir, out_data_dir, exp_name, window):
    save_loc = f"{out_data_dir}/{window}/{exp_name}"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    sensitive_type, resistant_type = get_cell_types(exp_name)
    cell_types = [sensitive_type, resistant_type]
    counts_df = calculate_counts(in_data_dir, exp_name)

    if window == "none":
        counts_df["GrowthRate_window_start"] = counts_df["Time"].min()
        counts_df["GrowthRate_window_end"] = counts_df["Time"].max()
    elif window == "dynamic":
        counts_df = optimize_growth_rate_window(counts_df)
    elif window == "extra_dynamic":
        counts_df = optimize_growth_rate_window2(counts_df)
    elif window == "expert":
        gr_window = get_growth_rate_window(in_data_dir, exp_name)
        counts_df["GrowthRate_window_start"] = gr_window[0]
        counts_df["GrowthRate_window_end"] = gr_window[1]
    elif window == "early":
        counts_df["GrowthRate_window_start"] = 0
        counts_df["GrowthRate_window_end"] = 40
    else:
        raise ValueError("Illegal growth rate window option supplied.")

    growth_rate_df = calculate_growth_rates(
        f"{out_data_dir}/{window}",
        exp_name,
        counts_df,
        cell_type_list=cell_types,
    )

    calculate_payoffs(
        f"{out_data_dir}/{window}",
        exp_name,
        growth_rate_df,
        cell_types,
        f"Fraction_{sensitive_type}",
    )
    fit(
        f"{out_data_dir}/{window}",
        exp_name,
        "replicator",
        counts_df,
        growth_rate_df,
        save_figs=False,
    )
    fit(f"{out_data_dir}/{window}", exp_name, "lv", counts_df, growth_rate_df, save_figs=False)


def main():
    # Read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str, default="data")
    parser.add_argument("-in", "--in_dir", type=str, default="experimental")
    parser.add_argument("-out", "--out_dir", type=str, default="gr_experimental")
    parser.add_argument(
        "-w",
        "--window",
        type=str,
        choices=["none", "dynamic", "early", "extra_dynamic", "expert"]
    )
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
        df = get_growth_rates(save_loc, args.in_dir)
        df_freqdepend = df[df["Model"] != "Lotka-Volterra"][
            ["Growth Rate Window", "Frequency Dependence Error", "Model"]
        ].drop_duplicates()
        df_freqdepend.to_csv("temp.csv")

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

        plot_qualitative(save_loc, df, "Game Assay")
        plot_qualitative(save_loc, df, "Replicator")
        plot_qualitative(save_loc, df, "Lotka-Volterra")


if __name__ == "__main__":
    main()
