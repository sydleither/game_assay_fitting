import argparse
import os

import pandas as pd
from scipy.stats import entropy
import seaborn as sns

from utils import (
    format_for_plotting,
    get_fit_df,
    label_data_type,
    label_qualitative_dynamics,
)
from fit_ode import fit
from game_assay.game_analysis import calculate_counts, calculate_growth_rates, calculate_payoffs
from game_assay.game_analysis_utils import (
    optimize_growth_rate_window_per_exp,
    optimize_growth_rate_window_per_well,
)
from run_game_assay import plot_fits, plot_freqdepend_fit
from utils import get_cell_types


def plot_accuracy(save_loc, df):
    facet = sns.catplot(
        data=df[df["Data Type"] != "Experimental"],
        kind="bar",
        col="Model",
        x="Data Type",
        y="Accuracy",
        hue="Exponential Growth Window Strategy",
    )
    facet.figure.patch.set_alpha(0.0)
    facet.tight_layout()
    facet.savefig(f"{save_loc}/accuracy_window.png", bbox_inches="tight", dpi=200)


def plot_entropy(save_loc, df):
    facet = sns.catplot(
        data=df[df["Data Type"] == "Experimental"],
        kind="bar",
        col="Model",
        x="Data Type",
        y="Entropy",
        hue="Exponential Growth Window Strategy",
    )
    facet.figure.patch.set_alpha(0.0)
    facet.tight_layout()
    facet.savefig(f"{save_loc}/entropy_window.png", bbox_inches="tight", dpi=200)


def get_ground_truth(in_data_dir, df):
    if not os.path.exists(f"{in_data_dir}/ground_truth.csv"):
        entropies = (
            df[["Experiment", "Model", "Dynamic"]]
            .groupby("Model")["Dynamic"]
            .value_counts(normalize=True)
            .groupby(level=0)
            .apply(entropy)
        )
        df = df.merge(entropies, on="Model")
        df = df.rename({"proportion": "Entropy"}, axis=1)
        return df
    df_gt = pd.read_csv(f"{in_data_dir}/ground_truth.csv")
    df_gt = label_qualitative_dynamics(df_gt, ["Experiment"])
    df_gt = df.merge(df_gt, on="Experiment", suffixes=("", " True"))
    df_gt["Correct"] = df_gt["Dynamic"] == df_gt["Dynamic True"]
    df_gt["Accuracy"] = df_gt.groupby("Model")["Correct"].transform("mean")
    return df_gt


def get_growth_rates():
    # Read in all the data files
    df = []
    for data_dir in os.listdir("data"):
        if not data_dir.endswith("gr"):
            continue
        for window in os.listdir(f"data/{data_dir}"):
            if os.path.isfile(f"data/{data_dir}/{window}"):
                continue
            for rep in os.listdir(f"data/{data_dir}/{window}"):
                if os.path.isfile(f"data/{data_dir}/{window}/{rep}"):
                    continue
                df_exp = get_fit_df(f"data/{data_dir}/{window}/{rep}")
                df_exp = df_exp.drop_duplicates(subset=["Model", "Experiment"])
                df_exp = label_qualitative_dynamics(df_exp, ["Model", "Experiment"])
                df_exp = get_ground_truth(f"data/{data_dir[:-3]}/{rep}", df_exp)
                df_exp["Replicate"] = rep
                df_exp["Growth Rate Window"] = window
                df_exp["Data Type"] = label_data_type(data_dir[:-3])
                df.append(df_exp)
    df = pd.concat(df)
    df = format_for_plotting(df)
    return df.reset_index()


def save_growth_rate(in_dir, out_dir, replicate, exp_name, window):
    # Set variables
    save_loc = f"{out_dir}/{window}/{replicate}/{exp_name}/images"
    sensitive_type, resistant_type = get_cell_types(exp_name)
    cell_types = [sensitive_type, resistant_type]
    counts_df = calculate_counts(f"{in_dir}/{replicate}", exp_name)

    # Set exponential growth window
    gr_window = None
    if window == "none":
        gr_window = (counts_df["Time"].min(), counts_df["Time"].max())
    elif window == "per_well":
        counts_df = optimize_growth_rate_window_per_well(counts_df)
    elif window == "per_exp":
        counts_df = optimize_growth_rate_window_per_exp(counts_df)

    # Calculate growth rates and plot fits
    growth_rate_df = calculate_growth_rates(
        f"{out_dir}/{window}/{replicate}",
        exp_name,
        counts_df,
        growth_rate_window=gr_window,
        cell_type_list=cell_types,
    )
    cell_colors = {sensitive_type: "#4C956C", resistant_type: "#EF7C8E"}
    plot_fits(save_loc, exp_name, counts_df, growth_rate_df, cell_types, cell_colors)

    # Plot frequency-dependent fits
    payoff_df = calculate_payoffs(
        f"{out_dir}/{window}/{replicate}",
        exp_name,
        growth_rate_df,
        cell_types,
        f"Fraction_{sensitive_type}",
    )
    plot_freqdepend_fit(save_loc, exp_name, growth_rate_df, payoff_df, cell_colors, cell_types)

    # Fit ODEs
    fit(
        f"{out_dir}/{window}/{replicate}",
        exp_name,
        "replicator",
        counts_df,
        growth_rate_df,
        save_figs=True,
    )
    fit(
        f"{out_dir}/{window}/{replicate}",
        exp_name,
        "lotka-volterra",
        counts_df,
        growth_rate_df,
        save_figs=True,
    )


def write_run_script(in_dir, out_dir, run_cmd):
    os.makedirs(out_dir)
    with open(f"{out_dir}/exponential_growth_windows.sh", "w") as f:
        for window in ["none", "per_exp", "per_well", "per_cell"]:
            for rep in os.listdir(in_dir):
                if os.path.isfile(f"{in_dir}/{rep}"):
                    continue
                for exp in os.listdir(f"{in_dir}/{rep}")[::2]:
                    if os.path.isfile(f"{in_dir}/{rep}/{exp}") or exp == "layout_files":
                        continue
                    os.makedirs(f"{out_dir}/{window}/{rep}/{exp}/images")
                    arg_str = f"-in {in_dir} -rep {rep} -exp {exp} -w {window}"
                    f.write(f"{run_cmd} exponential_growth_window.py {arg_str}\n")


def main():
    # Read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--in_dir", type=str)
    parser.add_argument("-run", "--run_cmd", type=str, default="python3")
    parser.add_argument("-rep", "--replicate", type=int, default=None)
    parser.add_argument("-exp", "--exp_name", type=str, default=None)
    parser.add_argument(
        "-w", "--window", type=str, choices=["none", "per_exp", "per_well", "per_cell"]
    )
    parser.add_argument("-plot", "--plot", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()
    in_dir = args.in_dir[:-1] if args.in_dir[-1] == "/" else args.in_dir
    out_dir = f"{in_dir}_gr"

    if args.exp_name is None and args.plot == 0:
        write_run_script(in_dir, out_dir, args.run_cmd)
        return

    if args.plot == 0:
        save_growth_rate(in_dir, out_dir, args.replicate, args.exp_name, args.window)

    if args.plot == 1:
        df = get_growth_rates()
        plot_accuracy("data", df)
        plot_entropy("data", df)


if __name__ == "__main__":
    main()
