import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score

from compare_fits import (
    format_for_plotting,
    get_fit_df,
    label_qualitative_dynamics,
    plot_errors,
    plot_qualitative,
)
from fit_ode import fit
from game_assay.game_analysis import calculate_counts, calculate_growth_rates, calculate_payoffs
from game_assay.game_analysis_utils import (
    optimize_growth_rate_window_per_exp,
    optimize_growth_rate_window_per_well,
)
from run_game_assay import plot_fits, plot_freqdepend_fit
from utils import get_cell_types


def plot_agreement(data_dir, df):
    def heatmap(df_model_agr, model):
        df_model_agr = df_model_agr.pivot(
            index="Exponential Growth Window Strategy 1",
            columns="Exponential Growth Window Strategy 2",
            values="Agreement",
        )

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(df_model_agr, annot=True, vmin=0, vmax=1, ax=ax)
        ax.set_title(f"Agreement Between Exponential Growth Window Strategys for {model}")
        fig.patch.set_alpha(0.0)
        fig.tight_layout()
        fig.savefig(f"{data_dir}/agreement_{model}.png", bbox_inches="tight", dpi=200)
        plt.close()

    df_agr = []
    for model in df["Model"].unique():
        df_model = df[df["Model"] == model]
        df_q = df_model[
            ["Exponential Growth Window Strategy", "Experiment", "Dynamic"]
        ].drop_duplicates()
        df_q = df_q.sort_values(by=["Exponential Growth Window Strategy", "Experiment"])
        gr_windows = df_model["Exponential Growth Window Strategy"].unique()

        agreements = []
        for i in range(len(gr_windows)):
            for j in range(i + 1, len(gr_windows)):
                acc = accuracy_score(
                    df_q[df_q["Exponential Growth Window Strategy"] == gr_windows[i]]["Dynamic"],
                    df_q[df_q["Exponential Growth Window Strategy"] == gr_windows[j]]["Dynamic"],
                )
                agreements.append(
                    {
                        "Exponential Growth Window Strategy 1": gr_windows[i],
                        "Exponential Growth Window Strategy 2": gr_windows[j],
                        "Agreement": acc,
                    }
                )

        df_model_agr = pd.DataFrame(agreements)
        df_model_agr2 = df_model_agr.copy()
        df_model_agr2["temp"] = df_model_agr2["Exponential Growth Window Strategy 1"]
        df_model_agr2["Exponential Growth Window Strategy 1"] = df_model_agr2[
            "Exponential Growth Window Strategy 2"
        ]
        df_model_agr2["Exponential Growth Window Strategy 2"] = df_model_agr2["temp"]
        df_model_agr = pd.concat([df_model_agr, df_model_agr2])
        heatmap(df_model_agr, model)
        df_model_agr["Model"] = model
        df_agr.append(df_model_agr)

    df_agr = pd.concat(df_agr, ignore_index=True)
    df_agr = df_agr.sort_values(by="Model")
    df_agr["Exponential Growth Window Strategy"] = df_agr["Exponential Growth Window Strategy 1"]
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(df_agr, x="Exponential Growth Window Strategy", y="Agreement", hue="Model", ax=ax)
    ax.tick_params("x", rotation=45)
    fig.suptitle(
        "Agreement in Qualitative Interaction Classification\nbetween Window Strategies and Models"
    )
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{data_dir}/agreement_aggregated.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_parameter_ranges(data_dir, df, data_type):
    df_param = pd.melt(
        df,
        id_vars=["Model", "Exponential Growth Window Strategy", "Experiment"],
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
        hue="Exponential Growth Window Strategy",
        kind="box",
        sharex=False,
        sharey=False,
        height=4,
        aspect=1,
    )
    facet_grid.set_titles("{col_name}")
    facet_grid.figure.suptitle(f"Parameter Ranges of Models Fit on {data_type} Data")
    facet_grid.figure.patch.set_alpha(0.0)
    facet_grid.tight_layout()
    facet_grid.savefig(f"{data_dir}/parameter_range.png", bbox_inches="tight", dpi=200)
    plt.close()

    for window in df["Exponential Growth Window Strategy"].unique():
        facet_grid = sns.FacetGrid(
            df_param[df_param["Exponential Growth Window Strategy"] == window],
            col="Model",
            sharex=False,
            sharey=False,
            height=4,
            aspect=1,
        )
        facet_grid.map_dataframe(sns.boxplot, x="Parameter", y="Value")
        facet_grid.set_titles("{col_name}")
        facet_grid.figure.suptitle(
            f"Parameter Ranges of Models Fit on {data_type} Data under {window} Window Strategy"
        )
        facet_grid.figure.patch.set_alpha(0.0)
        facet_grid.tight_layout()
        facet_grid.savefig(f"{data_dir}/parameter_range_{window}.png", bbox_inches="tight", dpi=200)
        plt.close()


def save_parameter_ranges(data_dir, df):
    df_param = pd.melt(
        df,
        id_vars=["Model", "Exponential Growth Window Strategy", "Experiment"],
        value_vars=[x for x in df.columns if x[0] == "$"],
        var_name="Parameter",
        value_name="Value",
    )
    df_param = df_param.reset_index(drop=True).dropna().drop_duplicates()

    df_range = (
        df_param[["Exponential Growth Window Strategy", "Model", "Parameter", "Value"]]
        .groupby(["Exponential Growth Window Strategy", "Model", "Parameter"])
        .agg(["mean", "min", "max", "std"])
        .reset_index()
    )
    df_range.columns = df_range.columns.to_series().apply(lambda x: " ".join(x))
    df_range["low"] = df_range["Value mean"] - 2 * df_range["Value std"]
    df_range["upper"] = df_range["Value mean"] + 2 * df_range["Value std"]
    df_range.to_csv(f"{data_dir}/parameter_ranges.csv", index=False)


def plot_dynamics(save_loc, df, data_type):
    df_dynamic = (
        df[["Exponential Growth Window Strategy", "Model", "Experiment", "Dynamic"]]
        .drop_duplicates()
        .groupby(["Exponential Growth Window Strategy", "Model", "Dynamic"])
        .count()
        .reset_index()
    )
    facet_grid = sns.displot(
        df_dynamic,
        col="Model",
        x="Exponential Growth Window Strategy",
        weights="Experiment",
        hue="Dynamic",
        hue_order=sorted(df_dynamic["Dynamic"].unique()),
        kind="hist",
        multiple="stack",
        height=4,
        aspect=1,
    )
    facet_grid.add_legend()
    facet_grid.set_titles("{col_name}")
    facet_grid.set_xticklabels(rotation=45)
    facet_grid.figure.suptitle(f"{data_type} Data Classified Qualitative Interactions")
    sns.move_legend(facet_grid, loc="center right", bbox_to_anchor=(1.1, 0))
    facet_grid.figure.patch.set_alpha(0.0)
    facet_grid.tight_layout()
    facet_grid.savefig(f"{save_loc}/window_dynamics.png", bbox_inches="tight", dpi=200)
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
        df_window["Growth Rate Window"] = gr_window
        df.append(df_window)
    df = pd.concat(df)

    # Final formatting and return
    df = get_ground_truth(in_dir, df)
    df = label_qualitative_dynamics(df, ["Growth Rate Window", "Model", "Experiment"])
    df = format_for_plotting(df)
    return df.reset_index()


def save_growth_rate(in_data_dir, out_data_dir, exp_name, window):
    save_loc = f"{out_data_dir}/{window}/{exp_name}/images"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    sensitive_type, resistant_type = get_cell_types(exp_name)
    cell_types = [sensitive_type, resistant_type]
    counts_df = calculate_counts(in_data_dir, exp_name)

    gr_window = None
    if window == "none":
        gr_window = (counts_df["Time"].min(), counts_df["Time"].max())
    elif window == "per_well":
        counts_df = optimize_growth_rate_window_per_well(counts_df)
    elif window == "per_exp":
        counts_df = optimize_growth_rate_window_per_exp(counts_df)

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
    parser.add_argument(
        "-w", "--window", type=str, choices=["none", "per_exp", "per_well", "per_cell"]
    )
    parser.add_argument("-plot", "--plot", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()
    data_type = args.in_dir.title()

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
        df = df[
            (df["Exponential Growth Window Strategy"] != "Ground Truth")
            & (df["Model"] != "Ground Truth")
        ]

        plot_dynamics(save_loc, df, data_type)

        for model in df["Model"].unique():
            plot_qualitative(
                save_loc,
                df[df["Model"] == model],
                focal_col="Exponential Growth Window Strategy",
                sub_col=model,
            )
        plot_agreement(save_loc, df)

        plot_errors(
            save_loc, df, sns.barplot, "Exponential Growth Window Strategy", "Error", "Model"
        )
        plot_errors(save_loc, df, sns.barplot, "Exponential Growth Window Strategy", "Error", None)
        plot_errors(save_loc, df, sns.barplot, "Exponential Growth Window Strategy", "BIC", "Model")
        plot_errors(save_loc, df, sns.barplot, "Exponential Growth Window Strategy", "BIC", None)

        plot_parameter_ranges(save_loc, df, data_type)
        save_parameter_ranges(save_loc, df)


if __name__ == "__main__":
    main()
