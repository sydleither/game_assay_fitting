import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
import seaborn as sns

from game_assay.game_archive_utils import load_spatial_data
from game_assay.game_analysis import (
    calculate_counts,
    calculate_growth_rates,
    calculate_locations,
    calculate_payoffs,
)
from utils import get_cell_types, get_growth_rate_window


def plot_counts(save_loc, df, cell_colors):
    for plate_id in df["PlateId"].unique():
        df_plate = df[df["PlateId"] == plate_id]
        well_letters = sorted(df_plate["WellId"].str[0].unique())
        well_nums = sorted(df_plate["WellId"].str[1:].astype(int).unique())
        num_letters = len(well_letters)
        num_nums = len(well_nums)
        fig, ax = plt.subplots(
            num_letters, num_nums, figsize=(2 * num_nums, 2 * num_letters), sharex=True, sharey=True
        )
        for wl in range(len(well_letters)):
            for wn in range(len(well_nums)):
                well = well_letters[wl] + str(well_nums[wn])
                sns.lineplot(
                    data=df_plate[df_plate["WellId"] == well],
                    x="Time",
                    y="Count",
                    hue="CellType",
                    marker="o",
                    legend=False,
                    ax=ax[wl][wn],
                    palette=cell_colors,
                )
                ax[wl][wn].set(title=well)
        fig.patch.set_alpha(0.0)
        fig.tight_layout()
        plt.savefig(f"{save_loc}/plate{plate_id}_counts.png")
        plt.close()


def plot_drug_concentration(save_loc, df, cell_types):
    df = df[(df["Time"] == 0) & (df["CellType"] == cell_types[0])]
    for plate_id in df["PlateId"].unique():
        df_plate = df[df["PlateId"] == plate_id].reset_index()
        df_plate = df_plate.pivot(index="RowId", columns="ColumnId", values="DrugConcentration")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sns.heatmap(df_plate, annot=True, cmap="Purples", ax=ax)
        ax.set_title(f"Plate {plate_id} Drug Concentration")
        fig.patch.set_alpha(0.0)
        fig.tight_layout()
        plt.savefig(f"{save_loc}/plate{plate_id}_dc.png")
        plt.close()


def plot_seeded_fraction(save_loc, df, cell_types):
    df = df[(df["Time"] == 0) & (df["CellType"] == cell_types[0])]
    for plate_id in df["PlateId"].unique():
        df_plate = df[df["PlateId"] == plate_id].reset_index()
        df_plate = df_plate.pivot(index="RowId", columns="ColumnId", values="Frequency")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sns.heatmap(df_plate, annot=True, cmap="Greens", ax=ax)
        ax.set_title(f"Plate {plate_id} {cell_types[0]} Frequency")
        fig.patch.set_alpha(0.0)
        fig.tight_layout()
        plt.savefig(f"{save_loc}/plate{plate_id}_seeded.png")
        plt.close()


def plot_freq_depend(save_loc, df, cell_types, cell_colors, extra=""):
    concentrations = sorted(df["DrugConcentration"].unique())
    fig, axes = plt.subplots(
        1, len(concentrations), figsize=(4 * len(concentrations), 4), sharey=False
    )
    if len(concentrations) == 1:
        axes = [axes]
    axis_cell = cell_types[0]
    for ax, conc in zip(axes, concentrations):
        df_sub = df[df["DrugConcentration"] == conc]
        for cell_type in cell_types:
            df_ct = df_sub[df_sub["CellType"] == cell_type].copy()
            df_ct = df_ct.dropna(subset=[f"Fraction_{axis_cell}", "GrowthRate"])
            if len(df_ct) < 2:
                continue
            x = df_ct[f"Fraction_{axis_cell}"].values
            y = df_ct["GrowthRate"].values
            slope, intercept, _, _ = theilslopes(y, x)
            ax.scatter(x, y, color=cell_colors[cell_type], alpha=0.7)
            x_fit = np.linspace(0, 1, 100)
            y_fit = intercept + slope * x_fit
            ax.plot(x_fit, y_fit, color=cell_colors[cell_type], linestyle="--")
        ax.set_title(f"Drug Concentration: {conc}")
        ax.set_xlabel(f"Fraction {axis_cell}")
        ax.set_ylabel("Growth Rate")
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    plt.savefig(f"{save_loc}/freq_depend{extra}.png", dpi=200)
    plt.close()


def plot_overview(
    save_loc,
    exp_name,
    counts_df,
    growth_rate_df,
    payoff_df,
    cell_types,
    cell_colors,
    dc=0.0,
):
    # Only for 1 drug concentration
    counts_df = counts_df[counts_df["DrugConcentration"] == dc]
    growth_rate_df = growth_rate_df[growth_rate_df["DrugConcentration"] == dc]
    payoff_df = payoff_df[payoff_df["DrugConcentration"] == dc]

    # Set figure dimensions
    plates = sorted(counts_df["PlateId"].unique())
    plate_rows = 1
    plate_cols = len(plates)
    well_rows = sorted(counts_df["RowId"].unique())
    well_cols = sorted(counts_df["ColumnId"].unique())
    grid_rows = plate_rows * len(well_rows)
    grid_cols = plate_cols * len(well_cols) + 1

    # Initalize figure
    fig, ax = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(3 * grid_cols, 3 * grid_rows),
        sharex=False,
        sharey=False,
        layout="compressed",
    )

    # Plot count data with growth rate overlaid
    y_max = counts_df["Count"].max()
    for p, plate in enumerate(plates):
        for r, row in enumerate(well_rows):
            for c, col in enumerate(well_cols):
                df = counts_df[
                    (counts_df["PlateId"] == plate)
                    & (counts_df["RowId"] == row)
                    & (counts_df["ColumnId"] == col)
                ]
                ax_curr = ax[r, p * len(well_cols) + c]
                sns.scatterplot(
                    data=df,
                    x="Time",
                    y="Count",
                    hue="CellType",
                    marker="o",
                    s=50,
                    legend=False,
                    ax=ax_curr,
                    palette=cell_colors,
                )
                for cell_type in cell_types:
                    gr = growth_rate_df[
                        (growth_rate_df["PlateId"] == plate)
                        & (growth_rate_df["RowId"] == row)
                        & (growth_rate_df["ColumnId"] == col)
                        & (growth_rate_df["CellType"] == cell_type)
                    ].to_dict("records")[0]
                    gr_window = (gr["GrowthRate_window_start"], gr["GrowthRate_window_end"])
                    if not np.isnan(gr_window[0]):
                        x = np.arange(gr_window[0], gr_window[1], 0.1)
                        y = gr["GrowthRate"] * (x - gr_window[0]) + gr["Intercept"]
                        ax_curr.plot(x, np.exp(y), color="black", alpha=0.5, linewidth=3)
                ax_curr.set(title=f"Plate {plate} Well {row}{col}", ylim=(0, y_max))

    # Plot frequency dependence dynamics
    payoff = payoff_df.to_dict("records")[0]
    ax_curr = ax[0, grid_cols - 1]
    for i, cell_type in enumerate(cell_types):
        df = growth_rate_df[growth_rate_df["CellType"] == cell_type].copy()
        df = df.dropna(subset=[f"Fraction_{cell_types[0]}", "GrowthRate"])
        if len(df) < 2:
            continue
        x = df[f"Fraction_{cell_types[0]}"].values
        y = df["GrowthRate"].values
        ax_curr.scatter(x, y, color=cell_colors[cell_type], alpha=0.75)
        if i == 0:
            y_0 = payoff["r1"] + payoff["c21"]
            y_1 = payoff["r1"] + payoff["c21"] - payoff["c21"]
        else:
            y_0 = payoff["r2"]
            y_1 = payoff["r2"] + payoff["c12"]
        ax_curr.plot([0, 1], [y_0, y_1], color=cell_colors[cell_type], linestyle="--")
        ax_curr.set(title="Frequency Dependence")
        ax_curr.set(xlabel=f"Fraction {cell_types[0]}", ylabel="Growth Rate")

    # Plot location in game space
    ax_curr = ax[1, grid_cols - 1]
    ax_curr.scatter(payoff["Advantage_1"], payoff["Advantage_0"], s=50, c="black")
    ax_curr.axvline(0, color="black")
    ax_curr.axhline(0, color="black")
    ax_curr.set(title="Game Space")
    ax_curr.set(xlabel=f"Advantage {cell_types[1]}", ylabel=f"Advantage {cell_types[0]}")

    # Hide empty plots
    for i in range(2, grid_rows):
        ax[i, grid_cols - 1].axis("off")

    # Format figure and save
    fig.suptitle(f"{exp_name} {dc} Drug Concentration")
    plt.savefig(f"{save_loc}/{exp_name}_{dc}dc.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_spatial(save_loc, data_dir, df, cell_colors, times):
    for plate_id in df["PlateId"].unique():
        df_plate = df[df["PlateId"] == plate_id]
        well_ids = sorted(df_plate["WellId"].unique(), key=lambda x: (x[0], int(x[1:])))
        fig, ax = plt.subplots(
            len(well_ids),
            len(times),
            figsize=(2 * len(times), 2 * len(well_ids)),
            sharex=True,
            sharey=True,
        )
        for w, well in enumerate(well_ids):
            df_spatial = load_spatial_data(df_plate[(df_plate["WellId"] == well)], data_dir)
            for t, time in enumerate(times):
                sns.scatterplot(
                    data=df_spatial[df_spatial["Time_hours"] == time],
                    x="Location_Center_X",
                    y="Location_Center_Y",
                    hue="CellType",
                    marker=".",
                    s=5,
                    legend=False,
                    ax=ax[w][t],
                    palette=cell_colors,
                    edgecolor="none",
                )
                ax[w][t].set(title=f"Well {well} Time = {time}")
        fig.patch.set_alpha(0.0)
        fig.tight_layout()
        plt.savefig(f"{save_loc}/plate{plate_id}_spatial.png", dpi=200)
        plt.close()


def plot_gamespace(save_loc, df, hue):
    if hue == "fit":
        palette = sns.color_palette("mako", len(df[hue].unique()))
    else:
        palette = sns.color_palette("hls", len(df[hue].unique()))
    hue_order = sorted(df[hue].unique())
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.scatterplot(
        data=df,
        x="Advantage_1",
        y="Advantage_0",
        s=200,
        hue=hue,
        palette=palette,
        hue_order=hue_order,
        ax=ax,
    )
    ax.axvline(0, color="black")
    ax.axhline(0, color="black")
    ax.legend(bbox_to_anchor=(1.01, 1))
    fig.patch.set_alpha(0.0)
    plt.savefig(f"{save_loc}/gamespace_{hue}.png", bbox_inches="tight", dpi=200)


def individual_analysis(data_dir, exp_name, dynamic_gr=True, rewrite=False):
    # Create images directory
    save_loc = f"{data_dir}/{exp_name}/images"
    if not os.path.exists(save_loc):
        os.mkdir(save_loc)

    if dynamic_gr:
        gr_window = None
    else:
        gr_window = get_growth_rate_window(data_dir, exp_name)

    counts_df = calculate_counts(data_dir, exp_name, rewrite)
    sensitive_type, resistant_type = get_cell_types(exp_name)
    cell_types = [sensitive_type, resistant_type]
    cell_colors = {sensitive_type: "#4C956C", resistant_type: "#EF7C8E"}
    growth_rate_df = calculate_growth_rates(
        data_dir, exp_name, counts_df, gr_window, cell_types, rewrite
    )
    payoff_df = calculate_payoffs(
        data_dir, exp_name, growth_rate_df, cell_types, f"Fraction_{sensitive_type}", rewrite
    )

    plot_counts(save_loc, counts_df, cell_colors)
    plot_drug_concentration(save_loc, counts_df, cell_types)
    plot_seeded_fraction(save_loc, counts_df, cell_types)
    plot_freq_depend(save_loc, growth_rate_df, cell_types, cell_colors)
    plot_freq_depend(
        save_loc,
        growth_rate_df[growth_rate_df["DrugConcentration"] == 0],
        cell_types,
        cell_colors,
        "_dc0",
    )

    # Plot overview of drug-free results
    plot_overview(
        save_loc,
        exp_name,
        counts_df,
        growth_rate_df,
        payoff_df,
        cell_types,
        cell_colors,
    )

    # Plot spatial visualizations
    locations_df = calculate_locations(data_dir, exp_name, counts_df)
    times = sorted(counts_df["Time"].unique())[::2]
    plot_spatial(save_loc, data_dir, locations_df, cell_colors, times)


def replicate_analysis(data_dir, dynamic_gr=True, rewrite=True):
    # Get payoff information of each experiment
    gr_window = None
    payoff_df = pd.DataFrame()
    for exp_name in os.listdir(data_dir):
        if "PIK3CA" in exp_name:  # TEMP
            continue
        if os.path.isfile(f"{data_dir}/{exp_name}") or exp_name == "layout_files":
            continue
        if not dynamic_gr:
            gr_window = get_growth_rate_window(data_dir, exp_name)
        sensitive_type, resistant_type = get_cell_types(exp_name)
        cell_types = [sensitive_type, resistant_type]
        counts_df_exp = calculate_counts(data_dir, exp_name, rewrite)
        growth_rate_df_exp = calculate_growth_rates(
            data_dir, exp_name, counts_df_exp, gr_window, cell_types, rewrite
        )
        payoff_df_exp = calculate_payoffs(
            data_dir,
            exp_name,
            growth_rate_df_exp,
            cell_types,
            f"Fraction_{sensitive_type}",
            rewrite,
        )
        payoff_df_exp["Experiment"] = exp_name
        payoff_df_exp["Mean Growth Rate"] = np.exp(np.mean(growth_rate_df_exp["Intercept"]))
        payoff_df_exp["Mean Fit"] = np.mean(growth_rate_df_exp["GrowthRate_fit"])
        payoff_df = pd.concat([payoff_df, payoff_df_exp])
    payoff_df = payoff_df[payoff_df["DrugConcentration"] == 0.0]

    # Validate S-E9-gfp is always player 1
    type1s = payoff_df["Type1"].unique()
    if len(type1s) > 1:
        raise ValueError(f"Inconsistent payoff matrices: type 1s of {type1s}")

    # Plot game spaces
    plot_gamespace(data_dir, payoff_df, "Experiment")
    plot_gamespace(data_dir, payoff_df, "Type2")
    plot_gamespace(data_dir, payoff_df, "fit")


def main():
    # Read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str, default="data/experimental")
    parser.add_argument("-exp", "--exp_name", type=str, default=None)
    args = parser.parse_args()

    # Run specified analysis type
    if args.exp_name:
        individual_analysis(args.data_dir, args.exp_name)
    else:
        replicate_analysis(args.data_dir)


if __name__ == "__main__":
    main()
