import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import theilslopes
import seaborn as sns

from game_assay.game_analysis import count_cells, calculate_growth_rates, calculate_payoffs
from utils import get_cell_types


def plot_counts(save_loc, df, cell_colors, extra=""):
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
        plt.savefig(f"{save_loc}/plate{plate_id}_counts{extra}.png")
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
            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = intercept + slope * x_fit
            ax.plot(x_fit, y_fit, color=cell_colors[cell_type], linestyle="--")
        ax.set_title(f"Drug Concentration: {conc}")
        ax.set_xlabel(f"Fraction {axis_cell}")
        ax.set_ylabel("Growth Rate")
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    plt.savefig(f"{save_loc}/freq_depend{extra}.png", dpi=200)
    plt.close()


def main():
    # Read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str, default="data/experimental")
    parser.add_argument(
        "-exp", "--exp_name", type=str, default="220405_S-E9_gfp_vs_BRAF_mcherry_Gefitinib"
    )
    #TODO automatic GR window or use overview.xlsx's
    parser.add_argument("-grs", "--growth_rate_start", type=int, default=24)
    parser.add_argument("-gre", "--growth_rate_end", type=int, default=72)
    args = parser.parse_args()

    # Set data io parameters
    data_dir = args.data_dir
    exp_name = args.exp_name
    save_loc = f"{data_dir}/{exp_name}/images"
    if not os.path.exists(save_loc):
        os.mkdir(save_loc)
    gr_window = [args.growth_rate_start, args.growth_rate_end]

    # Count cells
    counts_df = count_cells(data_dir, exp_name)
    sensitive_type, resistant_type = get_cell_types(exp_name)
    cell_types = [sensitive_type, resistant_type]
    cell_colors = {sensitive_type: "#4C956C", resistant_type: "#EF7C8E"}
    plot_counts(save_loc, counts_df, cell_colors)
    plot_counts(
        save_loc,
        counts_df[(counts_df["Time"] >= gr_window[0]) & (counts_df["Time"] <= gr_window[1])],
        cell_colors,
        "_gr",
    )
    plot_drug_concentration(save_loc, counts_df, cell_types)
    plot_seeded_fraction(save_loc, counts_df, cell_types)

    # Calculate growth rate
    growth_rate_df = calculate_growth_rates(data_dir, exp_name, counts_df, gr_window, cell_types)
    plot_freq_depend(save_loc, growth_rate_df, cell_types, cell_colors)
    plot_freq_depend(
        save_loc,
        growth_rate_df[growth_rate_df["DrugConcentration"] == 0],
        cell_types,
        cell_colors,
        "_dc0",
    )

    # Calculate payoff matrix parameters
    payoff_df = calculate_payoffs(data_dir, exp_name, growth_rate_df, cell_types, f"Fraction_{cell_types[0]}")


if __name__ == "__main__":
    main()
