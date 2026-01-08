import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
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


def plot_fits(
    save_loc,
    exp_name,
    counts_df,
    growth_rate_df,
    cell_types,
    cell_colors,
    dc=0.0,
):
    # Only for 1 drug concentration
    counts_df_dc = counts_df[counts_df["DrugConcentration"] == dc]
    growth_rate_df_dc = growth_rate_df[growth_rate_df["DrugConcentration"] == dc]

    # Set figure dimensions
    plates = sorted(counts_df_dc["PlateId"].unique())
    plate_rows = 1
    plate_cols = len(plates)
    well_rows = sorted(counts_df_dc["RowId"].unique())
    well_cols = sorted(counts_df_dc["ColumnId"].unique())
    grid_rows = plate_rows * len(well_rows)
    grid_cols = plate_cols * len(well_cols)

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
    y_max = counts_df_dc["Count"].max()
    for p, plate in enumerate(plates):
        for r, row in enumerate(well_rows):
            for c, col in enumerate(well_cols):
                df = counts_df_dc[
                    (counts_df_dc["PlateId"] == plate)
                    & (counts_df_dc["RowId"] == row)
                    & (counts_df_dc["ColumnId"] == col)
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
                    gr = growth_rate_df_dc[
                        (growth_rate_df_dc["PlateId"] == plate)
                        & (growth_rate_df_dc["RowId"] == row)
                        & (growth_rate_df_dc["ColumnId"] == col)
                        & (growth_rate_df_dc["CellType"] == cell_type)
                    ].to_dict("records")[0]
                    gr_window = (gr["GrowthRate_window_start"], gr["GrowthRate_window_end"])
                    if not np.isnan(gr_window[0]):
                        x = np.arange(gr_window[0], gr_window[1], 0.1)
                        y = gr["GrowthRate"] * (x - gr_window[0]) + gr["Intercept"]
                        ax_curr.plot(x, np.exp(y), color="black", alpha=0.5, linewidth=3)
                ax_curr.set(title=f"Plate {plate} Well {row}{col}", ylim=(0, y_max))

    # Format figure and save
    fig.suptitle(f"{exp_name} {dc} Drug Concentration")
    plt.savefig(f"{save_loc}/assay_{exp_name}_{dc}dc.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_freqdepend_fit(save_loc, exp_name, gr_df, payoff_df, cell_colors, cell_types, dc=0.0):
    gr_df_dc = gr_df[gr_df["DrugConcentration"] == dc]
    payoff_df_dc = payoff_df[payoff_df["DrugConcentration"] == dc]

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.scatterplot(
        data=gr_df_dc,
        x=f"Fraction_{cell_types[0]}",
        y="GrowthRate",
        hue="CellType",
        marker="o",
        s=50,
        legend=False,
        ax=ax,
        palette=cell_colors,
    )
    for i, cell_type in enumerate(cell_types):
        x0 = "p12" if i == 0 else "p22"
        x1 = "p11" if i == 0 else "p21"
        ax.plot(
            [0, 1],
            [payoff_df_dc[x0].values[0], payoff_df_dc[x1].values[0]],
            ls="--",
            color=cell_colors[cell_type],
        )
    ax.set_title(f"{exp_name}\n{dc} Drug Concentration")
    fig.patch.set_alpha(0.0)
    plt.savefig(
        f"{save_loc}/assay_freqdepend_{exp_name}_{dc}dc.png", bbox_inches="tight", dpi=200
    )
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


def individual_analysis(data_dir, exp_name, dynamic_gr=True, rewrite=True):
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

    # Plot experiment visualizations
    plot_counts(save_loc, counts_df, cell_colors)
    plot_drug_concentration(save_loc, counts_df, cell_types)
    plot_seeded_fraction(save_loc, counts_df, cell_types)

    # Plot fits for no drug condition
    plot_fits(save_loc, exp_name, counts_df, growth_rate_df, cell_types, cell_colors)
    plot_freqdepend_fit(save_loc, exp_name, growth_rate_df, payoff_df, cell_colors, cell_types)

    # Plot spatial visualizations
    locations_df = calculate_locations(data_dir, exp_name, counts_df)
    times = sorted(counts_df["Time"].unique())[::2]
    plot_spatial(save_loc, data_dir, locations_df, cell_colors, times)


def main():
    # Read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str, default="data/experimental")
    parser.add_argument("-exp", "--exp_name", type=str, default=None)
    args = parser.parse_args()

    # Save data csvs and plot experiment analysis
    exp_names = [args.exp_name] if args.exp_name else os.listdir(args.data_dir)
    for exp_name in exp_names:
        if os.path.isfile(f"{args.data_dir}/{exp_name}") or exp_name == "layout_files":
            continue
        print(exp_name)
        individual_analysis(args.data_dir, exp_name)


if __name__ == "__main__":
    main()
