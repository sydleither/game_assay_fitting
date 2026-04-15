import argparse
import os

from lmfit import minimize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import seaborn as sns

from fitting.fittingUtils import residual_multipleConditions
from fitting.myUtils import ExtractTreatmentFromDf
from fitting.odeModels import create_model, get_models
from game_assay.game_analysis import calculate_counts, calculate_growth_rates
from game_assay.game_analysis_utils import calculate_bic
from utils import get_cell_types, optimiser_kws, solver_kws


def plot_fits(save_loc, exp_name, model_name, df, df_model, cell_colors, dc):
    # Set figure dimensions
    plates = sorted(df["PlateId"].unique())
    plate_rows = 1
    plate_cols = len(plates)
    well_rows = sorted(df["RowId"].unique())
    well_cols = sorted(df["ColumnId"].unique())
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
    y_max = df["Count"].max()
    for p, plate in enumerate(plates):
        for r, row in enumerate(well_rows):
            for c, col in enumerate(well_cols):
                # Plot measured growth rates
                df1 = df[(df["PlateId"] == plate) & (df["RowId"] == row) & (df["ColumnId"] == col)]
                ax_curr = ax[r, p * len(well_cols) + c]
                sns.scatterplot(
                    data=df1,
                    x="Time",
                    y="Count",
                    hue="CellType",
                    marker="o",
                    s=50,
                    legend=False,
                    ax=ax_curr,
                    palette=cell_colors,
                )
                ax_curr.set(title=f"Plate {plate} Well {row}{col}", ylim=(0, y_max))
                # Plot fitted line
                df1_model = df_model[
                    (df_model["PlateId"] == plate) & (df_model["WellId"] == f"{row}{col}")
                ]
                if len(df1_model) == 0:
                    continue
                sns.lineplot(
                    data=df1_model,
                    x="Time",
                    y="Count",
                    hue="CellType",
                    linewidth=3,
                    legend=False,
                    ax=ax_curr,
                    palette=["black", "black"],
                    alpha=0.5,
                )

    # Format figure and save
    fig.suptitle(f"{exp_name} {dc} Drug Concentration")
    fig.patch.set_alpha(0.0)
    plt.savefig(f"{save_loc}/{model_name}_{exp_name}_{dc}dc.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_freqdepend(save_loc, exp_name, model_name, models_df, cell_colors, cell_types, dc):
    df = models_df.copy().drop(["Time", "Count"], axis=1).drop_duplicates().reset_index()
    fig, ax = plt.subplots(figsize=(4, 4))
    for i, cell_type in enumerate(cell_types):
        x0 = "p_SR" if i == 0 else "p_RR"
        x1 = "p_SS" if i == 0 else "p_RS"
        ax.plot(
            [0, 1],
            [df[x0].values[0], df[x1].values[0]],
            ls="--",
            color=cell_colors[cell_type],
        )
    ax.set_ylim(df["GrowthRate"].min(), df["GrowthRate"].max())
    ax.set_title(f"{exp_name}\n{dc} Drug Concentration")
    fig.patch.set_alpha(0.0)
    plt.savefig(
        f"{save_loc}/{model_name}_freqdepend_{exp_name}_{dc}dc.png", bbox_inches="tight", dpi=200
    )
    plt.close()


def get_fit_error(df_rep, df_model, cell_type, k):
    df_model_ct = df_model[df_model["CellType"] == cell_type]
    error = np.sum(np.square(df_rep[cell_type].values - df_model_ct["Count"].values))
    bic = calculate_bic(df_model_ct, error, k)
    return error, bic


def fit(
    data_dir,
    exp_name,
    model,
    counts_df,
    growth_rate_df,
    drug_concentration=0.0,
    trim=True,
    save_figs=True,
):
    # Combine count and growth rate dataframes
    float_cols = counts_df.select_dtypes(include=["float64"]).columns
    common_cols = [
        x for x in counts_df.columns if x in growth_rate_df.columns and x not in float_cols
    ]
    df = counts_df.merge(growth_rate_df, on=common_cols, suffixes=("", "_temp"))
    df = df.drop(df.filter(regex="_temp$").columns, axis=1)
    if len(counts_df) != len(df):
        raise ValueError("Rows dropped when merging counts and growth rates!")

    # Subset to desired samples
    df = df[df["DrugConcentration"] == drug_concentration]

    # Trim to exponential growth rate window
    df_pivot = df.copy()
    if trim:
        df_pivot = df_pivot[
            (df_pivot["Time"] >= df_pivot["GrowthRate_window_start"])
            & (df_pivot["Time"] <= df_pivot["GrowthRate_window_end"])
        ]
        df_pivot["Time"] = df_pivot["Time"] - df_pivot["GrowthRate_window_start"]

    # Transform dataframe from long format to wide
    df_pivot = df_pivot.pivot(
        index=[
            "PlateId",
            "WellId",
            "Time",
            "DrugConcentration",
        ],
        columns="CellType",
        values="Count",
    ).reset_index()
    df_pivot = df_pivot.dropna()
    df_pivot["UniqueId"] = df_pivot["PlateId"].map(str) + df_pivot["WellId"]
    df["UniqueId"] = df["PlateId"].map(str) + df["WellId"]

    # Map cell types
    sensitive, resistant = get_cell_types(exp_name)
    cell_type_map = {"S": sensitive, "R": resistant}

    # Define growth rate columns to keep in fit dataframe
    gr_cols = [
        f"Fraction_{sensitive}",
        "GrowthRate",
        "GrowthRate_lowerBound",
        "GrowthRate_higherBound",
        "Intercept",
        "GrowthRate_window_start",
        "GrowthRate_window_end",
    ]

    # Define and fit ODE model
    ode_model = create_model(model)
    params = ode_model.get_params()
    minimize(
        residual_multipleConditions,
        params,
        args=(0, df_pivot, ode_model, cell_type_map, "UniqueId", solver_kws, {}),
        **optimiser_kws,
    )
    k = len(params) - 2

    # Get estimated counts
    df_models = []
    for rep in df_pivot["UniqueId"].unique():
        # Set initial counts
        df_rep = df_pivot[(df_pivot["UniqueId"] == rep)]
        ode_model.paramDic["S0"] = df_rep[df_rep["Time"] == 0][sensitive].iloc[0]
        ode_model.paramDic["R0"] = df_rep[df_rep["Time"] == 0][resistant].iloc[0]
        ode_model.SetParams(**ode_model.paramDic)
        # Run model
        ode_model.Simulate(treatmentScheduleList=ExtractTreatmentFromDf(df_rep), **solver_kws)
        df_model = ode_model.resultsDf.reset_index()
        # Match times
        times = df_rep["Time"].unique()
        out = []
        for cell_type in ["S", "R"]:
            f = interp1d(df_model["Time"], df_model[cell_type], fill_value="extrapolate")
            out.append(
                pd.DataFrame(
                    {"Time": times, "CellType": cell_type_map[cell_type], "Count": f(times)}
                )
            )
        df_model = pd.concat(out)
        # Format results
        df_model = df_model.merge(
            df[(df["UniqueId"] == rep)][["Time", "CellType"] + gr_cols],
            on=["Time", "CellType"],
        )
        if trim:
            df_model["Time"] = df_model["Time"] + df_model["GrowthRate_window_start"]
        df_model["PlateId"] = int(rep[0])
        df_model["WellId"] = rep[1:]
        df_model["DrugConcentration"] = float(drug_concentration)
        # Add parameters
        for p, v in ode_model.paramDic.items():
            df_model[p] = v
        # Get growth rate fit information
        error_sensitive, bic_sensitive = get_fit_error(df_rep, df_model, sensitive, k)
        error_resistant, bic_resistant = get_fit_error(df_rep, df_model, resistant, k)
        df_model["Error"] = np.nan
        df_model.loc[df_model["CellType"] == sensitive, "Error"] = error_sensitive
        df_model.loc[df_model["CellType"] == resistant, "Error"] = error_resistant
        df_model["BIC"] = np.nan
        df_model.loc[df_model["CellType"] == sensitive, "BIC"] = bic_sensitive
        df_model.loc[df_model["CellType"] == resistant, "BIC"] = bic_resistant
        # Concat to experiment model df
        df_models.append(df_model)
    models_df = pd.concat(df_models)

    # Save estimated counts
    models_df.to_csv(f"{data_dir}/{exp_name}/{exp_name}_{model}_fit.csv", index=False)

    # Plot overview of estimated fits and parameters
    if save_figs:
        cell_colors = {sensitive: "#4C956C", resistant: "#EF7C8E"}
        save_loc = f"{data_dir}/{exp_name}/images"
        if not os.path.exists(save_loc):
            os.mkdir(save_loc)
        plot_fits(save_loc, exp_name, model, df, models_df, cell_colors, dc=drug_concentration)
        if model == "replicator":
            plot_freqdepend(
                save_loc,
                exp_name,
                model,
                models_df,
                cell_colors,
                [sensitive, resistant],
                dc=drug_concentration,
            )


def main():
    # Input args
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str, default="data/experimental")
    parser.add_argument("-exp", "--exp_name", type=str, default=None)
    parser.add_argument("-model", "--model", type=str, choices=list(get_models()))
    args = parser.parse_args()

    # Fit model and save results
    exp_names = [args.exp_name] if args.exp_name else os.listdir(args.data_dir)
    for exp_name in exp_names:
        if os.path.isfile(f"{args.data_dir}/{exp_name}") or exp_name == "layout_files":
            continue
        print(exp_name)
        counts_df = calculate_counts(args.data_dir, exp_name)
        growth_rate_df = calculate_growth_rates(args.data_dir, exp_name, counts_df)
        fit(
            args.data_dir,
            exp_name,
            args.model,
            counts_df,
            growth_rate_df,
            0.0,
            True if args.model == "replicator" else False,
        )


if __name__ == "__main__":
    main()
