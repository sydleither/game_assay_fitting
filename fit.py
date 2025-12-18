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
from fitting.odeModels import create_model
from game_assay.game_analysis import calculate_counts, calculate_growth_rates
from game_assay.game_analysis_utils import calculate_fit
from utils import get_cell_types


solver_kws = {
    "method": "RK45",
    "absErr": 1.0e-6,
    "relErr": 1.0e-6,
    "suppressOutputB": False,
    "max_step": 25,
}
optimiser_kws = {
    "method": "least_squares",
    "xtol": 1e-8,
    "ftol": 1e-8,
    "max_nfev": 500,
    "nan_policy": "omit",
    "verbose": 0,
}


def plot_fits(save_loc, exp_name, df, df_model, cell_colors, dc):
    # Set figure dimensions
    plates = sorted(df["PlateId"].unique())
    plate_rows = 1
    plate_cols = len(plates)
    well_rows = sorted(df["RowId"].unique())
    well_cols = sorted(df["ColumnId"].unique())
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
                # Plot fitted line
                df1_model = df_model[
                    (df_model["PlateId"] == str(plate)) & (df_model["WellId"] == f"{row}{col}")
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
                ax_curr.set(title=f"Plate {plate} Well {row}{col}", ylim=(0, y_max))

    # Format figure and save
    fig.suptitle(f"{exp_name} {dc} Drug Concentration")
    plt.savefig(f"{save_loc}/fits_{exp_name}_{dc}dc.png", bbox_inches="tight", dpi=200)
    plt.close()


def main():
    # Input args
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str, default="data/experimental")
    parser.add_argument(
        "-exp", "--exp_name", type=str, default="231010_S-E9_GFP_vs_BRAF_mcherry_Carboplatin"
    )
    models = ["replicator", "lv"]
    parser.add_argument("-model", "--model", type=str, default="replicator", choices=models)
    parser.add_argument("-dc", "--drug_concentration", type=float, default=0.0)
    args = parser.parse_args()

    # Get data
    counts_df = calculate_counts(args.data_dir, args.exp_name)
    growth_rate_df = calculate_growth_rates(args.data_dir, args.exp_name, counts_df)

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
    df = df[~df["GrowthRate"].isna()]
    df = df[df["DrugConcentration"] == args.drug_concentration]

    # Trim to exponential growth rate window
    df_pivot = df.copy()
    if args.model == "replicator":
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
    df_pivot["UniqueId"] = df_pivot["PlateId"].map(str) + df_pivot["WellId"]
    df["UniqueId"] = df["PlateId"].map(str) + df["WellId"]

    # Drop rows where one cell type had low-quality counts
    df_pivot = df_pivot.dropna(axis=0)

    # Map cell types
    sensitive, resistant = get_cell_types(args.exp_name)
    cell_type_map = {"S": sensitive, "R": resistant}

    # Define and fit ODE model
    ode_model = create_model(args.model)
    params = ode_model.get_params()
    minimize(
        residual_multipleConditions,
        params,
        args=(0, df_pivot, ode_model, cell_type_map, "UniqueId", solver_kws, {}),
        **optimiser_kws,
    )

    # Get estimated counts
    model_dfs = []
    for rep in df_pivot["UniqueId"].unique():
        # Set initial counts
        df_rep = df_pivot[(df_pivot["UniqueId"] == rep)]
        ode_model.paramDic["S0"] = df_rep[df_rep["Time"] == 0][sensitive].iloc[0]
        ode_model.paramDic["R0"] = df_rep[df_rep["Time"] == 0][resistant].iloc[0]
        # Run model
        ode_model.Simulate(treatmentScheduleList=ExtractTreatmentFromDf(df_rep), **solver_kws)
        model_df = ode_model.resultsDf.reset_index()
        # Match times
        times = df_rep["Time"].unique()
        out = []
        for cell_type in ["S", "R"]:
            f = interp1d(model_df["Time"], model_df[cell_type], fill_value="extrapolate")
            out.append(
                pd.DataFrame(
                    {"Time": times, "CellType": cell_type_map[cell_type], "Count": f(times)}
                )
            )
        model_df = pd.concat(out)
        # Format results
        model_df = model_df.merge(
            df[(df["UniqueId"] == rep)][["Time", "CellType", "GrowthRate_window_start"]],
            on=["Time", "CellType"],
        )
        model_df["Time"] = model_df["Time"] + model_df["GrowthRate_window_start"]
        model_df["PlateId"] = rep[0]
        model_df["WellId"] = rep[1:]
        # Add parameters
        for p, v in ode_model.paramDic.items():
            model_df[p] = v
        # Get fit information
        fit_sensitive = calculate_fit(
            df_rep[sensitive].values,
            model_df[model_df["CellType"] == sensitive]["Count"].values,
        )
        fit_resistant = calculate_fit(
            df_rep[resistant].values,
            model_df[model_df["CellType"] == resistant]["Count"].values,
        )
        model_df["Fit"] = np.nan
        model_df.loc[model_df["CellType"] == sensitive, "Fit"] = fit_sensitive
        model_df.loc[model_df["CellType"] == resistant, "Fit"] = fit_resistant
        # Concat to experiment model df
        model_dfs.append(model_df)

    # Save estimated counts
    models_df = pd.concat(model_dfs)
    models_df.to_csv(
        f"{args.data_dir}/{args.exp_name}/{args.exp_name}_{args.model}.csv", index=False
    )

    # Plot overview of estimated fits and parameters
    cell_types = [sensitive, resistant]
    cell_colors = {sensitive: "#4C956C", resistant: "#EF7C8E"}
    save_loc = f"{args.data_dir}/{args.exp_name}/images/{args.model}"
    if not os.path.exists(save_loc):
        os.mkdir(save_loc)

    plot_fits(save_loc, args.exp_name, df, models_df, cell_colors, dc=args.drug_concentration)


if __name__ == "__main__":
    main()
