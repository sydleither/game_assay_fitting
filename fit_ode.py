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


def plot_freqdepend_fit(save_loc, exp_name, model_name, models_df, cell_colors, cell_types, dc):
    df = models_df.copy().drop(["Time", "Count"], axis=1).drop_duplicates().reset_index()
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.scatterplot(
        data=df,
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
        x0 = "p_SR" if i == 0 else "p_SS"
        x1 = "p_RR" if i == 0 else "p_RS"
        ax.plot(
            [0, 1],
            [df[x0].values[0], df[x1].values[0]],
            ls="--",
            color=cell_colors[cell_type],
        )
    ax.set_title(f"{exp_name}\n{dc} Drug Concentration")
    fig.patch.set_alpha(0.0)
    plt.savefig(
        f"{save_loc}/{model_name}_freqdepend_{exp_name}_{dc}dc.png", bbox_inches="tight", dpi=200
    )
    plt.close()


def fit(data_dir, exp_name, model, drug_concentration):
    # Get data
    counts_df = calculate_counts(data_dir, exp_name)
    growth_rate_df = calculate_growth_rates(data_dir, exp_name, counts_df)

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
    df = df[df["DrugConcentration"] == drug_concentration]

    # Trim to exponential growth rate window
    df_pivot = df.copy()
    trimmed = False
    # if model == "replicator":
    #     df_pivot = df_pivot[
    #         (df_pivot["Time"] >= df_pivot["GrowthRate_window_start"])
    #         & (df_pivot["Time"] <= df_pivot["GrowthRate_window_end"])
    #     ]
    #     df_pivot["Time"] = df_pivot["Time"] - df_pivot["GrowthRate_window_start"]
    #     trimmed = True

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

    # Get estimated counts
    model_dfs = []
    for rep in df_pivot["UniqueId"].unique():
        # Set initial counts
        df_rep = df_pivot[(df_pivot["UniqueId"] == rep)]
        ode_model.paramDic["S0"] = df_rep[df_rep["Time"] == 0][sensitive].iloc[0]
        ode_model.paramDic["R0"] = df_rep[df_rep["Time"] == 0][resistant].iloc[0]
        ode_model.SetParams(**ode_model.paramDic)
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
            df[(df["UniqueId"] == rep)][["Time", "CellType"] + gr_cols],
            on=["Time", "CellType"],
        )
        if trimmed:
            model_df["Time"] = model_df["Time"] + model_df["GrowthRate_window_start"]
        model_df["PlateId"] = int(rep[0])
        model_df["WellId"] = rep[1:]
        model_df["DrugConcentration"] = float(drug_concentration)
        # Add parameters
        for p, v in ode_model.paramDic.items():
            model_df[p] = v
        # Get growth rate fit information
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
    models_df = pd.concat(model_dfs)

    # Get frequency dependence fits
    if model == "replicator":
        models_df["Sensitive Estimated"] = (
            (models_df["p_SS"] - models_df["p_SR"]) * models_df[f"Fraction_{sensitive}"]
        ) + models_df["p_SR"]
        models_df["Resistant Estimated"] = (
            (models_df["p_RS"] - models_df["p_RR"]) * models_df[f"Fraction_{sensitive}"]
        ) + models_df["p_RR"]
        sensitive_fit = calculate_fit(
            models_df[models_df["CellType"] == sensitive]["GrowthRate"],
            models_df[models_df["CellType"] == sensitive]["Sensitive Estimated"],
        )
        resistant_fit = calculate_fit(
            models_df[models_df["CellType"] == resistant]["GrowthRate"],
            models_df[models_df["CellType"] == resistant]["Resistant Estimated"],
        )
        models_df.loc[models_df["CellType"] == resistant, "Frequency Dependence Fit"] = resistant_fit
        models_df.loc[models_df["CellType"] == sensitive, "Frequency Dependence Fit"] = sensitive_fit
        models_df = models_df.drop(["Sensitive Estimated", "Resistant Estimated"], axis=1)

    # Save estimated counts
    models_df.to_csv(f"{data_dir}/{exp_name}/{exp_name}_{model}_fit.csv", index=False)

    # Plot overview of estimated fits and parameters
    cell_colors = {sensitive: "#4C956C", resistant: "#EF7C8E"}
    save_loc = f"{data_dir}/{exp_name}/images"
    if not os.path.exists(save_loc):
        os.mkdir(save_loc)
    plot_fits(save_loc, exp_name, model, df, models_df, cell_colors, dc=drug_concentration)
    if model == "replicator":
        plot_freqdepend_fit(
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
    models = ["replicator", "lv"]
    parser.add_argument("-model", "--model", type=str, default="replicator", choices=models)
    args = parser.parse_args()

    # Fit model and save results
    exp_names = [args.exp_name] if args.exp_name else os.listdir(args.data_dir)
    for exp_name in exp_names:
        if os.path.isfile(f"{args.data_dir}/{exp_name}") or exp_name == "layout_files":
            continue
        print(exp_name)
        fit(args.data_dir, exp_name, args.model, 0)


if __name__ == "__main__":
    main()
