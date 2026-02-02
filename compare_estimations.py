import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from compare_fits import get_fit_df, plot_errors, plot_errors_facet, qualitative_results
from fitting.odeModels import create_model
from utils import abm_parameter_map


def quantitative_results(save_loc, df):
    # Pivot dataframe from wide to long
    df = pd.melt(
        df,
        id_vars=["Model", "Experiment"],
        value_vars=list(abm_parameter_map().values()),
        var_name="Parameter",
        value_name="Value",
    )

    # Get absolute differences
    for param in abm_parameter_map().values():
        for experiment in df["Experiment"].unique():
            filt = (df["Experiment"] == experiment) & (df["Parameter"] == param)
            gt_value = df[(df["Model"] == "Ground Truth") & filt]["Value"].values[0]
            df.loc[filt, "Difference"] = np.abs(gt_value - df["Value"])
    df = df.dropna(axis=0)

    # Plot differences
    df_diff = df[df["Model"] != "Ground Truth"]
    plot_errors_facet(save_loc, df_diff, sns.barplot, "Model", "Difference", None, "Parameter")
    plot_errors_facet(save_loc, df, sns.barplot, "Model", "Value", None, "Parameter")
    plot_errors_facet(save_loc, df, sns.scatterplot, "Model", "Value", None, "Parameter")


def main():
    # Read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str, default="data/spatial_egt/formatted")
    parser.add_argument("-exp", "--exp_name", type=str, default=None)
    args = parser.parse_args()

    # Get fits and ground truth
    fit_df = get_fit_df(args.data_dir)
    gt_df = pd.read_csv(f"{args.data_dir}/ground_truth.csv")

    # Get parameter names for each model
    fit_params = {}
    for model in fit_df["Model"].unique():
        model_name = model.lower()
        if model == "Game Assay":
            model_name = "replicator"
        param_names = list(create_model(model_name).paramDic.keys())
        param_names.remove("S0")
        param_names.remove("R0")
        fit_params[model] = param_names

    # Get ground truth parameter names for ABM
    gt_params = [x for x in gt_df.columns if x in abm_parameter_map()]

    # Extract out parameter values
    models = []
    for model in fit_df["Model"].unique():
        fit_df_model = fit_df[fit_df["Model"] == model]
        fit_df_model = fit_df_model[["Model", "Experiment"] + fit_params[model]].drop_duplicates()
        models.append(fit_df_model)
    gt_df = gt_df[["Experiment"] + gt_params].drop_duplicates()

    # Combine fit and ground truth dataframes
    gt_df["Model"] = "Ground Truth"
    gt_df = gt_df.rename(columns=abm_parameter_map())
    models.append(gt_df)
    df = pd.concat(models)

    # Save results
    quantitative_results(args.data_dir, df)
    qualitative_results(args.data_dir, df)


if __name__ == "__main__":
    main()
