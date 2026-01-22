import argparse
import os
from warnings import filterwarnings

import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import euclidean
import seaborn as sns

from compare_fits import get_fit_df
from fitting.odeModels import create_model
from utils import get_cell_types

filterwarnings("ignore")


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
    gt_params = ["A", "B", "C", "D"]

    # Extract out parameter values
    models = []
    for model in fit_df["Model"].unique():
        fit_df_model = fit_df[fit_df["Model"] == model]
        fit_df_model = fit_df_model[["Model", "Experiment"] + fit_params[model]].drop_duplicates()
        models.append(fit_df_model)
    gt_df = gt_df[["Experiment"] + gt_params].drop_duplicates()

    for model in models:
        print(model)
    print(gt_df)


if __name__ == "__main__":
    main()
