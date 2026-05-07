import os
from string import ascii_uppercase

import numpy as np
import pandas as pd


solver_kws = {
    "method": "RK45",
    "absErr": 1.0e-6,
    "relErr": 1.0e-4,
    "suppressOutputB": True,
    "max_step": 1,
    "dt": 0.01,
}

optimiser_kws = {
    "method": "least_squares",
    "xtol": 1e-8,
    "ftol": 1e-8,
    "max_nfev": 1000,
    "nan_policy": "omit",
    "verbose": 0,
}


def get_cell_types(exp_name):
    parts = exp_name.split("_")
    s = f"{parts[1]}-{parts[2].lower()}"
    r = f"{parts[4]}-{parts[5]}"
    return s, r


def get_plate_structure():
    seeding = [0.1, 0.3, 0.5, 0.7, 0.9]
    colids = [2, 3, 4, 5, 6]
    rowids = ascii_uppercase[1:4]
    return seeding, colids, rowids


def get_parameter_names():
    return [
        "p_SS",
        "p_SR",
        "p_RS",
        "p_RR",
        "a_SS",
        "a_SR",
        "a_RS",
        "a_RR",
        "r_S",
        "r_R",
    ]


def get_parameter_ranges(model):
    if model == "replicator":
        return [(0.03, 0.05)] * 4
    elif model == "lotka-volterra":
        return [(0.03, 0.05)] * 2 + [(-4e-6, -2e-7)] * 4


def get_colors():
    return {
        "Sensitive Wins": "#4C956C",
        "Coexistence": "#C28367",
        "Bistability": "#047495",
        "Resistant Wins": "#EF7C8E",
        "Neutrality": "#767567"
    }


def label_data_type(data_dir):
    noisy = "Noisy " if "noisy" in data_dir else ""
    if "ode_egt" in data_dir:
        return f"{noisy}Exponential Growth ODE"
    if "ode_lv" in data_dir:
        return f"{noisy}Lotka-Volterra ODE"
    if "abm" in data_dir and "_" in data_dir:
        return data_dir.split("/")[1].split("_")[1].title() + " Spatial Agent-Based Model"
    return data_dir.replace("data/", "").title().replace("_", " ")


##################
# Data Wrangling #
##################
def read_and_format_ode(data_dir, exp_name, file_name, sensitive_type):
    model = file_name.split("_")[-2].title()
    df = pd.read_csv(f"{data_dir}/{exp_name}/{file_name}")
    df = df.drop(["Time", "Count"], axis=1).drop_duplicates()
    df = df.rename({f"Fraction_{sensitive_type}": "Fraction Sensitive"}, axis=1)
    df["Model"] = model
    if model == "Replicator":
        df["Advantage Sensitive"] = df["p_SR"] - df["p_RR"]
        df["Advantage Resistant"] = df["p_RS"] - df["p_SS"]
    return df.reset_index(drop=True)


def read_and_format_game_assay(data_dir, exp_name, sensitive_type):
    # Read in growth rate and payoff data
    gr_path = f"{data_dir}/{exp_name}/{exp_name}_growth_rate_df_processed.csv"
    payoff_path = f"{data_dir}/{exp_name}/{exp_name}_game_params_df_processed.csv"
    if not os.path.exists(gr_path) or not os.path.exists(payoff_path):
        return
    growth_rate_df = pd.read_csv(gr_path)
    payoff_df = pd.read_csv(payoff_path)
    # Transform payoff dataframe from wide format to long
    payoff_df = payoff_df.drop(["error"], axis=1)
    payoff_df = pd.wide_to_long(
        payoff_df, stubnames=["Type", "error"], i="DrugConcentration", j="n"
    )
    payoff_df = payoff_df.reset_index()
    payoff_df = payoff_df.rename(
        {
            "Type": "CellType",
            "error": "Frequency Dependence Error",
            "p11": "p_SS",
            "p12": "p_SR",
            "p21": "p_RS",
            "p22": "p_RR",
            "Advantage_0": "Advantage Sensitive",
            "Advantage_1": "Advantage Resistant",
        },
        axis=1,
    )
    payoff_df = payoff_df.drop(["n", "c12", "c21", "r1", "r2"], axis=1)
    # Format growth rate dataframe
    if "BIC" in growth_rate_df:  # TODO
        growth_rate_df = growth_rate_df.drop("BIC", axis=1)
    growth_rate_df = growth_rate_df.rename(
        {
            "GrowthRate_error": "Error",
            "GrowthRate_BIC": "BIC",
            f"Fraction_{sensitive_type}": "Fraction Sensitive",
        },
        axis=1,
    )
    growth_rate_df = growth_rate_df[
        [
            "PlateId",
            "WellId",
            "DrugConcentration",
            "CellType",
            "Fraction Sensitive",
            "GrowthRate",
            "GrowthRate_lowerBound",
            "GrowthRate_higherBound",
            "Intercept",
            "GrowthRate_window_start",
            "GrowthRate_window_end",
            "Error",
            "BIC",
        ]
    ]
    # Combine dataframes
    df = growth_rate_df.merge(payoff_df, on=["DrugConcentration", "CellType"])
    df["Model"] = "Game Assay"
    return df.reset_index(drop=True)


def get_fit_df(data_dir):
    df = []
    for exp_name in os.listdir(data_dir):
        if os.path.isfile(f"{data_dir}/{exp_name}") or exp_name == "layout_files":
            continue
        sensitive_type, resistant_type = get_cell_types(exp_name)
        # Read in ODE fit data
        df_ode = []
        for file_name in os.listdir(f"{data_dir}/{exp_name}"):
            if file_name.split("_")[-1] == "fit.csv":
                df_ode.append(read_and_format_ode(data_dir, exp_name, file_name, sensitive_type))
        if len(df_ode) == 0:
            continue
        df_ode = pd.concat(df_ode, ignore_index=True)
        # Read in game assay fit data
        df_assay = read_and_format_game_assay(data_dir, exp_name, sensitive_type)
        # Merge game assay and ODE dataframes
        df_comb = pd.concat([df_ode, df_assay], ignore_index=True)
        df_comb["Experiment"] = exp_name
        df_comb["Resistant Type"] = resistant_type
        df.append(df_comb)
    df = pd.concat(df)
    df = df[df["DrugConcentration"] == 0.0]
    return df


def classify_game(a, b, c, d):
    if np.any(np.isnan([a, b, c, d])):
        return np.nan
    if a > c and b > d:
        return "Sensitive Wins"
    if c > a and b > d:
        return "Coexistence"
    if a > c and d > b:
        return "Bistability"
    if c > a and d > b:
        return "Resistant Wins"
    return "Neutrality"


def classify_lv_dynamic(r_S, r_R, a_SS, a_SR, a_RS, a_RR):
    if np.any(np.isnan([r_S, r_R, a_SS, a_SR, a_RS, a_RR])):
        return np.nan
    denom = a_SS * a_RR - a_SR * a_RS
    if denom == 0:
        return "Neutrality"
    mix = ((a_SR * r_R - a_RR * r_S) / denom, (a_RS * r_S - a_SS * r_R) / denom)
    if mix[0] < 0 or mix[1] < 0:
        mix_stable = np.nan
    else:
        mix_stable = denom > 0
    all_0_stable = r_R - a_RS * (r_S / a_SS) < 0
    all_1_stable = r_S - a_SR * (r_R / a_RR) < 0
    if np.isnan(mix_stable):
        if all_0_stable:
            return "Sensitive Wins"
        if all_1_stable:
            return "Resistant Wins"
        return "Unbounded Growth"
    else:
        if mix_stable:
            return "Coexistence"
        if not mix_stable:
            return "Bistability"


def label_qualitative_dynamics(df, keys=["Replicate", "Model", "Experiment"]):
    # Reduce dataframe
    for param in get_parameter_names():
        if param not in df:
            df[param] = np.nan
    df_q = df[keys + get_parameter_names()].drop_duplicates()

    # Get game quadrant of game-theoretic models
    df_q["Replicator Dynamic"] = df_q.apply(
        lambda x: classify_game(x["p_SS"], x["p_SR"], x["p_RS"], x["p_RR"]), axis=1
    )

    # Get fixed point dynamics of lotka-volterra models
    df_q["Lotka-Volterra Dynamic"] = df_q.apply(
        lambda x: classify_lv_dynamic(
            x["r_S"], x["r_R"], x["a_SS"], x["a_SR"], x["a_RS"], x["a_RR"]
        ),
        axis=1,
    )

    # Combine LV and EGT long-term dynamics columns
    df_q["Dynamic"] = df_q["Lotka-Volterra Dynamic"].fillna(df_q["Replicator Dynamic"])
    df_q = df_q[keys + ["Dynamic"]]
    return df.merge(df_q, on=keys)


def format_for_plotting(df):
    # Formatting exponential growth window
    if "Growth Rate Window" in df.columns:
        df.loc[df["Growth Rate Window"] == "per_exp", "Growth Rate Window"] = "Per-Experiment"
        df.loc[df["Growth Rate Window"] == "per_well", "Growth Rate Window"] = "Per-Replicate"
        df.loc[df["Growth Rate Window"] == "per_cell", "Growth Rate Window"] = "Per-Cell-Type"
        df.loc[df["Growth Rate Window"] == "none", "Growth Rate Window"] = "None"
        df = df.rename({"Growth Rate Window": "Exponential Growth Window Strategy"}, axis=1)
    # Formatting parameter names
    new_param_names = {}
    for param in get_parameter_names():
        new_param = param
        if "p_" in param or "a_" in param:
            new_param = param.upper()
        new_param = rf"${new_param[0]}_{{{new_param[2:]}}}$"
        new_param_names[param] = new_param
    df = df.rename(new_param_names, axis=1)
    # Renaming replicator to exponential growth
    df.loc[df["Model"] == "Replicator", "Model"] = "Exponential Growth ODE"
    df.loc[df["Model"] == "Lotka-Volterra", "Model"] = "Lotka-Volterra ODE"
    return df
