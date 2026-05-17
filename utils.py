import os
from string import ascii_uppercase

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import eigvals
from scipy.optimize import approx_fprime

from game_assay.game_analysis_utils import (
    optimize_growth_rate_window_per_exp,
    optimize_growth_rate_window_per_well,
)


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


def get_colors():
    return {
        "Sensitive Wins": "#4C956C",
        "Coexistence": "#C28367",
        "Bistability": "#047495",
        "Resistant Wins": "#EF7C8E",
        # "Neutrality": "#767567",
    }


def latin_hypercube_sample(num_samples, param_names, lower_bounds, upper_bounds, ints, seed):
    sampler = stats.qmc.LatinHypercube(d=len(lower_bounds), seed=seed)
    unscaled_sample = sampler.random(n=num_samples)
    sample = stats.qmc.scale(unscaled_sample, lower_bounds, upper_bounds).tolist()
    sampled_params = [
        {param_names[i]: round(s[i]) if ints[i] else s[i] for i in range(len(s))}
        for s in sample
    ]
    return sampled_params


def label_data_type(data_dir):
    noisy = "Noisy " if "noisy" in data_dir else ""
    if "ode_egt" in data_dir:
        return f"{noisy}Exponential Growth ODE"
    if "ode_lv" in data_dir:
        return f"{noisy}Lotka-Volterra ODE"
    if "abm" in data_dir and "_" in data_dir:
        parts = data_dir.split("/")[1].split("_")
        return f"{parts[1]} Strategy {parts[2]} Spatial Agent-Based Model"
    return data_dir.replace("data/", "").title().replace("_", " ")


def set_growth_rate_window(counts_df, window):
    gr_window = None
    if window == "none":
        gr_window = (counts_df["Time"].min(), counts_df["Time"].max())
    elif window == "per_well":
        counts_df = optimize_growth_rate_window_per_well(counts_df)
    elif window == "per_exp":
        counts_df = optimize_growth_rate_window_per_exp(counts_df)
    elif window == "per_cell":
        gr_window = None
    else:
        raise ValueError(f"Illegal growth rate window: {window}")
    return counts_df, gr_window


# From gemini
def analyze_significance(df, group_col, value_col, control_label):
    # 1. Prepare grouped data
    groups = df.groupby(group_col)[value_col].apply(list).to_dict()

    if control_label not in groups:
        raise ValueError(f"Control label '{control_label}' not found in column '{group_col}'")

    control_data = np.array(groups[control_label])
    control_mean = np.mean(control_data)

    results = []

    # 2. Define the bootstrap statistic for difference
    def mean_diff(sample_target, sample_control):
        return np.mean(sample_target) - np.mean(sample_control)

    for label, data in groups.items():
        data = np.array(data)
        current_mean = np.mean(data)

        # --- Calculate Individual CI ---
        res_ind = stats.bootstrap(
            (data,), np.mean, confidence_level=0.95, n_resamples=1000, method="percentile"
        )

        # --- Calculate CI of Difference (vs Control) ---
        # If it's the control itself, the difference is 0 and CI is [0,0]
        if label == control_label:
            diff_mean, diff_low, diff_high = 0.0, 0.0, 0.0
            is_significant = False
        else:
            res_diff = stats.bootstrap(
                (data, control_data),
                mean_diff,
                confidence_level=0.95,
                n_resamples=1000,
                method="percentile",
            )
            diff_mean = current_mean - control_mean
            diff_low = res_diff.confidence_interval.low
            diff_high = res_diff.confidence_interval.high
            # Significant if the interval does not contain 0
            is_significant = not (diff_low <= 0 <= diff_high)

        results.append(
            {
                group_col: label,
                "Mean": current_mean,
                "CI_Low": res_ind.confidence_interval.low,
                "CI_High": res_ind.confidence_interval.high,
                "Diff_vs_Control": diff_mean,
                "Diff_CI_Low": diff_low,
                "Diff_CI_High": diff_high,
                "Significant": is_significant,
            }
        )

    return pd.DataFrame(results)


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
        if os.path.isfile(f"{data_dir}/{exp_name}") or not exp_name.startswith("2"):
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


def classify_three_strategy_replicator(P):
    def check_edge(P, i, j):
        mix_denom = P[i][i] - P[j][i] - P[i][j] + P[j][j]
        if mix_denom != 0:
            mix = (-P[i][j] + P[j][j]) / mix_denom
            mix_stable = mix_denom < 0
        else:
            mix = np.nan
            mix_stable = np.nan
        if mix <= 0 or mix >= 1:
            mix = np.nan
            mix_stable = np.nan
        return mix, mix_stable

    def replicator(t, x, P):
        return x * (P @ x - x.T @ P @ x)

    def f(x):
        return replicator(None, np.array([x[0], x[1], 1 - x[0] - x[1]]), P)[:2]

    def interior_stability(mix_all, P):
        x_eq = np.array(mix_all[:2])
        J = np.array([approx_fprime(x_eq, lambda x: f(x)[i], 1e-6) for i in range(2)])
        eigs = eigvals(J)
        return bool(np.all(np.real(eigs) < 0))

    # Corners
    all_0_stable = P[0][0] > P[1][0] and P[0][0] > P[2][0]
    all_1_stable = P[1][1] > P[0][1] and P[1][1] > P[2][1]
    all_2_stable = P[2][2] > P[0][2] and P[2][2] > P[1][2]
    # Edges
    mix_01, mix_01_stable = check_edge(P, 0, 1)
    mix_02, mix_02_stable = check_edge(P, 0, 2)
    mix_12, mix_12_stable = check_edge(P, 1, 2)
    # Interior
    eqs_lhs = np.array([P[0] - P[1], P[0] - P[2], [1, 1, 1]])
    eqs_rhs = np.array([0, 0, 1])
    try:
        mix_all = np.linalg.solve(eqs_lhs, eqs_rhs)
        mix_all = (
            mix_all if np.all(mix_all >= 0) and np.all(mix_all <= 1) else [np.nan, np.nan, np.nan]
        )
    except np.linalg.LinAlgError:
        mix_all = [np.nan, np.nan, np.nan]
    mix_all_stable = np.nan
    if not np.any(np.isnan(mix_all)):
        mix_all_stable = interior_stability(mix_all, P)
    # Return
    return (
        [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (mix_01, 1 - mix_01, 0),
            (mix_02, 0, 1 - mix_02),
            (0, mix_12, 1 - mix_12),
            mix_all,
        ],
        [
            all_0_stable,
            all_1_stable,
            all_2_stable,
            mix_01_stable,
            mix_02_stable,
            mix_12_stable,
            mix_all_stable,
        ],
    )


def classify_three_strategy_dynamic(stable):
    stable_points = sum(1 for x in stable if x and not np.isnan(x))
    players = [
        "Sensitive",
        "Resistant",
        "Empty",
        "Sensitive and Resistant",
        "Sensitive and Empty",
        "Resistant and Empty",
        "All",
    ]
    if stable_points == 0:
        return "Neutrality"
    elif stable_points == 1:
        for i in range(len(stable)):
            if stable[i] and not np.isnan(stable[i]):
                if 0 <= i < 3:
                    return f"{players[i]} Wins"
                if 3 <= i <= 6:
                    return "Coexistence"  # f"coexistence: {players[i]}"
    else:
        # outcome = ", ".join(
        #     [players[p] for p in range(len(stable)) if stable[p] and not np.isnan(stable[p])]
        # )
        return "Bistability"  # f"bistability: {outcome}"


def classify_two_strategy_dynamic(a, b, c, d):
    if a > c and b > d:
        return "Sensitive Wins"
    if c > a and b > d:
        return "Coexistence"
    if a > c and d > b:
        return "Bistability"
    if c > a and d > b:
        return "Resistant Wins"
    return "Neutrality"


def classify_game(p_ss, p_sr, p_se, p_rs, p_rr, p_re, p_es, p_er, p_ee):
    # Not a game
    if np.any(np.isnan([p_ss, p_sr, p_rs, p_rr])):
        return np.nan
    # Two strategy game
    if np.any(np.isnan([p_se, p_re, p_es, p_er, p_ee])):
        return classify_two_strategy_dynamic(p_ss, p_sr, p_rs, p_rr)
    # Three strategy game
    _, stabilities = classify_three_strategy_replicator(
        np.array([p_ss, p_sr, p_se, p_rs, p_rr, p_re, p_es, p_er, p_ee]).reshape(3, 3)
    )
    return classify_three_strategy_dynamic(stabilities)


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
        lambda x: classify_game(
            x["p_SS"],
            x["p_SR"],
            x["p_SE"],
            x["p_RS"],
            x["p_RR"],
            x["p_RE"],
            x["p_ES"],
            x["p_ER"],
            x["p_EE"],
        ),
        axis=1,
    )
    if df_q["Replicator Dynamic"].isna().sum() < len(df_q):
        df_q.loc[(df_q["p_SS"].isna()) & (~df_q["p_RR"].isna()), "Replicator Dynamic"] = (
            "Resistant Wins"
        )
        df_q.loc[(df_q["p_RR"].isna()) & (~df_q["p_SS"].isna()), "Replicator Dynamic"] = (
            "Sensitive Wins"
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
