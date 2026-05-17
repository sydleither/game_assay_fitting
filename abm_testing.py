import argparse
import json
import os
import random

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from create_abm_data import create_run_cmd, sample_three_strategy, sample_two_strategy
from utils import classify_game, classify_three_strategy_replicator, get_colors


def run_experiment(save_loc, args):
    if args.strategies == 2:
        samples = sample_two_strategy(42, args.num_samples)
    else:
        samples = sample_three_strategy(42, args.num_samples)

    run_output = []
    for s, sample in enumerate(samples):
        sample_output = create_run_cmd(
            f"{save_loc}/{s}",
            args.run_cmd,
            42,
            sample,
            s,
            args.strategies,
            random.uniform(0.1, 0.9),
            args.grid,
            args.radius,
            args.write_freq,
            args.steps,
        )
        run_output.append(sample_output)

    with open(f"{save_loc}/run.sh", "w") as f:
        for line in run_output:
            f.write(line)


def get_three_strategy_fp(fixed_point, stable):
    stable_points = sum(1 for x in stable if x and not np.isnan(x))
    if stable_points == 1:
        if stable[0]:
            return 1
        if stable[1]:
            return 0
        if stable[3]:
            return fixed_point[3][0]
    return np.nan


def get_two_strategy_fp(fixed_point, stable):
    if np.isnan(stable[2]):
        if stable[0]:
            return fixed_point[0]
        if stable[1]:
            return fixed_point[1]
    else:
        if stable[2]:
            return fixed_point[2]
        if not stable[2]:
            return np.nan


def classify_two_strategy_replicator(P):
    all_0_stable = P[0][0] > P[1][0]
    all_1_stable = P[1][1] > P[0][1]
    mix_denom = P[0][0] - P[1][0] - P[0][1] + P[1][1]
    if mix_denom != 0:
        mix = (-P[0][1] + P[1][1]) / mix_denom
        mix_stable = mix_denom < 0
    else:
        mix = np.nan
        mix_stable = np.nan
    if mix < 0 or mix > 1:
        mix = np.nan
        mix_stable = np.nan
    return [1, 0, mix], [all_0_stable, all_1_stable, mix_stable]


def ohtsuki_nowak_transform(payoff, radius):
    k = radius**2 + (radius + 1) ** 2 - 1
    dispersal = (1 / (k - 2)) * (np.diag(payoff)[:, None] - np.diag(payoff)[None, :])
    finite = (1 / ((k + 1) * (k - 2))) * (payoff - payoff.T)
    return payoff + dispersal + finite


def plot_space(save_loc):
    coords = pd.read_csv(f"{save_loc}/coords.csv")
    times = sorted(coords["time"].unique())
    fig, ax = plt.subplots(1, len(times), figsize=(5 * len(times), 5))
    for i, time in enumerate(times):
        coords_t = coords[coords["time"] == time]
        grid = np.zeros((100, 100), dtype=np.uint8)
        for x, y, cell_type in coords_t[["x", "y", "strategy"]].values:
            grid[x, y] = cell_type
        colors = ListedColormap(["#4C956C", "#EF7C8E", "#000000"])
        ax[i].imshow(grid, cmap=colors, vmin=0, vmax=len(colors.colors) - 1)
    fig.savefig(f"{save_loc}/spatial.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_fixed_point_distance(save_loc, df, strategies):
    # Dataframe transforms
    df = df.dropna()
    df["Original"] = np.abs(df["frequency"] - df["fp"])
    df["Transformed"] = np.abs(df["frequency"] - df["on fp"])
    df = pd.melt(
        df,
        id_vars=["Sample", "Dynamic"],
        value_vars=["Original", "Transformed"],
        var_name="Payoff Matrix",
        value_name="Distance",
    )
    # Aggregated plot
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.boxplot(data=df, x="Payoff Matrix", y="Distance", color="pink", linecolor="black", ax=ax)
    sns.stripplot(data=df, x="Payoff Matrix", y="Distance", color="black", alpha=0.5, ax=ax)
    ax.set(title=f"{strategies} Strategy Distances Between\nAnalytical and Empirical Fixed Point")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/distance_{strategies}.png", dpi=200)
    plt.close()
    # Split across dynamics plot
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.boxplot(
        data=df,
        x="Payoff Matrix",
        y="Distance",
        hue="Dynamic",
        hue_order=["Sensitive Wins", "Coexistence", "Resistant Wins"],
        palette=get_colors(),
        linecolor="black",
        ax=ax,
    )
    ax.set(title=f"{strategies} Strategy Distances Between\nAnalytical and Empirical Fixed Point")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/distance_{strategies}_dynamic.png", dpi=200)
    plt.close()


def analyze_experiment(save_loc, strategies):
    df = []
    for sample in os.listdir(f"{save_loc}"):
        if os.path.isfile(f"{save_loc}/{sample}"):
            continue
        df_s = pd.read_csv(f"{save_loc}/{sample}/summary.csv")
        df_s = df_s[df_s["time"] == df_s["time"].max()]
        config = json.load(open(f"{save_loc}/{sample}/config.json"))
        payoff = np.array(config["payoff"]).reshape(
            len(config["init_freq"]), len(config["init_freq"])
        )
        on_payoff = ohtsuki_nowak_transform(payoff, config["radius"])
        df_s = df_s[df_s["strategy"] == 0]
        df_s["frequency"] = df_s["frequency"] / config["grid"] ** 2
        if payoff.shape[0] == 2:
            df_s["fp"] = get_two_strategy_fp(*classify_two_strategy_replicator(payoff))
            df_s["on fp"] = get_two_strategy_fp(*classify_two_strategy_replicator(on_payoff))
            df_s["Dynamic"] = classify_game(
                payoff[0, 0],
                payoff[0, 1],
                np.nan,
                payoff[1, 0],
                payoff[1, 1],
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )
        else:
            df_s["fp"] = get_three_strategy_fp(*classify_three_strategy_replicator(payoff))
            df_s["on fp"] = get_three_strategy_fp(*classify_three_strategy_replicator(on_payoff))
            df_s["Dynamic"] = classify_game(*payoff.flatten().tolist())
        df_s["Sample"] = sample
        df.append(df_s)
        if int(sample) % 10 == 0:
            plot_space(f"{save_loc}/{sample}")
    df = pd.concat(df)
    plot_fixed_point_distance(save_loc, df.copy(), strategies)


def main():
    # Get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str, default="data/abm_test")
    parser.add_argument("-run_cmd", "--run_cmd", type=str, default="python3")
    parser.add_argument("-strats", "--strategies", type=int, choices=[2, 3])
    parser.add_argument("-samples", "--num_samples", type=int, default=100)
    parser.add_argument("-l", "--grid", type=int, default=100)
    parser.add_argument("-r", "--radius", type=int, default=1)
    parser.add_argument("-write", "--write_freq", type=int, default=200)
    parser.add_argument("-steps", "--steps", type=int, default=1000)
    args = parser.parse_args()

    save_loc = f"{args.data_dir}/{args.strategies}"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
        run_experiment(save_loc, args)
    else:
        analyze_experiment(save_loc, args.strategies)


if __name__ == "__main__":
    main()
