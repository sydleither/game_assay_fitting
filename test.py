"""
Plots to help understand the replicator equation and game spaces.
"""

import matplotlib.pyplot as plt
import numpy as np


def replicator(x, a, b, c, d):
    """
    The replicator equation defined in Rowan et al., 2024
    """
    return x * (1 - x) * ((c - a) * (1 - x) - (b - d) * x)


def classify_game(a, b, c, d):
    """
    Return game given a,b,c,d
    """
    if a > c and b > d:
        game = "sensitive_wins"
    elif c > a and b > d:
        game = "coexistence"
    elif a > c and d > b:
        game = "bistability"
    elif c > a and d > b:
        game = "resistant_wins"
    else:
        game = "unknown"
    return game


def phase_space(save_loc):
    """
    Plot sample phase spaces
    """
    games = [(0.2, 0.4, 0.1, 0.3), (0.1, 0.4, 0.2, 0.3), (0.2, 0.3, 0.1, 0.4), (0.1, 0.3, 0.2, 0.4)]
    timepoints = np.arange(0, 1.1, 0.1)
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    for i, game in enumerate(games):
        a, b, c, d = game
        if np.isclose(((b - d) + (c - a)), 0):
            xstar = 0
        else:
            xstar = (c - a) / ((b - d) + (c - a))
        y_results = []
        for t in timepoints:
            y = replicator(t, a, b, c, d)
            y_results.append(y)
        ax[i].plot(timepoints, y_results, color="black")
        for equilibrium in [[0,0], [1,0], [xstar,0]]:
            stability = check_stability(equilibrium[0], a, b, c, d)
            fc = "none" if stability != 1 else "black"
            ax[i].scatter(equilibrium[0], equilibrium[1], facecolors=fc, edgecolors="black")
        ax[i].set(title=classify_game(a, b, c, d), xlabel="x", ylabel="dx/dt")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(f"{save_loc}/phase_spaces.png", bbox_inches="tight", dpi=200)


def check_stability(x, a, b, c, d):
    """
    Check if a point x is stable
    """
    neg_epsilon = replicator(x - 0.01, a, b, c, d)
    pos_epsilon = replicator(x + 0.01, a, b, c, d)
    if pos_epsilon < 0 < neg_epsilon:
        return 1
    if neg_epsilon < 0 < pos_epsilon:
        return -1
    return 0


def plot_stability(save_loc, file_name, x, y, zero_stable, int_stable, one_stable, xlabel, ylabel):
    """
    Plot stability of x=0, x=x*, x=1 equilibria
    """
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].scatter(x, y, c=zero_stable, cmap="PiYG")
    ax[0].set(title="Stability of x=0 Equilibrium", xlabel=xlabel, ylabel=ylabel)
    ax[1].scatter(x, y, c=int_stable, cmap="PiYG")
    ax[1].set(title="Stability of x=x* Equilibrium", xlabel=xlabel, ylabel=ylabel)
    points = ax[2].scatter(x, y, c=one_stable, cmap="PiYG")
    ax[2].set(title="Stability of x=1 Equilibrium", xlabel=xlabel, ylabel=ylabel)
    for i in range(3):
        ax[i].axhline(0, c="black")
        ax[i].axvline(0, c="black")
    fig.colorbar(points)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/{file_name}.png")


def stability(save_loc):
    """
    Plot stability of equilibria of the replicator equation with varying a,b,c,d
    """
    vals = np.arange(-0.1, 0.1, 0.02)
    x = []
    y = []
    zero_stable = []
    one_stable = []
    int_stable = []
    for a in vals:
        for b in vals:
            for c in vals:
                for d in vals:
                    x.append(c - a)
                    y.append(b - d)
                    if np.isclose((b - d) + (c - a), 0):
                        int_stable.append(0)
                    else:
                        int_eq = (c - a) / ((b - d) + (c - a))
                        int_stable.append(check_stability(int_eq, a, b, c, d))
                    zero_stable.append(check_stability(0, a, b, c, d))
                    one_stable.append(check_stability(1, a, b, c, d))
    plot_stability(save_loc, "stability", x, y, zero_stable, int_stable, one_stable, "c-a", "b-d")


def stability_transformed(save_loc, sm):
    """
    Plot stability of equilibria of the transformed replicator equation with varying awm, amw, sm
    """
    vals = np.arange(-0.1, 0.1, 0.01)
    x = []
    y = []
    zero_stable = []
    one_stable = []
    int_stable = []
    for awm in vals:
        for amw in vals:
            x.append(amw / sm)
            y.append(awm / sm)
            a = 0
            b = awm
            c = sm + amw
            d = sm
            if np.isclose((b - d) + (c - a), 0):
                int_stable.append(0)
            else:
                int_eq = (c - a) / ((b - d) + (c - a))
                int_stable.append(check_stability(int_eq, a, b, c, d))
            zero_stable.append(check_stability(0, a, b, c, d))
            one_stable.append(check_stability(1, a, b, c, d))
    plot_stability(
        save_loc, f"stability_{sm}sm", x, y, zero_stable, int_stable, one_stable, "amw/sm", "awm/sm"
    )


def main():
    """
    Plot explanatory figures.
    """
    save_loc = "."
    phase_space(save_loc)
    stability(save_loc)
    stability_transformed(save_loc, 0.05)


if __name__ == "__main__":
    main()
