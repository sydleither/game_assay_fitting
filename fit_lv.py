import argparse

import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from game_assay.game_analysis import count_cells
from utils import get_cell_types


def lnprob(params, time, x0, x1, true_time, true_s, true_r):
    for p in params[0:2]:
        if p > 2 or p < -2:
            return -np.inf
    for p in params[2:]:
        if p > 1e-2 or p < -1e-2:
            return -np.inf
    glv_s, glv_r = glv(time, x0, x1, *params)
    glv_s = [glv_s[i] for i in range(time+1) if i in true_time]
    glv_r = [glv_r[i] for i in range(time+1) if i in true_time]
    s_err = -0.5 * np.sum((true_s - glv_s)**2)
    r_err = -0.5 * np.sum((true_r - glv_r)**2)
    return (s_err+r_err)/2


def glv(time, x0, x1, r0, r1, a00, a01, a10, a11):
    dx0dt = []
    dx1dt = []
    for _ in range(time+1):
        dx0dt.append(x0)
        dx1dt.append(x1)
        x0 = min(r0*x0 * (1 - (a00*x1 + a01*x1)), 1e8)
        x1 = min(r1*x1 * (1 - (a10*x0 + a11*x0)), 1e8)
    return dx0dt, dx1dt


def main():
    # Input args
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str, default="../agent-based-games/data/in_vitro_pc9/raw")
    parser.add_argument("-exp", "--exp_name", type=str, default="220405_S-E9_gfp_vs_BRAF_mcherry_Gefitinib")
    parser.add_argument("-plate", "--plate_id", type=int, default=1)
    parser.add_argument("-well", "--well_id", type=str, default="B2")
    parser.add_argument("-model", "--model", type=str, default="logistic")
    args = parser.parse_args()

    # Experimental count data
    df = count_cells(args.data_dir, args.exp_name)
    df = df[(df["PlateId"] == args.plate_id) & (df["WellId"] == args.well_id)]

    # Extract counts
    sensitive_type, resistant_type = get_cell_types(args.exp_name)
    sensitive_counts = df[df["CellType"] == sensitive_type]["Count"].values
    resistant_counts = df[df["CellType"] == resistant_type]["Count"].values
    x0 = sensitive_counts[0]
    x1 = resistant_counts[0]

    # Extract times
    times = sorted(list(df["Time"].unique()))
    time_end = times[-1]

    # MCMC
    nwalkers = 50
    niter = 500
    p0 = np.array([1, 1, 0, 0, 0, 0])
    p0 = [p0 + 1e-5 * np.random.randn(6) for _ in range(nwalkers)]
    lnprob_args = (time_end, x0, x1, times, sensitive_counts, resistant_counts)
    sampler = emcee.EnsembleSampler(nwalkers, 6, lnprob, args=lnprob_args)
    sampler.run_mcmc(p0, niter)
    walker_ends = sampler.get_chain(discard=niter - 1)[0, :, :]

    # Plot
    fig, ax = plt.subplots()
    for walker in walker_ends:
        s, r = glv(time_end, x0, x1, *walker)
        ax.plot(range(time_end+1), s, color="hotpink", alpha=0.5)
        ax.plot(range(time_end+1), r, color="forestgreen", alpha=0.5)
    ax.plot(times, sensitive_counts, color="hotpink", ls="--")
    ax.plot(times, resistant_counts, color="forestgreen", ls="--")
    fig.savefig("test.png")


if __name__ == "__main__":
    main()
