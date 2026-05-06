import argparse
from datetime import date
import os
import random

import pandas as pd

from EGT_HAL.config_utils import latin_hybercube_sample
from fitting.odeModels import create_model, get_models
from game_assay.game_analysis import calculate_growth_rates, calculate_payoffs
from run_game_assay import plot_counts, plot_fits, plot_freqdepend_fit
from utils import get_plate_structure, solver_kws


def main():
    # Get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str)
    parser.add_argument("-model", "--model", type=str, choices=list(get_models()))
    parser.add_argument("-seed", "--seed", type=int, default=42)
    parser.add_argument("-samples", "--num_samples", type=int, default=100)
    parser.add_argument("-noise", "--noise", type=float, default=0.0)
    parser.add_argument("-end", "--end_time", type=int, default=80)
    args = parser.parse_args()

    if args.noise > 0.25:
        raise ValueError("Provide noise (sigma=x*noise) <= 0.25")

    # Set interaction parameters
    if args.model == "replicator":
        samples = latin_hybercube_sample(
            args.num_samples,
            ["p_SS", "p_SR", "p_RS", "p_RR"],
            [0] * 4,
            [0.1] * 4,
            [False, False, False, False],
            seed=args.seed,
        )
    else:
        # 1e-5 = 0.1 / 10000
        samples = latin_hybercube_sample(
            args.num_samples,
            ["r_S", "r_R", "a_SS", "a_SR", "a_RS", "a_RR"],
            [0, 0, -1e-5, -1e-5, -1e-5, -1e-5],
            [0.1, 0.1, 0, 0, 0, 0],
            [False, False, False, False, False, False],
            seed=args.seed,
        )

    # Mimic plate structure
    seeding, colids, rowids = get_plate_structure()
    times = [[x, x + 4, 0] for x in range(0, args.end_time, 4)]
    today_yyyymmdd = date.today().strftime("%y%m%d")

    # Run and save samples
    ground_truth = []
    for s, sample in enumerate(samples):
        df = []
        exp_name = f"{today_yyyymmdd}_sensitive_green_vs_resistant_pink_s{s}"
        if not os.path.exists(f"{args.data_dir}/{exp_name}/images"):
            os.makedirs(f"{args.data_dir}/{exp_name}/images")
        for plate in [1]:
            for i, fs in enumerate(seeding):
                for row in rowids:
                    # Run ODE
                    ode_model = create_model(args.model)
                    for param_name, param_val in sample.items():
                        ode_model.paramDic[param_name] = param_val
                    density = random.randint(500, 1000)
                    ode_model.paramDic["S0"] = fs * density
                    ode_model.paramDic["R0"] = (1 - fs) * density
                    ode_model.SetParams(**ode_model.paramDic)
                    ode_model.Simulate(treatmentScheduleList=times, **solver_kws)
                    model_df = ode_model.resultsDf.reset_index(drop=True)
                    model_df = model_df[model_df["Time"] % 4 == 0]
                    # Add noise to results, if specified
                    if args.noise > 0.0:
                        model_df["S"] = model_df["S"].apply(
                            lambda x: random.gauss(x, args.noise * x)
                        )
                        model_df["R"] = model_df["R"].apply(
                            lambda x: random.gauss(x, args.noise * x)
                        )
                        model_df["TumourSize"] = model_df["S"] + model_df["R"]
                    # Format ODE results
                    model_df["RowId"] = row
                    model_df["ColumnId"] = colids[i]
                    model_df["PlateId"] = plate
                    model_df["Frequency S"] = model_df["S"] / model_df["TumourSize"]
                    model_df["Frequency R"] = model_df["R"] / model_df["TumourSize"]
                    df.append(model_df)
        # Format ODE results as counts_df_processed
        df = pd.concat(df)
        df["WellId"] = df["RowId"] + df["ColumnId"].astype(str)
        df["ReplicateId"] = 0
        df["ImageId"] = 0
        df["Drug"] = f"s{s}"
        df["Time"] = df["Time"].astype(int)
        df_count = pd.melt(
            df,
            id_vars=["Time", "WellId", "PlateId"],
            value_vars=["S", "R"],
            var_name="CellType",
            value_name="Count",
        )
        df_freq = pd.melt(
            df,
            id_vars=["Time", "WellId", "PlateId"],
            value_vars=["Frequency S", "Frequency R"],
            var_name="CellType",
            value_name="Frequency",
        )
        df_freq["CellType"] = df_freq["CellType"].map({"Frequency S": "S", "Frequency R": "R"})
        df_long = df_count.merge(df_freq, on=["CellType", "Time", "WellId", "PlateId"])
        df = df.drop(["S", "R", "Frequency S", "Frequency R", "TumourSize"], axis=1)
        counts_df = df.merge(df_long, on=["Time", "WellId", "PlateId"])
        counts_df["Count"] = counts_df["Count"].astype(int)
        counts_df["CellType"] = counts_df["CellType"].map(
            {"S": "sensitive-green", "R": "resistant-pink"}
        )
        counts_df.to_csv(
            f"{args.data_dir}/{exp_name}/{exp_name}_counts_df_processed.csv", index=False
        )
        # Run rest of game assay
        sensitive_type = "sensitive-green"
        resistant_type = "resistant-pink"
        cell_types = [sensitive_type, resistant_type]
        cell_colors = {sensitive_type: "#4C956C", resistant_type: "#EF7C8E"}
        growth_rate_df = calculate_growth_rates(
            args.data_dir, exp_name, counts_df, None, cell_types
        )
        payoff_df = calculate_payoffs(
            args.data_dir, exp_name, growth_rate_df, cell_types, f"Fraction_{sensitive_type}"
        )
        # Plot experiment visualizations
        save_loc = f"{args.data_dir}/{exp_name}/images"
        plot_counts(save_loc, counts_df, cell_colors)
        # Plot fits for no drug condition
        plot_fits(save_loc, exp_name, counts_df, growth_rate_df, cell_types, cell_colors)
        plot_fits(
            save_loc, exp_name, counts_df, growth_rate_df, cell_types, cell_colors, log_space=True
        )
        plot_freqdepend_fit(
            save_loc,
            exp_name,
            growth_rate_df,
            payoff_df,
            cell_colors,
            cell_types,
        )
        ground_truth.append(sample | {"Experiment": exp_name})

    # Save ground truth csv
    pd.DataFrame(ground_truth).to_csv(f"{args.data_dir}/ground_truth.csv", index=False)


if __name__ == "__main__":
    main()
