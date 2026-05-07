import argparse
from datetime import date
import os
import random

import pandas as pd

from EGT_HAL.config_utils import latin_hybercube_sample
from fitting.odeModels import create_model, get_models
from utils import get_plate_structure, solver_kws


def main():
    # Get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str)
    parser.add_argument("-model", "--model", type=str, choices=list(get_models()))
    parser.add_argument("-reps", "--reps", type=int, default=10)
    parser.add_argument("-samples", "--num_samples", type=int, default=40)
    parser.add_argument("-noise", "--noise", type=float, default=0.0)
    parser.add_argument("-end", "--end_time", type=int, default=80)
    parser.add_argument("-run", "--run_cmd", type=str, default="python3")
    args = parser.parse_args()

    if args.noise > 0.25:
        raise ValueError("Provide noise (sigma=x*noise) <= 0.25")

    # Mimic plate structure
    seeding, colids, rowids = get_plate_structure()
    times = [[x, x + 4, 0] for x in range(0, args.end_time, 4)]
    today_yyyymmdd = date.today().strftime("%y%m%d")
    today_mmddyyyy = date.today().strftime("%m/%d/%y")

    for rep in range(args.reps):
        data_path = f"{args.data_dir}/{rep}"

        # Set interaction parameters
        if args.model == "replicator":
            samples = latin_hybercube_sample(
                args.num_samples,
                ["p_SS", "p_SR", "p_RS", "p_RR"],
                [0] * 4,
                [0.1] * 4,
                [False, False, False, False],
                seed=rep,
            )
        else:
            # 1e-5 = 0.1 / 10000
            samples = latin_hybercube_sample(
                args.num_samples,
                ["r_S", "r_R", "a_SS", "a_SR", "a_RS", "a_RR"],
                [0.05, 0.05, -1e-5, -1e-5, -1e-5, -1e-5],
                [0.1, 0.1, 0, 0, 0, 0],
                [False, False, False, False, False, False],
                seed=rep,
            )

        # Create layout.xlsx
        os.makedirs(f"{data_path}/layout_files")
        layout = []
        for _ in range(3):
            layout.append({j: 0.0 for j in range(2, 7)})
        df_lo = pd.DataFrame(layout, index=["b", "c", "d"])
        df_lo.to_excel(f"{data_path}/layout_files/layout.xlsx")

        # Run and save samples
        ground_truth = []
        overview = []
        for s, sample in enumerate(samples):
            df = []
            exp_name = f"{today_yyyymmdd}_sensitive_green_vs_resistant_pink_s{s}"
            if not os.path.exists(f"{data_path}/{exp_name}/images"):
                os.makedirs(f"{data_path}/{exp_name}/images")
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
                f"{data_path}/{exp_name}/{exp_name}_counts_df_processed.csv", index=False
            )
            ground_truth.append(sample | {"Experiment": exp_name})

            # Add row to overview.xlsx
            overview.append(
                {
                    "Name": exp_name,
                    "Cell Type 1": "sensitive",
                    "Cell Type 2": "resistant",
                    "Fluorophore 1": "green",
                    "Fluorophore 2": "pink",
                    "Drug": f"s{s}",
                    "Experimentalist": "User",
                    "Imaging Frequency": 4,
                    "Number of Plates": 1,
                    "Date": today_mmddyyyy,
                    "Project": "ABM",
                    "Layout File": "layout.xlsx",
                    "Layout File (Original)": "layout.xlsx",
                    "Location": "Local",
                    "Copied?": "y",
                    "Tags": ["green", "pink"],
                    "Growth Rate Window": [24, 72],
                    "Minimum Cell Number": 5,
                    "Notes": "",
                }
            )

        # Save ground truth csv
        df_gt = pd.DataFrame(ground_truth)
        df_gt.to_csv(f"{data_path}/ground_truth.csv", index=False)

        # Save overview.xlsx
        overview_df = pd.DataFrame(overview)
        overview_df["Date"] = pd.to_datetime(overview_df["Date"], format="%m/%d/%y")
        overview_df.to_excel(f"{data_path}/overview.xlsx", index=False)

        # Make images directory
        os.mkdir(f"{data_path}/{exp_name}/images")

    # Save sbatch run scripts
    out = args.data_dir
    with open(f"{out}/fit_models.sh", "w") as f:
        for rep in range(args.reps):
            for exp_name in df_gt["Experiment"].unique():
                arg_str = f"-dir {out}/{rep} -exp {exp_name}"
                f.write(f"{args.run_cmd} run_game_assay.py {arg_str}\n")
                f.write(f"{args.run_cmd} fit_ode.py {arg_str} -model replicator\n")
                f.write(f'{args.run_cmd} fit_ode.py {arg_str} -model "lotka-volterra"\n')


if __name__ == "__main__":
    main()
