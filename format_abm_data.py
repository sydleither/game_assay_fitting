import argparse
from datetime import date
import json
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str)
    parser.add_argument("-in", "--in_dir", type=str, default="raw")
    parser.add_argument("-out", "--out_dir", type=str, default=".")
    parser.add_argument("-run", "--run_cmd", type=str, default="python3")
    args = parser.parse_args()

    data_dir = args.data_dir
    raw_dir = args.in_dir
    out_dir = args.out_dir
    today_yyyymmdd = date.today().strftime("%y%m%d")
    today_mmddyyyy = date.today().strftime("%m/%d/%y")

    # Save cell data and create overview.xlsx
    reps = []
    for rep in os.listdir(data_dir):
        if os.path.isfile(f"{data_dir}/{rep}"):
            continue
        reps.append(rep)

        # Create layout.xlsx
        if not os.path.exists(f"{data_dir}/{rep}/{out_dir}/layout_files"):
            os.makedirs(f"{data_dir}/{rep}/{out_dir}/layout_files")
        layout = []
        for _ in range(3):
            layout.append({j: 0.0 for j in range(2, 7)})
        df_lo = pd.DataFrame(layout, index=["b", "c", "d"])
        df_lo.to_excel(f"{data_dir}/{rep}/{out_dir}/layout_files/layout.xlsx")

        overview = []
        ground_truth = []
        for exp in os.listdir(f"{data_dir}/{rep}/{raw_dir}"):
            full_path = f"{data_dir}/{rep}/{raw_dir}/{exp}"
            if os.path.isfile(full_path):
                continue
            exp_new = f"{today_yyyymmdd}_sensitive_green_vs_resistant_pink_s{exp}"
            plates = 0
            for plate in os.listdir(full_path):
                plates += 1
                out_loc = (
                    f"{data_dir}/{rep}/{out_dir}/{exp_new}/results_stitched_images_plate{plate}"
                )
                if not os.path.exists(out_loc):
                    os.makedirs(out_loc)
                for well in os.listdir(f"{full_path}/{plate}"):
                    out_name = f"segmentation_results_well_{well}"
                    # Read in coordinates
                    df = pd.read_csv(f"{full_path}/{plate}/{well}/coords.csv")
                    # Save ground truth data
                    config = json.load(open(f"{full_path}/{plate}/{well}/config.json"))
                    config["WellId"] = well
                    config["PlateId"] = plate
                    config["Experiment"] = exp_new
                    ground_truth.append(config)
                    # Save counts file
                    counts = pd.read_csv(f"{full_path}/{plate}/{well}/summary.csv")
                    counts = pd.pivot(counts, index="time", columns="strategy", values="frequency")
                    counts = counts.reset_index()
                    counts = counts[counts["time"].isin(df["time"])]
                    counts["time"] = pd.factorize(counts["time"])[0] + 1
                    df["time"] = pd.factorize(df["time"])[0] + 1
                    if 0 not in counts.columns:
                        counts[0] = 0
                    if 1 not in counts.columns:
                        counts[1] = 0
                    counts["ImageNumber"] = counts.index
                    counts = counts.rename(
                        {0: "Count_green_objects", 1: "Count_pink_objects"},
                        axis=1,
                    )
                    counts["timestamp"] = counts["time"].astype(str).str.zfill(3)
                    counts["FileName_green"] = well + "_na_" + counts["timestamp"] + ".tif"
                    counts["FileName_pink"] = well + "_na_" + counts["timestamp"] + ".tif"
                    counts.to_csv(f"{out_loc}/{out_name}_results.csv", index=False)
                    # Save location files
                    df["Metadata_Well"] = well
                    df["Location_Center_Z"] = 0
                    df["Metadata_Timepoint"] = df["time"].astype(str).str.zfill(3)
                    df = df.rename(
                        {"time": "ImageNumber", "x": "Location_Center_X", "y": "Location_Center_Y"},
                        axis=1,
                    )
                    sensitive = df[df["strategy"] == 0].copy()
                    sensitive["ObjectNumber"] = sensitive.groupby("ImageNumber").cumcount() + 1
                    sensitive["Number_Object_Number"] = sensitive["ObjectNumber"]
                    resistant = df[df["strategy"] == 1].copy()
                    resistant["ObjectNumber"] = resistant.groupby("ImageNumber").cumcount() + 1
                    resistant["Number_Object_Number"] = resistant["ObjectNumber"]
                    sensitive.to_csv(f"{out_loc}/{out_name}_locations_green.csv", index=False)
                    resistant.to_csv(f"{out_loc}/{out_name}_locations_pink.csv", index=False)
            # Add row to overview.xlsx
            overview.append(
                {
                    "Name": exp_new,
                    "Cell Type 1": "sensitive",
                    "Cell Type 2": "resistant",
                    "Fluorophore 1": "green",
                    "Fluorophore 2": "pink",
                    "Drug": f"s{exp}",
                    "Experimentalist": "User",
                    "Imaging Frequency": 4,
                    "Number of Plates": plates,
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
            # Make images directory
            if not os.path.exists(f"{data_dir}/{rep}/{out_dir}/{exp_new}/images"):
                os.mkdir(f"{data_dir}/{rep}/{out_dir}/{exp_new}/images")

        # Save ground truth
        gt_df = pd.DataFrame(ground_truth)
        gt_df["p_SS"] = config["payoff"][0]
        gt_df["p_SR"] = config["payoff"][1]
        gt_df["p_SE"] = config["payoff"][2]
        gt_df["p_RS"] = config["payoff"][3]
        gt_df["p_RR"] = config["payoff"][4]
        gt_df["p_RE"] = config["payoff"][5]
        gt_df["p_ES"] = config["payoff"][6]
        gt_df["p_ER"] = config["payoff"][7]
        gt_df["p_EE"] = config["payoff"][8]
        gt_df.to_csv(f"{data_dir}/{rep}/{out_dir}/ground_truth.csv", index=False)

        # Save overview.xlsx
        overview_df = pd.DataFrame(overview)
        overview_df["Date"] = pd.to_datetime(overview_df["Date"], format="%m/%d/%y")
        overview_df.to_excel(f"{data_dir}/{rep}/{out_dir}/overview.xlsx", index=False)

    # Save sbatch run scripts
    with open(f"{data_dir}/fit_models.sh", "w") as f:
        for rep in reps:
            for exp_name in gt_df["Experiment"].unique():
                arg_str = f"-dir {data_dir}/{rep}/{out_dir} -exp {exp_name}"
                f.write(f"{args.run_cmd} run_game_assay.py {arg_str}\n")
                f.write(f"{args.run_cmd} fit_ode.py {arg_str} -model replicator\n")
                f.write(f'{args.run_cmd} fit_ode.py {arg_str} -model "lotka-volterra"\n')


if __name__ == "__main__":
    main()
