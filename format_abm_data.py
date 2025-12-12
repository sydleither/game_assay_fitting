import argparse
from datetime import date
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_type", type=str, default="data/spatial_egt")
    parser.add_argument("-in", "--in_dir", type=str, default="raw")
    parser.add_argument("-out", "--out_dir", type=str, default="formatted")
    args = parser.parse_args()

    data_dir = args.data_type
    raw_dir = args.in_dir
    out_dir = args.out_dir
    today_yyyymmdd = date.today().strftime("%Y%m%d")
    today_mmddyyyy = date.today().strftime("%m/%d/%Y")
    os.mkdir(f"{data_dir}/{out_dir}")

    # Create layout.xlsx
    os.mkdir(f"{data_dir}/{out_dir}/layout_files")
    layout = []
    for i in range(6):
        layout.append({j:0.0 for j in range(2, 12)})
    df_lo = pd.DataFrame(layout, index=["b", "c", "d", "e", "f", "g"])
    df_lo.to_excel(f"{data_dir}/{out_dir}/layout.xlsx")

    # Save cell data and create overview.xlsx
    overview = []
    for exp in os.listdir(f"{data_dir}/{raw_dir}"):
        if os.path.isfile(f"{data_dir}/{raw_dir}/{exp}"):
            continue
        exp_new = f"{today_yyyymmdd}_sensitive_green_vs_resistant_pink_s{exp}"
        plates = 0
        os.mkdir(f"{data_dir}/{out_dir}/{exp_new}")
        for plate in os.listdir(f"{data_dir}/{raw_dir}/{exp}"):
            plates += 1
            out_loc = f"{data_dir}/{out_dir}/{exp_new}/results_stitched_images_plate{plate}"
            os.mkdir(out_loc)
            for well in os.listdir(f"{data_dir}/{raw_dir}/{exp}/{plate}"):
                # Assume only 1 rep
                for rep in os.listdir(f"{data_dir}/{raw_dir}/{exp}/{plate}/{well}"):
                    rep_path = f"{data_dir}/{raw_dir}/{exp}/{plate}/{well}/{rep}"
                    if os.path.isfile(rep_path):
                        continue
                    df = pd.read_csv(f"{rep_path}/2Dcoords.csv")
                out_name = f"segmentation_results_well_{well}"
                # Save counts file
                df["time"] = pd.factorize(df["time"])[0] + 1
                counts = df.groupby(["time", "type"]).count().reset_index()
                counts = pd.pivot(counts, index="time", columns="type", values="x")
                counts = counts.reset_index()
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
                sensitive = df[df["type"] == 0].copy()
                sensitive["ObjectNumber"] = sensitive.groupby("ImageNumber").cumcount() + 1
                sensitive["Number_Object_Number"] = sensitive["ObjectNumber"]
                resistant = df[df["type"] == 1].copy()
                resistant["ObjectNumber"] = resistant.groupby("ImageNumber").cumcount() + 1
                resistant["Number_Object_Number"] = resistant["ObjectNumber"]
                sensitive.to_csv(f"{out_loc}/{out_name}_locations_green.csv", index=False)
                resistant.to_csv(f"{out_loc}/{out_name}_locations_pink.csv", index=False)
        # Add row to overview.xlsx
        overview.append({
            "Name": exp_new,
            "Cell Type 1": "sensitive",
            "Cell Type 2": "resistant",
            "Fluorophore 1": "green",
            "Fluorophore 2": "pink",
            "Drug": "None",
            "Experimentalist": "User",
            "Imaging Frequency": 4,
            "Number of Plates": plates,
            "Date": today_mmddyyyy,
            "Project": "ABM",
            "Layout File": "layout.xlsx",
            "Layout File (Original)": "layout.xlsx",
            "Location": "Local",
            "Copied?": "y",
            "Tags": ["pink", "green"],
            "Growth Rate Window": [24, 72],
            "Minimum Cell Number": 10,
            "Notes": ""
        })

    # Save overview.xlsx
    pd.DataFrame(overview).to_excel(f"{data_dir}/{out_dir}/overview.xlsx")


if __name__ == "__main__":
    main()
