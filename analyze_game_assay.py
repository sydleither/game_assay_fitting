import argparse
import os

import pandas as pd


def get_seeding_info(data_dir):
    df = []
    for exp_name in os.listdir(data_dir):
        if os.path.isfile(f"{data_dir}/{exp_name}") or exp_name == "layout_files":
            continue
        count_df = pd.read_csv(f"{data_dir}/{exp_name}/{exp_name}_counts_df_processed.csv")
        count_df["Experiment"] = exp_name
        df.append(count_df)
    df = pd.concat(df)
    df = df[df["DrugConcentration"] == 0.0]
    df = pd.pivot(
        df, index=["Experiment", "PlateId", "WellId", "Time"], columns="CellType", values="Count"
    )
    if len(df.columns) != 2:
        raise ValueError("Seeding density calculation assumes two cell types")
    df["Sum"] = df[df.columns[0]] + df[df.columns[1]]
    df = df.reset_index()
    df["Max Sum"] = df.groupby(["Experiment", "PlateId", "WellId"])["Sum"].transform("max")
    df["Frequency"] = df["Sum"] / df["Max Sum"]
    df = df[df["Time"] == 0]
    print("Average max density: ", df["Max Sum"].mean(), "+-", df["Max Sum"].sem())
    print("Average initial seeding density: ", df["Sum"].mean(), "+-", df["Sum"].sem())
    print(
        "Average initial seeding frequency: ", df["Frequency"].mean(), "+-", df["Frequency"].sem()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str, default="data/experimental")
    args = parser.parse_args()

    get_seeding_info(args.data_dir)


if __name__ == "__main__":
    main()
