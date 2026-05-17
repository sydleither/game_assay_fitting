import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from exponential_growth_window import get_ground_truth
from utils import (
    analyze_significance,
    format_for_plotting,
    get_fit_df,
    label_data_type,
    label_qualitative_dynamics,
)


def plot_accuracy(save_loc, df):
    fig, ax = plt.subplots()
    sns.barplot(
        data=df[df["Data Type"] != "Experimental"],
        x="Model",
        y="Accuracy",
        color="#8da0cb",
        ax=ax,
    )
    ax.tick_params("x", rotation=45)
    ax.set_title(
        "Qualitative Interaction Classification Accuracy\nacross Models Fit on Synthetic Data"
    )
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/accuracy_model.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_entropy(save_loc, df):
    fig, ax = plt.subplots()
    sns.barplot(
        data=df[df["Data Type"] == "Experimental"],
        x="Model",
        y="Entropy",
        color="#fc8d62",
        ax=ax,
    )
    ax.tick_params("x", rotation=45)
    ax.set_title(
        "Qualitative Interaction Classification Entropy\nacross Models Fit on Experimental Data"
    )
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/entropy_model.png", bbox_inches="tight", dpi=200)
    plt.close()


def compare_noise(save_loc, df):
    df = df[df["Data Type"].str.contains("ODE")]
    df["Noisy"] = df["Data Type"].str.contains("Noisy")
    df["Data"] = df["Data Type"].str.replace("Noisy ", "")

    for model in df["Model"].unique():
        print(f"\n{model}")
        print(analyze_significance(
            df=df[(df["Data"] == "Exponential Growth ODE") & (df["Model"] == model)], 
            group_col="Noisy", 
            value_col="Accuracy", 
            control_label=True
        ))


def get_data():
    # Read in all the data files
    if not os.path.exists("data/all.csv"):
        df = []
        for data_dir in os.listdir("data"):
            if data_dir.endswith("gr") or os.path.isfile(f"data/{data_dir}"):
                continue
            for rep in os.listdir(f"data/{data_dir}"):
                if os.path.isfile(f"data/{data_dir}/{rep}"):
                    continue
                df_exp = get_fit_df(f"data/{data_dir}/{rep}")
                df_exp = df_exp.drop_duplicates(subset=["Model", "Experiment"])
                df_exp = label_qualitative_dynamics(df_exp, ["Model", "Experiment"])
                df_exp = get_ground_truth(f"data/{data_dir}/{rep}", df_exp)
                df_exp = df_exp.drop_duplicates(subset="Model")
                df_exp["Replicate"] = rep
                df_exp["Data Type"] = label_data_type(f"data/{data_dir}")
                df.append(df_exp)
        df = pd.concat(df)
        df = format_for_plotting(df)
        df = df.reset_index()
        df.to_csv("data/all.csv", index=False)
    else:
        df = pd.read_csv("data/all.csv")
    return df


def main():
    df = get_data()
    df = df.sort_values(by=["Data Type", "Model", "Replicate", "Experiment"])

    compare_noise("data", df.copy())
    plot_accuracy("data", df)
    plot_entropy("data", df)

    # print("Synthetic Data")
    # print(analyze_significance(
    #     df=df[df["Data Type"] != "Experimental"], 
    #     group_col="Exponential Growth Window Strategy", 
    #     value_col="Accuracy", 
    #     control_label="None"
    # ))
    # print("\nExperimental Data")
    # print(analyze_significance(
    #     df=df[df["Data Type"] == "Experimental"], 
    #     group_col="Exponential Growth Window Strategy", 
    #     value_col="Entropy", 
    #     control_label="Per-Cell-Type"
    # ))


if __name__ == "__main__":
    main()
