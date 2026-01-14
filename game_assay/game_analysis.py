from itertools import product
import os

import numpy as np
import pandas as pd

from game_assay.game_analysis_utils import (
    estimate_game_parameters,
    estimate_growth_rate,
    load_cellprofiler_data,
    map_well_to_experimental_condition,
    optimize_growth_rate_window,
)


def read_overview_xlsx(data_dir, exp_name):
    exp_parts = exp_name.split("_")
    overview_df = pd.read_excel(f"{data_dir}/overview.xlsx")
    overview_df["Date"] = overview_df["Date"].dt.strftime("%y%m%d")
    experiment = overview_df[
        (overview_df["Date"] == exp_parts[0])
        & (overview_df["Cell Type 1"] == exp_parts[1])
        & (overview_df["Fluorophore 1"] == exp_parts[2].lower())
        & (overview_df["Cell Type 2"] == exp_parts[4])
        & (overview_df["Fluorophore 2"] == exp_parts[5])
        & (overview_df["Drug"] == exp_parts[6])
    ].to_dict("records")[0]
    return experiment


def calculate_counts(data_dir, dir_curr_experiment, rewrite=False):
    cell_count_path = os.path.join(
        data_dir,
        dir_curr_experiment,
        "%s_counts_df_processed.csv" % dir_curr_experiment.split("/")[-1],
    )
    if os.path.exists(cell_count_path) and not rewrite:
        return pd.read_csv(cell_count_path)

    experiment = read_overview_xlsx(data_dir, dir_curr_experiment)

    # Assemble the count file
    dir_curr_experiment = os.path.join(data_dir, dir_curr_experiment)
    layout_dir = os.path.join(f"{data_dir}/layout_files/" + experiment["Layout File"])
    n_plates = int(experiment["Number of Plates"])
    tmp_list_experiment = []
    for plate_id in range(1, n_plates + 1):
        raw_counts_dir = os.path.join(
            dir_curr_experiment, "results_stitched_images_plate%d" % plate_id
        )
        file_list = [
            f
            for f in os.listdir(raw_counts_dir)
            if f.endswith(".csv") and "well" in f and "results.csv" in f
        ]
        tmp_list = []
        for file in file_list:
            curr_source_file = os.path.join(raw_counts_dir, file)
            tmp_df = pd.read_csv(curr_source_file)
            tmp_df["WellId"] = file.split("_well_")[1].split("_")[0]
            tmp_list.append(tmp_df)
        raw_counts_df = pd.concat(tmp_list)

        # Annotate with metadata
        pop_names = [
            "-".join([experiment["Cell Type %d" % x], experiment["Fluorophore %d" % x]])
            for x in range(1, 3)
        ]
        tags = experiment["Tags"].strip("[]',").split("', '")
        counts_df = load_cellprofiler_data(
            raw_counts_df,
            imaging_frequency=experiment["Imaging Frequency"],
            ignore_column=None,
            tags=tags,
            pop_names=pop_names,
        )
        counts_df["PlateId"] = plate_id
        counts_df["ReplicateId"] = counts_df["RowId"].apply(
            lambda x: {"B": 1, "C": 2, "D": 3, "E": 1, "F": 2, "G": 3}[x]
        )
        # Add metadata from the layout file
        if "exp_9+" in experiment["Layout File"]:
            # For these experiments each plate has a different drug concentrations
            experimental_conditions_df = pd.read_excel(
                layout_dir,
                header=0,
                index_col=0,
                sheet_name="Plate%d" % plate_id,
            )
        else:
            experimental_conditions_df = pd.read_excel(layout_dir, header=0, index_col=0)
        # Format index and column names to account for spelling/spacing differences
        experimental_conditions_df.index = experimental_conditions_df.index.str.replace(" ", "")
        experimental_conditions_df.index = experimental_conditions_df.index.str.lower()
        experimental_conditions_df.columns = experimental_conditions_df.columns.astype(int)

        # Map each well to the experimental condition
        counts_df["DrugConcentration"] = counts_df["WellId"].apply(
            lambda x: map_well_to_experimental_condition(x, experimental_conditions_df)
        )
        counts_df["Drug"] = experiment["Drug"]
        tmp_list_experiment.append(counts_df)

    counts_df = pd.concat(tmp_list_experiment)
    counts_df["Minimum Cell Number"] = experiment["Minimum Cell Number"]
    counts_df.to_csv(cell_count_path, index=False)
    return counts_df


def calculate_growth_rates(
    data_dir,
    exp_dir,
    counts_df,
    growth_rate_window=None,
    cell_type_list=None,
    rewrite=False,
):
    gr_path = os.path.join(
        data_dir,
        exp_dir,
        "%s_growth_rate_df_processed.csv" % exp_dir.split("/")[-1],
    )
    if os.path.exists(gr_path) and not rewrite:
        return pd.read_csv(gr_path)

    if cell_type_list is None:
        raise ValueError("Set the cell type list.")

    metadata_columns = [
        col
        for col in counts_df.columns
        if col not in ["Time", "Count", "ImageId", "CellType", "WellId", "PlateId"]
    ]
    count_threshold = counts_df["Minimum Cell Number"].iloc[0]
    tmp_list = []

    # Calculate growth rate window
    if growth_rate_window is None:
        counts_df = counts_df.groupby("PlateId", group_keys=False)[
            counts_df.columns
        ].apply(optimize_growth_rate_window)
    else:
        counts_df["GrowthRate_window_start"] = growth_rate_window[0]
        counts_df["GrowthRate_window_end"] = growth_rate_window[1]

    for plate_id, well_id, cell_type in product(
        counts_df["PlateId"].unique(), counts_df["WellId"].unique(), cell_type_list
    ):
        curr_df = counts_df[
            (counts_df["PlateId"] == plate_id)
            & (counts_df["WellId"] == well_id)
            & (counts_df["CellType"] == cell_type)
        ]
        growth_rate_window = (
            curr_df["GrowthRate_window_start"].values[0],
            curr_df["GrowthRate_window_end"].values[0],
        )
        # Quality control data
        if (
            curr_df["Count"].min() <= 0
            or curr_df["Count"].mean() < count_threshold
            or np.isnan(growth_rate_window[0])
        ):
            slope, intercept, low_slope, high_slope, error = np.nan, np.nan, np.nan, np.nan, np.nan
        # Estimate growth rate
        else:
            slope, intercept, low_slope, high_slope, error = estimate_growth_rate(
                data_df=counts_df[counts_df["PlateId"] == plate_id],
                well_id=well_id,
                cell_type=cell_type,
                growth_rate_window=growth_rate_window,
            )
        # Add initial frequency of cell types
        initial_freq = counts_df[
            (counts_df["Time"] == 0)
            & (counts_df["PlateId"] == plate_id)
            & (counts_df["WellId"] == well_id)
        ]
        fractions = {}
        for ct, freq in initial_freq[["CellType", "Frequency"]].values:
            fractions[f"Fraction_{ct}"] = freq
        # Compile growth_rate_df row
        tmp_list.append(
            {
                "PlateId": plate_id,
                "WellId": well_id,
                "CellType": cell_type,
                **fractions,
                **curr_df[metadata_columns].iloc[0].to_dict(),
                "GrowthRate": slope,
                "GrowthRate_lowerBound": low_slope,
                "GrowthRate_higherBound": high_slope,
                "Intercept": intercept,
                "GrowthRate_window_start": growth_rate_window[0],
                "GrowthRate_window_end": growth_rate_window[1],
                "GrowthRate_error": error,
            }
        )
    growth_rate_df = pd.DataFrame(tmp_list)
    growth_rate_df.to_csv(gr_path, index=False)
    return growth_rate_df


def calculate_payoffs(
    data_dir, exp_dir, growth_rate_df, cell_type_list, fraction_col, rewrite=False
):
    game_path = os.path.join(
        data_dir,
        exp_dir,
        "%s_game_params_df_processed.csv" % exp_dir.split("/")[-1],
    )
    if os.path.exists(game_path) and not rewrite:
        return pd.read_csv(game_path)

    tmp_list = []
    for drug_concentration in growth_rate_df["DrugConcentration"].unique():
        curr_data_df = growth_rate_df[(growth_rate_df["DrugConcentration"] == drug_concentration)]
        game_params_dict = estimate_game_parameters(
            growth_rate_df=curr_data_df,
            fraction_col=fraction_col,
            growth_rate_col="GrowthRate",
            cell_type_col="CellType",
            cell_type_list=cell_type_list,
            ci=0.95,
        )
        tmp_list.append(
            {
                "DrugConcentration": float(drug_concentration),
                "Type1": cell_type_list[0],
                "Type2": cell_type_list[1],
                **game_params_dict,
            }
        )
    game_params_df = pd.DataFrame(tmp_list)
    game_params_df.to_csv(game_path, index=False)
    return game_params_df


def calculate_locations(data_dir, exp_dir, counts_df, drug_concentration=0, rewrite=False):
    loc_path = os.path.join(
        data_dir,
        exp_dir,
        "%s_locations_df_processed.csv" % exp_dir.split("/")[-1],
    )
    if os.path.exists(loc_path) and not rewrite:
        return pd.read_csv(loc_path)

    # Add experiment info to dataframe
    experiment = read_overview_xlsx(data_dir, exp_dir)
    counts_df["Name"] = exp_dir
    counts_df["Imaging Frequency"] = experiment["Imaging Frequency"]
    counts_df["Fluorophore 1"] = experiment["Fluorophore 1"]
    mcherry = "mCherry" if experiment["Fluorophore 2"] == "mcherry" else experiment["Fluorophore 2"]
    counts_df["Fluorophore 2"] = mcherry
    counts_df["Cell Type 1"] = experiment["Cell Type 1"]
    counts_df["Cell Type 2"] = experiment["Cell Type 2"]
    # Make an overview data frame
    counts_df = counts_df.drop(columns=["Time", "ImageId", "Frequency", "Count", "CellType"])
    overview_df_spatial_data = counts_df.drop_duplicates().copy()
    # Add the locations of the cell type files
    overview_df_spatial_data["Locations Cell Type 1"] = overview_df_spatial_data.apply(
        lambda x: os.path.join(
            x["Name"],
            "results_stitched_images_plate%d" % x["PlateId"],
            "segmentation_results_well_%s_locations_%s.csv" % (x["WellId"], x["Fluorophore 1"]),
        ),
        axis=1,
    )
    overview_df_spatial_data["Locations Cell Type 2"] = overview_df_spatial_data.apply(
        lambda x: os.path.join(
            x["Name"],
            "results_stitched_images_plate%d" % x["PlateId"],
            "segmentation_results_well_%s_locations_%s.csv" % (x["WellId"], x["Fluorophore 2"]),
        ),
        axis=1,
    )
    # Save the overview data frame
    overview_df_spatial_data = overview_df_spatial_data[
        overview_df_spatial_data["DrugConcentration"] == drug_concentration
    ]
    overview_df_spatial_data.reset_index(drop=True, inplace=True)
    overview_df_spatial_data.to_csv(loc_path, index=False)
    return overview_df_spatial_data
