from itertools import product
import os

import pandas as pd

from game_assay.game_analysis_utils import (
    compute_population_fraction,
    estimate_game_parameters,
    estimate_growth_rate,
    load_cellprofiler_data,
    map_well_to_experimental_condition,
)
from game_assay.game_archive_utils import (
    map_well_to_seeded_proportion_1_9,
    map_well_to_seeded_proportion_9_up,
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


def count_cells(data_dir, dir_curr_experiment):
    cell_count_path = os.path.join(
        data_dir,
        dir_curr_experiment,
        "%s_counts_df_processed.csv" % dir_curr_experiment.split("/")[-1],
    )
    if os.path.exists(cell_count_path):
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
            counts_df["SeededProportion_Parental"] = counts_df["WellId"].apply(
                lambda x: map_well_to_seeded_proportion_9_up(x)
            )
        else:
            experimental_conditions_df = pd.read_excel(layout_dir, header=0, index_col=0)
            counts_df["SeededProportion_Parental"] = counts_df["WellId"].apply(
                lambda x: map_well_to_seeded_proportion_1_9(x, plate_id)
            )
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
    counts_df.to_csv(cell_count_path, index=False)
    return counts_df


def calculate_payoffs(data_dir, exp_dir, growth_rate_df, cell_type_list, fraction_col):
    game_path = os.path.join(
        data_dir,
        exp_dir,
        "%s_game_params_df_processed.csv" % exp_dir.split("/")[-1],
    )
    if os.path.exists(game_path):
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
            method="theil",
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


def calculate_growth_rates(data_dir, exp_dir, counts_df, growth_rate_window, cell_type_list):
    gr_path = os.path.join(
        data_dir,
        exp_dir,
        "%s_growth_rate_df_processed.csv" % exp_dir.split("/")[-1],
    )
    if os.path.exists(gr_path):
        return pd.read_csv(gr_path)

    metadata_columns = [
        col
        for col in counts_df.columns
        if col not in ["Time", "Count", "ImageId", "CellType", "WellId", "PlateId"]
    ]
    tmp_list = []
    for plate_id, well_id, cell_type in product(
        counts_df["PlateId"].unique(), counts_df["WellId"].unique(), cell_type_list
    ):
        slope, intercept, low_slope, high_slope = estimate_growth_rate(
            data_df=counts_df[counts_df["PlateId"] == plate_id],
            well_id=well_id,
            cell_type=cell_type,
            growth_rate_window=growth_rate_window,
            count_threshold=100,
        )
        fractions_dict = compute_population_fraction(
            counts_df[counts_df["PlateId"] == plate_id],
            well_id=well_id,
            fraction_window=growth_rate_window,
            n_images="all",
            cell_type_list=cell_type_list,
        )
        curr_df = counts_df[
            (counts_df["PlateId"] == plate_id)
            & (counts_df["WellId"] == well_id)
            & (counts_df["CellType"] == cell_type)
        ]
        tmp_list.append(
            {
                "PlateId": plate_id,
                "WellId": well_id,
                "CellType": cell_type,
                **fractions_dict,
                **curr_df[metadata_columns].iloc[0].to_dict(),
                "GrowthRate": slope,
                "GrowthRate_lowerBound": low_slope,
                "GrowthRate_higherBound": high_slope,
                "Intercept": intercept,
                "GrowthRate_normalised": slope,
                "GrowthRate_lowerBound_normalised": low_slope,
                "GrowthRate_higherBound_normalised": high_slope,
            }
        )
    growth_rate_df = pd.DataFrame(tmp_list)
    growth_rate_df.to_csv(gr_path, index=False)
    return growth_rate_df
