import os
import pandas as pd

# ------------------------------------ Organise Metadata for Dag's experiments ------------------------------------
# Define a function to map each well to a seeding proportion (based on dag's plate layout)
def map_well_to_seeded_proportion_1_9(well_id, plate_id):
    row_id = well_id[0]
    seeding_fraction_list = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1]
    well_in_bottom_half = row_id in ["E", "F", "G"]
    return seeding_fraction_list[(plate_id-1)*2 + well_in_bottom_half]

def map_well_to_seeded_proportion_9_up(well_id):
    col_id = int(well_id[1:])
    seeding_fraction_list = [0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1]
    return seeding_fraction_list[col_id-2]

# ------------------------------------ Load spatial data ------------------------------------
def load_spatial_data(row_df, data_dir):
    '''
    Load spatial data from a single well from the data archive. Assumes that 
    we inpute the row from the overview file that corresponds to the well, which will
    contain all the metadata necessary to load the data.
    '''
    tmp_list = []    
    for i, file in enumerate(row_df[['Locations Cell Type 1', 'Locations Cell Type 2']]):
        curr_source_file = os.path.join(data_dir, file)
        tmp_df = pd.read_csv(curr_source_file)
        # Format data
        population_id = i+1
        # cell_type = "-".join([row_df["Cell Type %d"%population_id], row_df["Fluorophore %d"%population_id]])
        cell_type = row_df["Cell Type %d"%population_id]
        tmp_df = tmp_df[["ObjectNumber", "Metadata_Timepoint", "Metadata_Well", "Location_Center_X", "Location_Center_Y", "Location_Center_Z"]].copy()
        tmp_df.rename(columns={"ObjectNumber": "CellId", "Metadata_Timepoint": "Time_index", "Metadata_Well": "WellId"}, inplace=True)
        tmp_df["Time_hours"] = (tmp_df["Time_index"] - 1) * row_df['Imaging Frequency']
        tmp_df["CellType"] = cell_type
        tmp_df["ExperimentName"] = row_df["ExperimentName"]
        tmp_df["PlateId"] = row_df["PlateId"]
        tmp_df["SeededProportion_Parental"] = row_df["SeededProportion_Parental"]
        tmp_df["Drug"] = row_df["Drug"]
        tmp_df["WellId"] = row_df["WellId"]
        tmp_df["DrugConcentration"] = row_df["DrugConcentration"]
        # Check that z-column is all 0 (only 2D data); if so, drop it
        if tmp_df["Location_Center_Z"].sum() == 0:
            tmp_df.drop(columns=["Location_Center_Z"], inplace=True)
        else:
            raise ValueError("Z-column is not zero")
        tmp_list.append(tmp_df)
    return pd.concat(tmp_list)
