"""
Script to run image segmentation using cell profiler
"""

import argparse
from functools import partial
import os
import shutil
import tqdm
import multiprocess

from game_assay.game_analysis_utils import run_cellprofiler_on_well

# ==================================== Parameters ====================================
parser = argparse.ArgumentParser()
parser.add_argument("-dir", "--data_dir", type=str, default="data/experimental")
parser.add_argument("-out", "--output_dir", type=str, default="data/experimental")
parser.add_argument("-cell", "--cellprofiler_path", type=str)
parser.add_argument("-temp", "--local_tmp_dir", type=str, default="data/temp")
args = parser.parse_args()

abs_path = os.path.dirname(os.path.realpath(__file__)).replace(" ", "\ ")+"/"

data_dir = os.path.join(abs_path+args.data_dir)
output_dir = os.path.join(abs_path+args.output_dir)
cellprofiler_path = os.path.join(args.cellprofiler_path)
local_tmp_dir = os.path.join(abs_path+args.local_tmp_dir)

pipeline_file = os.path.join("./analysis_positions.cppipe")
n_cores = 4
dirs_to_analyse = {  # Directories to analyse. Key is the directory name, value is the plate ID
    "plate1": "stitched_images"
}


# ==================================== Functions ====================================
# ------------------------------------ Move and process files ------------------------------------
def process_files(
    well_id,
    plate_id,
    source_dir,
    local_tmp_dir="./",
    target_dir_outlines=None,
    target_dir_results=None,
    cellprofiler_path=cellprofiler_path,
    pipeline_file="./",
    var_name_well="Metadata_Well",
):
    """
    Move the files from the biospa organisation into a more organised format and run cellprofiler on them.
    source_dir: Directory with files in biospa organisation. This should point to the directory for this plate (231010_150258_MD_DS_032822_GT_033123_10-Oct-2023 13-59-45)
    well_id: Well to process
    plate_id: Plate to process
    local_tmp_dir: Local directory to use for temporary files
    organise_files: Whether to organise the files into a more organised format
    cellprofiler_path: Path to cellprofiler
    pipeline_file: Path to cellprofiler pipeline file
    var_name_well: Variable name for the well in the cellprofiler pipeline
    """
    # Skip if already processed this file
    if os.path.exists(
        os.path.join(target_dir_results, "segmentation_results_well_{}_results.csv".format(well_id))
    ):
        return

    # Setup local temporary directory
    curr_tmp_dir = os.path.join(local_tmp_dir, "tmp_" + plate_id, well_id)
    os.makedirs(curr_tmp_dir, exist_ok=True)

    # Run cellprofiler on local directory
    run_cellprofiler_on_well(
        well_id=well_id,
        cellprofiler_path=cellprofiler_path,
        pipeline_file=pipeline_file,
        dir_to_analyse=source_dir,
        output_dir=curr_tmp_dir,
        var_name_well=var_name_well,
        create_out_dir=False,
        print_command=False,
        log_level=10,
        suppress_output=False,
    )  # xxx

    # Move outlines
    os.makedirs(target_dir_outlines, exist_ok=True)
    image_list = [x for x in os.listdir(curr_tmp_dir) if x.split(".")[-1] == "png"]
    image_list = [
        x for x in image_list if int(x.split("_")[5]) % 3 == 0
    ]  # only save some of the outlines for validation # xxx
    for image in image_list:
        image_to_move = os.path.join(curr_tmp_dir, image)
        image_to_make = os.path.join(target_dir_outlines, image)
        if not os.path.exists(image_to_make):
            shutil.copyfile(image_to_move, image_to_make)
        os.remove(image_to_move)

    # Move results file
    for file_name in ["results", "locations_gfp", "locations_mCherry"]:
        source_file = os.path.join(curr_tmp_dir, "%s.csv" % file_name)
        target_file = os.path.join(
            target_dir_results, "segmentation_results_well_{}_{}.csv".format(well_id, file_name)
        )
        shutil.copyfile(source_file, target_file)
        os.remove(source_file)

    # Remove temporary directory
    # shutil.rmtree(curr_tmp_dir)


# ==================================== Main ====================================
if __name__ == "__main__":
    # Loop through each directory and run cellprofiler to process the images
    for plate_id, curr_dir in dirs_to_analyse.items():
        # Get a list of all wells in this directory
        curr_source_dir = os.path.join(data_dir, plate_id, curr_dir)
        wells_to_analyse = [
            x.split("_")[0] for x in os.listdir(curr_source_dir) if x.split(".")[-1] == "tif"
        ]
        wells_to_analyse = list(set(wells_to_analyse))  # find unique values in the list

        # Run cellprofiler on each well in parallel
        curr_output_dir_images = os.path.join(data_dir, plate_id, curr_dir + "_outlines")
        curr_output_dir_results = os.path.join(data_dir, plate_id, curr_dir + "_results")
        os.makedirs(curr_output_dir_images, exist_ok=True)
        os.makedirs(curr_output_dir_results, exist_ok=True)
        pool = multiprocess.Pool(processes=n_cores)
        func = partial(
            process_files,
            plate_id=plate_id,
            source_dir=curr_source_dir,
            local_tmp_dir=local_tmp_dir,
            target_dir_outlines=curr_output_dir_images,
            target_dir_results=curr_output_dir_results,
            cellprofiler_path=cellprofiler_path,
            pipeline_file=pipeline_file,
            var_name_well="Metadata_Well",
        )

        list(tqdm(pool.imap(func, wells_to_analyse), total=len(wells_to_analyse)))

        # # Collate results files into one single results file
        # target_dir_outlines = curr_output_dir
        # if organise_files:
        #     target_dir_individual = os.path.join(curr_output_dir, "individual_images")
        #     target_dir_stitched = os.path.join(curr_output_dir, "stitched_images")
        #     target_dir_outlines = target_dir_individual+"_outlines" if type == "individual" else target_dir_stitched+"_outlines"
        # file_list = [x for x in os.listdir(target_dir_outlines) if x.split(".")[-1] == "csv"]
        # tmp_list = []
        # for file in file_list:
        #     curr_source_file = os.path.join(target_dir_outlines, file)
        #     tmp_df = pd.read_csv(curr_source_file)
        #     tmp_df["well_id"] = file.split("_")[-1].split(".")[0]
        #     tmp_list.append(tmp_df)
        #     # os.remove(curr_source_file)
        # results_df = pd.concat(tmp_list)
        # results_df.to_csv(os.path.join(curr_output_dir, "segmentation_results_{}.csv".format(image_type)), index=False)

        # # Copy file to dropbox
        # os.makedirs(dropbox_dir, exist_ok=True)
        # shutil.copyfile(os.path.join(curr_output_dir, "segmentation_results_{}.csv".format(image_type)),
        #                 os.path.join(dropbox_dir, "segmentation_results_{}_{}.csv".format(plate_id, image_type)))

        # Remove temporary directory
        try:
            shutil.rmtree(os.path.join(local_tmp_dir, "tmp_" + plate_id))
        except:
            pass
