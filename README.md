# game_assay_fitting
Testing the fit of different models on experimental game assay data

## Get cell counts and positions from images
Download [CellProfiler](https://cellprofiler.org/) and create a python environment with requirements.txt.

Replicate the following data directory structure:

```
└── data
    └── experimental
        ├── overview.xlsx
        ├── layout_files
        └── {experiment name}
            └── plate1
                └── stitched_images
            └── plate{i}
                └── stitched_images
```

Within data/experimental/{experiment name}/plate{i}/stitched_images are the stitched bright field images of plate{i}.

To label the cell positions, run `python3 run_segmentation_laptop.py -cell {cell profiler path}`.

## Structure cell counts and positions
With the post-cellprofiler data, replicate this file structure:

```
└── data
    └── experimental
        ├── overview.xlsx
        ├── layout_files
        └── {experiment name}
            └── results_stitched_images_plate{i}
                └── segmentation_results_well_{...}_locations_{...}.csv
```

## Run game assay on experimental data
Set up virtual environment

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run game assay on all experiments
```
python3 run_game_assay.py
```

Run game assay on an individual experiment and get specific plots
```
python3 run_game_assay.py -exp {experiment name}
```

