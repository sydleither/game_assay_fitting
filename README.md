# game_assay_fitting
Testing the fit of different models on ode, abm, and experimental game assay data

## Set up virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Process experimental game assay data
### Get cell counts and positions from images
Download [CellProfiler](https://cellprofiler.org/).

Replicate the following data directory structure:

```
└── data
    └── experimental
        ├── overview.xlsx
        ├── layout_files
        └── {experiment name}
            └── plate{i}
                └── stitched_images
```

Within `data/experimental/{experiment name}/plate{i}/stitched_images` are the stitched bright field images of plate{i}.

To label the cell positions, run `python3 run_segmentation_laptop.py -cell {cell profiler path}`.

### Structure cell counts and positions
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

## Run game assay, ODE models, and plot fits
### On EGT ODE data
```
python3 create_ode_data.py -dir data/ode_egt -model replicator
python3 fit_ode.py -dir data/ode_egt -model replicator
python3 fit_ode.py -dir data/ode_egt -model "lotka-volterra"
python3 compare_fits.py -dir data/ode_egt
python3 compare_estimations.py -dir data/ode_egt
```
### On LV ODE data
```
python3 create_ode_data.py -dir data/ode_lv -model "lotka-volterra"
python3 fit_ode.py -dir data/ode_lv -model replicator
python3 fit_ode.py -dir data/ode_lv -model "lotka-volterra"
python3 compare_fits.py -dir data/ode_lv
python3 compare_estimations.py -dir data/ode_lv
```
### On EGT ABM data
```
python3 create_spatialegt_data.py -dir data/spatial_egt -samples 10
python3 format_abm_data.py -dir data/spatial_egt
python3 run_game_assay.py -dir data/spatial_egt/formatted
python3 fit_ode.py -dir data/spatial_egt/formatted -model replicator
python3 fit_ode.py -dir data/spatial_egt/formatted -model "lotka-volterra"
python3 compare_fits.py -dir data/spatial_egt/formatted
python3 compare_estimations.py -dir data/spatial_egt/formatted
```
### On LV ABM data
```
python3 create_spatiallv_data.py -dir data/spatial_lv -samples 10
python3 format_abm_data.py -dir data/spatial_lv
python3 run_game_assay.py -dir data/spatial_lv/formatted
python3 fit_ode.py -dir data/spatial_lv/formatted -model replicator
python3 fit_ode.py -dir data/spatial_lv/formatted -model "lotka-volterra"
python3 compare_fits.py -dir data/spatial_lv/formatted
```
### On experimental data
```
python3 run_game_assay.py -dir data/experimental
python3 fit_ode.py -dir data/experimental -model replicator
python3 fit_ode.py -dir data/experimental -model "lotka-volterra"
python3 compare_fits.py -dir data/experimental
```