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

## Find ideal exponential growth window stratedgy
```
python3 run_game_assay.py -dir data/experimental
python3 exponential_growth_window.py -w per_well
python3 exponential_growth_window.py -w per_exp
python3 exponential_growth_window.py -w none
python3 exponential_growth_window.py -p 1
```

## Run game assay, ODE models, and plot fits:
### On frequency-dependent ODE data
```
python3 create_ode_data.py -dir data/ode_egt -model replicator
python3 fit_ode.py -dir data/ode_egt -model replicator
python3 fit_ode.py -dir data/ode_egt -model "lotka-volterra"
python3 compare_fits.py -dir data/ode_egt
python3 compare_estimations.py -dir data/ode_egt
```
### On density-dependent ODE data
```
python3 create_ode_data.py -dir data/ode_lv -model "lotka-volterra"
python3 fit_ode.py -dir data/ode_lv -model replicator
python3 fit_ode.py -dir data/ode_lv -model "lotka-volterra"
python3 compare_fits.py -dir data/ode_lv
python3 compare_estimations.py -dir data/ode_lv
```
### On spatial density-dependent data
```
python3 create_abm_data.py -dir data/abm -samples 10
bash data/abm/raw/run0.sh
python3 format_abm_data.py -dir data/abm
bash data/abm/game_assay.sh
bash data/abm/ode_freq.sh
bash data/abm/ode_density.sh
python3 compare_fits.py -dir data/abm/formatted
python3 compare_estimations.py -dir data/abm/formatted
```
### On experimental data
```
python3 run_game_assay.py -dir data/experimental
python3 fit_ode.py -dir data/experimental -model replicator
python3 fit_ode.py -dir data/experimental -model "lotka-volterra"
python3 compare_fits.py -dir data/experimental
```