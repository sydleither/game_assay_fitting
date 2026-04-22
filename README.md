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
python3 exponential_growth_window.py -w per_cell
python3 exponential_growth_window.py -w per_well
python3 exponential_growth_window.py -w per_exp
python3 exponential_growth_window.py -w none
python3 exponential_growth_window.py -p 1
```

## Run game assay, ODE models, and plot classification accuracy:
### On frequency-dependent ODE data
```
python3 create_ode_data.py -dir data/ode_egt -model replicator
python3 fit_ode.py -dir data/ode_egt -model replicator
python3 fit_ode.py -dir data/ode_egt -model "lotka-volterra"
python3 compare_estimations.py -dir data/ode_egt
```
### On density-dependent ODE data
```
python3 create_ode_data.py -dir data/ode_lv -model "lotka-volterra"
python3 fit_ode.py -dir data/ode_lv -model replicator
python3 fit_ode.py -dir data/ode_lv -model "lotka-volterra"
python3 compare_estimations.py -dir data/ode_lv
```
### On spatial data
```
python3 create_abm_data.py -dir data/abm_strong -m 2 -n 2
bash data/abm_strong/raw/run0.sh
python3 format_abm_data.py -dir data/abm_strong
bash data/abm_strong/game_assay.sh
bash data/abm_strong/ode_freq.sh
bash data/abm_strong/ode_density.sh
python3 compare_fits.py -dir data/abm_strong/formatted
python3 compare_estimations.py -dir data/abm_strong/formatted
```
```
python3 create_abm_data.py -dir data/abm_medium -m 4 -n 4
bash data/abm_medium/raw/run0.sh
python3 format_abm_data.py -dir data/abm_medium
bash data/abm_medium/game_assay.sh
bash data/abm_medium/ode_freq.sh
bash data/abm_medium/ode_density.sh
python3 compare_fits.py -dir data/abm_medium/formatted
python3 compare_estimations.py -dir data/abm_medium/formatted
```
```
python3 create_abm_data.py -dir data/abm_weak -m 6 -n 6
bash data/abm_weak/raw/run0.sh
python3 format_abm_data.py -dir data/abm_weak
bash data/abm_weak/game_assay.sh
bash data/abm_weak/ode_freq.sh
bash data/abm_weak/ode_density.sh
python3 compare_fits.py -dir data/abm_weak/formatted
python3 compare_estimations.py -dir data/abm_weak/formatted
```
