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

## Run game assay, ODE models, and plot classification accuracy:
### On perfect ODE data
```
python3 create_ode_data.py -dir data/ode_egt -model replicator
bash data/ode_egt/fit_models.sh
python3 analyze_synthetic.py -dir data/ode_egt
```
```
python3 create_ode_data.py -dir data/ode_lv -model "lotka-volterra"
bash data/ode_lv/fit_models.sh
python3 analyze_synthetic.py -dir data/ode_lv
```
### On noisy ODE data
```
python3 create_ode_data.py -dir data/ode_egt_noisy -model replicator -noise 0.2
bash data/ode_egt_noisy/fit_models.sh
python3 analyze_synthetic.py -dir data/ode_egt_noisy
```
```
python3 create_ode_data.py -dir data/ode_lv_noisy -model "lotka-volterra" -noise 0.2
bash data/ode_lv_noisy/fit_models.sh
python3 analyze_synthetic.py -dir data/ode_lv_noisy
```
### On spatial data
```
python3 create_abm_data.py -dir data/abm_strong -m 2 -n 2
bash data/abm_strong/run0.sh
python3 format_abm_data.py -dir data/abm_strong
bash data/abm_strong/fit_models.sh
python3 analyze_synthetic.py -dir data/abm_strong/formatted
```
```
python3 create_abm_data.py -dir data/abm_medium -m 4 -n 4
bash data/abm_medium/run0.sh
python3 format_abm_data.py -dir data/abm_medium
bash data/abm_medium/fit_models.sh
python3 analyze_synthetic.py -dir data/abm_medium/formatted
```
```
python3 create_abm_data.py -dir data/abm_weak -m 6 -n 6
bash data/abm_weak/run0.sh
python3 format_abm_data.py -dir data/abm_weak
bash data/abm_weak/fit_models.sh
python3 analyze_synthetic.py -dir data/abm_weak/formatted
```
### On experimental data
'''
python3 run_game_assay.py -dir data/experimental/0 -window per_cell
python3 fit_ode.py -dir data/experimental/0 -model replicator -window per_cell
python3 fit_ode.py -dir data/experimental/0 -model "lotka-volterra" -window per_cell
python3 analyze_experimental.py -dir data/experimental
'''

## Test different exponential growth windows
```
python3 exponential_growth_window.py -in data/{data directory name}
bash data/{data directory name}_gr/exponential_growth_windows.sh
python3 exponential_growth_window.py -in data -p 1
```