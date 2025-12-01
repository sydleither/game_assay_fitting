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

Within {experiment name}/plate{i}/stitched_images are the stitched bright field images of plate{i}.

To label the cell positions, run `python3 run_segmentation_laptop.py -cell {cell profiler path}`.