import argparse

from EGT_HAL.config_utils import latin_hybercube_sample, write_config, write_run_scripts
from utils import get_plate_structure


def main():
    # Get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_type", type=str, default="data/abm")
    parser.add_argument("-exp", "--experiment_name", type=str, default="raw")
    parser.add_argument(
        "-run_cmd",
        "--run_command",
        type=str,
        default="java -cp build/:lib/* SpatialEGT.SpatialEGT",
    )
    parser.add_argument("-seed", "--seed", type=int, default=42)
    parser.add_argument("-samples", "--num_samples", type=int, default=10)
    parser.add_argument("-lgr", "--lower_param_range", type=float, default=0.01)
    parser.add_argument("-ugr", "--upper_param_range", type=float, default=0.099)
    parser.add_argument("-x", "--grid_x", type=int, default=100)
    parser.add_argument("-y", "--grid_y", type=int, default=100)
    parser.add_argument("-m", "--interaction_radius", type=int, default=2)
    parser.add_argument("-n", "--reproduction_radius", type=int, default=2)
    parser.add_argument("-freq", "--write_freq", type=int, default=4)
    parser.add_argument("-end", "--end_time", type=int, default=80)
    parser.add_argument("-init", "--init_freq", type=int, default=0.01)
    args = parser.parse_args()

    lpr = args.lower_param_range
    upr = args.upper_param_range
    init_count = args.init_freq * args.grid_x * args.grid_y

    # Set interaction parameters
    samples = latin_hybercube_sample(
        args.num_samples,
        ["r_0", "r_1", "A_00", "A_01", "A_10", "A_11"],
        [0.0, 0.0, -0.0001, -0.0001, -0.0001, -0.0001],
        [0.2, 0.2, 0.0, 0.0001, 0.0001, 0.0],
        [False] * 6,
        rnd=6,
        seed=args.seed,
    )

    # Mimic plate structure
    seeding, colids, rowids = get_plate_structure()

    # Create ABM configs
    run_output = []
    for s, sample in enumerate(samples):
        # Save configs in data directory structure mimicing game assay's
        data_dir = f"{args.data_type}/{args.experiment_name}"
        for plate in [1]:
            for i, fs in enumerate(seeding):
                for j, row in enumerate(rowids):
                    save_loc = f"{data_dir}/{s}/{plate}/{row}{colids[i]}"
                    write_config(
                        save_loc=save_loc,
                        seed=j,
                        num_types=2,
                        interaction_matrix=[
                            [sample["A_00"], sample["A_01"]],
                            [sample["A_10"], sample["A_11"]],
                        ],
                        intrinsic_growths=[sample["r_0"], sample["r_1"]],
                        initial_counts=[fs * init_count, (1 - fs) * init_count],
                        grid_length=args.grid_x,
                        grid_height=args.grid_y,
                        interaction_radius=args.interaction_radius,
                        reproduction_radius=args.reproduction_radius,
                        ticks=args.end_time,
                        write_freq=args.write_freq,
                        dimension=2,
                    )
                    run_output.append(f"{args.run_command} ../{save_loc}\n")
    write_run_scripts(args.data_type, args.experiment_name, run_output)


if __name__ == "__main__":
    main()
