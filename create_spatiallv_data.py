import argparse
import string

from EGT_HAL.config_utils import latin_hybercube_sample, write_spatiallv_config, write_run_scripts


def main():
    # Get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_type", type=str, default="data/spatial_lv")
    parser.add_argument("-exp", "--experiment_name", type=str, default="raw")
    parser.add_argument(
        "-run_cmd", "--run_command", type=str, default="java -cp build_spatiallv/:lib/* SpatialLV.SpatialLV"
    )
    parser.add_argument("-seed", "--seed", type=int, default=42)
    parser.add_argument("-samples", "--num_samples", type=int, default=10)
    parser.add_argument("-lir", "--lower_interaction_range", type=float, default=0.001)
    parser.add_argument("-uir", "--upper_interaction_range", type=float, default=0.099)
    parser.add_argument("-k", "--carrying_capacities", nargs="+", type=float, default=[5, 5])
    parser.add_argument("-x", "--grid_x", type=int, default=100)
    parser.add_argument("-y", "--grid_y", type=int, default=100)
    parser.add_argument("-m", "--interaction_radius", type=int, default=2)
    parser.add_argument("-n", "--reproduction_radius", type=int, default=2)
    parser.add_argument("-freq", "--write_freq", type=int, default=4)
    parser.add_argument("-end", "--end_time", type=int, default=80)
    args = parser.parse_args()

    lgr = args.lower_interaction_range
    ugr = args.upper_interaction_range
    capacity = args.grid_x * args.grid_y

    # Set interaction parameters
    samples = latin_hybercube_sample(
        args.num_samples,
        ["r1", "r2", "a11", "a12", "a21", "a22"],
        [lgr]*6,
        [ugr]*6,
        [False]*6,
        rnd=3,
        seed=args.seed,
    )

    # Mimic plate structure
    num_cells = int(0.1*capacity)
    seeding = [0.1, 0.3, 0.5, 0.7, 0.9]
    colids = [2, 3, 4, 5, 6]
    rowids = string.ascii_uppercase[1:4]

    # Create ABM configs
    run_output = []
    for s, sample in enumerate(samples):
        interaction_matrix = [[sample["a11"], sample["a12"]], [sample["a21"], sample["a22"]]]
        intrinsic_growths = [sample["r1"], sample["r2"]]
        for plate in [1]:
            exp_name = f"{args.experiment_name}/{s}/{plate}"
            run_str = f"{args.run_command} ../{args.data_type} {exp_name}"
            for i, fs in enumerate(seeding):
                for rep, row in enumerate(rowids):
                    config_name = f"{row}{colids[i]}"
                    write_spatiallv_config(
                        data_dir=args.data_type,
                        exp_dir=exp_name,
                        config_name=config_name,
                        seed=rep,
                        dimension=2,
                        growth_model="linear",
                        num_types=2,
                        interaction_matrix=interaction_matrix,
                        intrinsic_growths=intrinsic_growths,
                        carrying_capacities=args.carrying_capacities,
                        initial_counts=[int(fs*num_cells), int((1-fs)*num_cells)],
                        death_rates=[0.0, 0.0],
                        interaction_radius=args.interaction_radius,
                        reproduction_radius=args.reproduction_radius,
                        grid_length=args.grid_x,
                        grid_height=args.grid_y,
                        ticks=args.end_time,
                        write_freq=args.write_freq,
                    )
                    run_output.append(f"{run_str} {config_name}\n")
    write_run_scripts(args.data_type, args.experiment_name, run_output)


if __name__ == "__main__":
    main()
