import argparse
import string

from EGT_HAL.config_utils import latin_hybercube_sample, write_spatialegt_config, write_run_scripts


def main():
    # Get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_type", type=str, default="data/spatial_egt")
    parser.add_argument("-exp", "--experiment_name", type=str, default="raw")
    parser.add_argument(
        "-run_cmd", "--run_command", type=str, default="java -cp build_spatialegt/:lib/* SpatialEGT.SpatialEGT"
    )
    parser.add_argument("-seed", "--seed", type=int, default=42)
    parser.add_argument("-samples", "--num_samples", type=int, default=None)
    parser.add_argument("-lgr", "--lower_game_range", type=float, default=0.001)
    parser.add_argument("-ugr", "--upper_game_range", type=float, default=0.099)
    parser.add_argument("-x", "--grid_x", type=int, default=100)
    parser.add_argument("-y", "--grid_y", type=int, default=100)
    parser.add_argument("-m", "--interaction_radius", type=int, default=2)
    parser.add_argument("-n", "--reproduction_radius", type=int, default=2)
    parser.add_argument("-freq", "--write_freq", type=int, default=4)
    parser.add_argument("-end", "--end_time", type=int, default=80)
    args = parser.parse_args()

    lgr = args.lower_game_range
    ugr = args.upper_game_range
    capacity = args.grid_x * args.grid_y

    # Set interaction parameters
    if args.num_samples is None:
        samples = [
            {"A": ugr, "B": ugr, "C": lgr, "D": lgr},
            {"A": lgr, "B": ugr, "C": ugr, "D": lgr},
            {"A": ugr, "B": lgr, "C": lgr, "D": ugr},
            {"A": lgr, "B": lgr, "C": ugr, "D": ugr},
        ]
    else:
        samples = latin_hybercube_sample(
            args.num_samples,
            ["A", "B", "C", "D"],
            [lgr, lgr, lgr, lgr],
            [ugr, ugr, ugr, ugr],
            [False, False, False, False],
            rnd=3,
            seed=args.seed,
        )

    # Mimic plate structure
    seeding = [0.1, 0.3, 0.5, 0.7, 0.9]
    colids = [2, 3, 4, 5, 6]
    rowids = string.ascii_uppercase[1:4]

    # Create ABM configs
    run_output = []
    for s, sample in enumerate(samples):
        payoff = [sample["A"], sample["B"], sample["C"], sample["D"]]
        for plate in [1]:
            exp_name = f"{args.experiment_name}/{s}/{plate}"
            run_str = f"{args.run_command} ../{args.data_type} {exp_name}"
            for i, fs in enumerate(seeding):
                for rep, row in enumerate(rowids):
                    config_name = f"{row}{colids[i]}"
                    write_spatialegt_config(
                        args.data_type,
                        exp_name,
                        config_name,
                        rep,
                        payoff,
                        int(0.1 * capacity),
                        float(1 - fs),
                        x=args.grid_x,
                        y=args.grid_y,
                        interaction_radius=args.interaction_radius,
                        reproduction_radius=args.reproduction_radius,
                        turnover=0.0,
                        write_freq=args.write_freq,
                        ticks=args.end_time,
                    )
                    run_output.append(f"{run_str} {config_name} 2D {rep}\n")
    write_run_scripts(args.data_type, args.experiment_name, run_output)


if __name__ == "__main__":
    main()
