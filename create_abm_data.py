import argparse
import os
import random

from utils import get_plate_structure
from EGT_ABM.utils import create_run_cmd, latin_hypercube_sample


def sample_three_strategy(seed, num_samples):
    samples = latin_hypercube_sample(
        num_samples,
        ["P_00", "P_01", "P_02", "P_10", "P_11", "P_12"],
        [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [False] * 6,
        seed=seed,
    )
    return samples


def sample_two_strategy(seed, num_samples):
    samples = latin_hypercube_sample(
        num_samples,
        ["P_00", "P_01", "P_10", "P_11"],
        [0.0] * 4,
        [0.1] * 4,
        [False] * 4,
        seed=seed,
    )
    return samples


def main():
    # Get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_dir", type=str, default="data/abm")
    parser.add_argument("-exp", "--experiment_name", type=str, default="raw")
    parser.add_argument("-run_cmd", "--run_cmd", type=str, default="python3")
    parser.add_argument("-reps", "--reps", type=int, default=10)
    parser.add_argument("-strats", "--strategies", type=int, default=3, choices=[3])
    parser.add_argument("-samples", "--num_samples", type=int, default=20)
    parser.add_argument("-l", "--grid", type=int, default=100)
    parser.add_argument("-r", "--radius", type=int, default=1)
    parser.add_argument("-write", "--write_freq", type=int, default=4)
    parser.add_argument("-steps", "--steps", type=int, default=80)
    args = parser.parse_args()

    # Mimic plate structure
    seeding, colids, rowids = get_plate_structure()

    # Create ABM configs
    run_output = []
    for rep in range(args.reps):
        # Set interaction parameters
        if args.strategies == 2:
            samples = sample_two_strategy(rep, args.num_samples)
        else:
            samples = sample_three_strategy(rep, args.num_samples)
        # Save configs in data directory structure mimicing game assay's
        for s, sample in enumerate(samples):
            data_dir = f"{args.data_dir}/{rep}/{args.experiment_name}"
            for plate in [1]:
                for i, fs in enumerate(seeding):
                    for j, row in enumerate(rowids):
                        save_loc = f"{data_dir}/{s}/{plate}/{row}{colids[i]}"
                        if args.strategies == 2:
                            init_freq = [fs, 1 - fs]
                        else:
                            init_empty = random.uniform(0.9, 0.95)
                            init_freq = [
                                (1 - init_empty) * fs,
                                (1 - init_empty) * (1 - fs),
                                init_empty,
                            ]
                            sample["P_20"] = 0.0
                            sample["P_21"] = 0.0
                            sample["P_22"] = 0.0
                        sample_output = create_run_cmd(
                            save_loc,
                            args.run_cmd,
                            j,
                            sample,
                            args.strategies,
                            init_freq,
                            args.grid,
                            args.radius,
                            args.write_freq,
                            args.steps,
                            "EGT_ABM"
                        )
                        run_output.append(sample_output)
                        os.makedirs(save_loc)
    # Run bash script with commands to run ABM simulations
    with open(f"{args.data_dir}/run.sh", "w") as f:
        for line in run_output:
            f.write(line)


if __name__ == "__main__":
    main()
