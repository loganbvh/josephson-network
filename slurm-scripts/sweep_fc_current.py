import os
import time

import numpy as np

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--currents",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--linspace",
        type=float,
        nargs=3,
        default=None,
    )
    parser.add_argument(
        "--lattice-constant", type=float, default=0.5, help="Lattice constant in um."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't actually move any files."
    )

    args = parser.parse_args()
    if args.currents is not None:
        assert args.linspace is None
        currents = np.array(args.currents)
    else:
        assert args.linspace is not None
        start, stop, N = args.linspace
        N = int(N)
        currents = np.linspace(start, stop, N)

    base_cmd = (
        "sbatch"
        f" --export=ALL,fc_current={{}},lattice_constant={args.lattice_constant}"
        " square-array-twoloop.sbatch"
    )

    if args.dry_run:
        print("DRY RUN:")

    for current in currents:
        cmd = base_cmd.format(current)
        print(cmd)
        if not args.dry_run:
            print(os.popen(cmd).read())
            time.sleep(0.1)
