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
        "--lattice-constant", type=float, default=500, help="Lattice constant in nm."
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

    for current in currents:
        cmd = base_cmd.format(current)
        print(cmd)
        os.popen(cmd).read()
        time.sleep(0.1)
