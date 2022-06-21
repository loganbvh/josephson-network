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
        "--temp",
        type=float,
        help="Reduced temperature, T / T_c.",
    )
    parser.add_argument(
        "--lambda0",
        type=float,
        help="Zero temperature London penetration depth in length_units.",
    )
    parser.add_argument(
        "--xi0",
        type=float,
        help="Zero temperature GL coherence length in length_units.",
    )
    parser.add_argument(
        "--thickness",
        type=float,
        help="Film thickness in length_units.",
    )
    parser.add_argument(
        "--squid-height",
        type=float,
        help="SQUID height in length_units.",
    )
    parser.add_argument(
        "--starts",
        type=int,
        default=10,
        help="Number of solve starts.",
    )
    parser.add_argument(
        "--include-screening", action="store_true", help="Whether to include screening."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't actually submit and jobs."
    )

    args = parser.parse_args()
    kwargs = vars(args)
    if args.currents is not None:
        assert args.linspace is None
        kwargs.pop("linspace")
        currents = np.array(kwargs.pop("currents"))
    else:
        assert args.linspace is not None
        kwargs.pop("currents")
        start, stop, N = kwargs.pop("linspace")
        N = int(N)
        currents = np.linspace(start, stop, N)

    dry_run = kwargs.pop("dry_run")
    kwargs["screening"] = str(int(kwargs.pop("include_screening")))

    export = ",".join(["ALL"] + [f"{k}={v}" for k, v in kwargs.items()])

    base_cmd = f"sbatch --export={export},fc_current={{}} susc_thin_film.sbatch"

    if dry_run:
        print("DRY RUN:")

    for current in currents:
        cmd = base_cmd.format(current)
        print(cmd)
        if not dry_run:
            print(os.popen(cmd).read())
            time.sleep(0.1)
