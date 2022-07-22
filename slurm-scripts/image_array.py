import os
import sys
from datetime import datetime
import traceback

import numpy as np
import pandas as pd

sys.path.append(os.pardir)
import jjnetwork as jn  # noqa: E402

ureg = jn.ureg
DTFORMAT = jn.DTFORMAT


def simulate_image(**kwargs):
    outdir = kwargs.pop("directory")
    fc_xs = kwargs.pop("fc_xs")
    fc_ys = kwargs.pop("fc_ys")
    fc_z = kwargs.pop("fc_z")
    pl_z = kwargs.pop("pl_z") + fc_z
    fc_xs = np.linspace(*fc_xs[:2], int(fc_xs[2]))
    fc_ys = np.linspace(*fc_ys[:2], int(fc_ys[2]))
    pl_triangles = kwargs.pop("pl_triangles")
    num_starts = kwargs.pop("starts")
    positions_fname = kwargs.pop("island_positions")
    island_positions = pd.read_csv(positions_fname)
    island_positions = island_positions.values[:, -2:]
    assert island_positions.ndim == 2, island_positions.ndim
    assert island_positions.shape[1] == 2, island_positions.shape
    kwargs["island_positions"] = island_positions

    start_time = datetime.now()
    kwargs["directory"] = os.path.join(outdir, start_time.strftime(jn.DTFORMAT))
    print(kwargs)

    models = []
    for x in fc_xs:
        for y in fc_ys:
            _kwargs = kwargs.copy()
            _kwargs["fc_center"] = (x, y, fc_z)
            _kwargs["pl_center"] = (x, y, pl_z)
            model = jn.TwoLoopModel(**_kwargs)
            print(f"Created model {model.basedir}...\n")
            models.append(model)

    print(f"Solving {len(models)} models serially...")
    print(kwargs["directory"])

    while models:
        model = models.pop(0)
        model.build_model(pl_triangles=pl_triangles)
        print(f"{datetime.now().isoformat()}: Starting model {model.basedir}...\n")
        try:
            model.run_multistart(number_of_starts=num_starts)
        except Exception:
            print(traceback.format_exc())
        print(
            f"{datetime.now().isoformat()}: Finished model {model.basedir} in "
            f"{model.timing.total_time:.3} seconds.\n"
        )
        del model

    print(kwargs["directory"])


def main(**kwargs):
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        row_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    else:
        row_id = 0
    fc_ys = kwargs.pop("fc_ys")
    fc_ys = np.linspace(*fc_ys[:2], int(fc_ys[2]))
    fc_y = fc_ys[row_id]
    kwargs["fc_ys"] = (fc_y, fc_y, 1)
    simulate_image(**kwargs)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default="./results",
        help="Output directory.",
    )
    parser.add_argument(
        "--length-units",
        type=str,
        default="um",
        help="Units to use for all lengths in the model.",
    )
    parser.add_argument(
        "--island-positions",
        type=str,
        help="Path to .mat file containing island positions.",
    )
    parser.add_argument(
        "--fc-xs",
        type=float,
        nargs=3,
        default=0,
        help="x position (start, stop, num) for the field coil (current loop) center.",
    )
    parser.add_argument(
        "--fc-ys",
        type=float,
        nargs=3,
        default=0,
        help="y position (start, stop, num) for the field coil (current loop) center.",
    )
    parser.add_argument(
        "--fc-z",
        type=float,
        default=0,
        help="z position for the field coil (current loop) center.",
    )
    parser.add_argument(
        "--fc-radius",
        type=float,
        default=1,
        help="Radius of the field coild (loop of current).",
    )
    parser.add_argument(
        "--fc-current",
        type=str,
        default="100 uA",
        help="Pint-parsable string representing the current in the loop.",
    )
    parser.add_argument(
        "--pl-z",
        type=float,
        default=0,
        help="z position of the pickup loop center relative to the FC center.",
    )
    parser.add_argument(
        "--pl-radius",
        type=float,
        default=0.5,
        help="Radius of the pickup loop.",
    )
    parser.add_argument(
        "--pl-triangles",
        type=int,
        default=5000,
        help="Number of triangles in the pickup loop mesh.",
    )
    parser.add_argument(
        "--junction-cutoff-radius",
        type=float,
        default=0.7,
        help=(
            "Neighbors within this distance from a given island "
            "are assumed to form junctions."
        ),
    ),
    parser.add_argument(
        "--junction-length-dependence",
        type=str,
        choices=("power_law", "exponential"),
        help="Dependence of junction critical current on edge-to-edge spacing, d.",
    )
    parser.add_argument(
        "--junction-I0",
        type=str,
        help=(
            "Pint-parsable string representing the energy scale "
            "for junction critical currents. If junction_length_dependence is "
            "'power_law', junction_I0 should have dimensions of current * area. "
            "Otherwise, junction_I0 should have dimensions of current."
        ),
    )
    parser.add_argument(
        "--junction-d0",
        type=str,
        help=(
            "Pint-parsable string with dimensions of length representing the "
            "characteristic scale for junction critical current length dependence. "
        ),
    )
    parser.add_argument(
        "--island-diameter",
        type=str,
        default="260 nm",
        help="Pint-parsable string representing the island diameter.",
    )
    parser.add_argument(
        "--patch-radius-factor",
        type=float,
        default=4,
        help=(
            "Radius of the patch of islands to be modeled, "
            "in units of the loop_radius."
        ),
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

    args = parser.parse_args()

    main(**vars(args))
