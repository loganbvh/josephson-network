import numpy as np
import pandas as pd

from ..susceptibility import TwoLoopModel, SSMModel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=("twoloop", "squid"),
    )
    parser.add_argument(
        "--geometry",
        type=str,
        choices=("square", "triangular"),
        default=None,
    )
    parser.add_argument(
        "--lattice-constant",
        type=float,
        default=None,
        help="Lattice constant in length_units.",
    )
    parser.add_argument(
        "--starts",
        type=int,
        default=1,
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
        default=None,
        help="Path to csv file containing island positions.",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="./results",
        help="Output directory.",
    )
    parser.add_argument(
        "--squid-type",
        type=str,
        choices=(
            "ibm.small",
            "ibm.medium",
            "ibm.large",
            "ibm.xlarge",
            "huber",
        ),
        default="ibm.medium",
        help="Model for the SQUID susceptometer.",
    )
    parser.add_argument(
        "--squid-position",
        type=float,
        nargs=3,
        default=(0, 0, 0),
        help="x, y, z position of the SQUID susceptometer.",
    )
    parser.add_argument(
        "--squid-points",
        type=int,
        default=5000,
        help="Number of points in the SQUID mesh.",
    )
    parser.add_argument(
        "--squid-iterations",
        type=int,
        default=4,
        help="Number of SuperScreen iterations to use to model the SQUID.",
    )
    parser.add_argument(
        "--fc-center",
        type=float,
        nargs=3,
        default=(0, 0, 0),
        help="x, y, z position of the field coil (current loop) center.",
    )
    parser.add_argument(
        "--fc-radius",
        type=float,
        default=1,
        help="Radius of the field coil (loop of current).",
    )
    parser.add_argument(
        "--fc-current",
        type=str,
        default="100 uA",
        help="Pint-parsable string representing the current in the loop.",
    )
    parser.add_argument(
        "--pl-center",
        type=float,
        nargs=3,
        default=(0, 0, 0),
        help="x, y, z position of the pickup loop center.",
    )
    parser.add_argument(
        "--pl-radius",
        type=float,
        default=0.5,
        help="Radius of the pickup loop.",
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
        "--junction-d0",
        type=str,
        help=(
            "Pint-parsable string with dimensions of length representing the "
            "characteristic scale for junction critical current length dependence. "
        ),
    )
    parser.add_argument(
        "--junction-I0",
        type=str,
        help=(
            "Pint-parsable string with dimensions of current representing the "
            "current scale for junction critical currents. "
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
        default=2.5,
        help=(
            "Radius of the patch of islands to be modeled, "
            "in units of the loop_radius."
        ),
    )
    parser.add_argument(
        "--include-screening",
        type=bool,
        default=False,
        help="Whether to include screening in the calculation.",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)
    model_type = kwargs.pop("model")
    number_of_starts = kwargs.pop("starts")
    geometry = kwargs.pop("geometry")
    lattice_constant = kwargs.pop("lattice_constant")
    positions_fname = kwargs.pop("island_positions")

    if model_type == "twoloop":
        model_cls = TwoLoopModel
        pop_args = [
            "squid_type",
            "squid_position",
            "squid_points",
            "squid_iterations",
        ]
    else:
        model_cls = SSMModel
        pop_args = [
            "fc_center",
            "fc_radius",
            "pl_center",
            "pl_radius",
        ]
    for name in pop_args:
        _ = kwargs.pop(name)

    if geometry is not None:
        assert lattice_constant is not None
        assert positions_fname is None

        if geometry == "triangular":
            raise NotImplementedError

        if geometry == "square":
            a = lattice_constant
            # width = height = 2 * (
            #     (2.5 * kwargs["fc_radius"] * kwargs["patch_radius_factor"]) // 2 + 1
            # )
            width = height = 100
            xs = ys = np.linspace(-width / 2, width / 2, int(width / a) + 1)
            X, Y = np.meshgrid(xs, ys)
            island_positions = np.stack([X.ravel(), Y.ravel()], axis=1)
    else:
        # Load island positions from file
        assert lattice_constant is None
        assert positions_fname is not None

        island_positions = pd.read_csv(positions_fname)
        island_positions = island_positions.values[:, -2:]
        assert island_positions.ndim == 2, island_positions.ndim
        assert island_positions.shape[1] == 2, island_positions.shape

    kwargs["island_positions"] = island_positions
    model = model_cls(**kwargs)
    model.run_multistart(number_of_starts=number_of_starts)
