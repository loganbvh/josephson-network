import numpy as np
import pandas as pd

from ..uniform_field import UniformFieldModel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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
        "--array-size",
        type=int,
        nargs=2,
        default=None,
        help="Array size in units of unit cells.",
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
        "--Bz",
        type=str,
        help="Pint-parseable string specifiying the applied out-of-plane field.",
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
        "--rng-seed",
        type=int,
        default=-1,
        help=(
            "RNG seed for initial junction phases. "
            "If rng-seed is 0, junctions are initialized to zero."
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
    geometry = kwargs.pop("geometry")
    lattice_constant = kwargs.pop("lattice_constant")
    array_size = kwargs.pop("array_size")
    positions_fname = kwargs.pop("island_positions")
    number_of_starts = kwargs.pop("starts")
    if geometry is not None:
        assert lattice_constant is not None
        assert array_size is not None
        assert positions_fname is None

        if geometry == "triangular":
            raise NotImplementedError

        if geometry == "square":
            a = lattice_constant
            width, height = a * np.array(array_size)
            xs = np.linspace(-width / 2, width / 2, int(width / a) + 1)
            ys = np.linspace(-height / 2, height / 2, int(height / a) + 1)
            X, Y = np.meshgrid(xs, ys)
            island_positions = np.stack([X.ravel(), Y.ravel()], axis=1)
    else:
        assert lattice_constant is None
        assert array_size is None
        assert positions_fname is not None

        island_positions = pd.read_csv(positions_fname)
        island_positions = island_positions.values[:, -2:]
        assert island_positions.ndim == 2, island_positions.ndim
        assert island_positions.shape[1] == 2, island_positions.shape

    kwargs["island_positions"] = island_positions
    model = UniformFieldModel(**kwargs)
    model.run_multistart(number_of_starts=number_of_starts)
