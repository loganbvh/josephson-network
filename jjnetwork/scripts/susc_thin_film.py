import numpy as np

from ..em import ureg
from ..susceptibility import TwoLoopModel, SSMModel


def lambda_vs_T(T: float, Tc: float = 1.0, lambda_0: float = 1.0):
    # London penetration depth for T \approx T_c
    return lambda_0 / np.sqrt((1 - (T / Tc) ** 2))


def xi_vs_T(T: float, Tc: float = 1.0, xi_0: float = 1.0, bcs_prefactor: float = 0.74):
    # Tinkham Eq. 4.24
    return bcs_prefactor * xi_0 / np.sqrt(1 - T / Tc)


def Lambda_array(Ic):
    return (ureg("Phi_0") / (2 * np.pi * ureg("mu_0") * Ic)).to("um")


def Lambda_to_Ic(Lambda):
    return (ureg("Phi_0") / (2 * np.pi * ureg("mu_0") * Lambda)).to("uA")


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
            "hypres.small",
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
        # default="100 uA",
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
        "--patch-radius-factor",
        type=float,
        default=3,
        help=(
            "Radius of the patch of islands to be modeled, "
            "in units of the loop_radius."
        ),
    )
    parser.add_argument(
        "--include-screening",
        type=lambda s: bool(int(s)),
        default=False,
        help="Whether to include screening in the calculation ('0' or '1').",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)
    model_type = kwargs.pop("model")
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
    number_of_starts = kwargs.pop("starts")
    geometry = kwargs.pop("geometry")
    length_units = kwargs["length_units"]

    d = kwargs.pop("thickness")
    xi_0 = kwargs.pop("xi0")
    lambda_0 = kwargs.pop("lambda0")
    t = kwargs.pop("temp")
    assert 0 <= t < 1, t
    Lambda = lambda_vs_T(t, lambda_0=lambda_0) ** 2 / d
    xi = xi_vs_T(t, xi_0=xi_0)
    # See Physical Review. B, Condensed Matter 47 (9): 5219–29. p.5227 right column,
    # and Physical Review. B, Condensed Matter 43 (13): 10218–28 p.10 223 left column.
    lattice_constant = np.sqrt(2 * np.pi) * xi
    Ic = Lambda_to_Ic(Lambda * ureg(length_units))

    kwargs["junction_d0"] = str(lattice_constant * ureg(length_units))
    kwargs["junction_I0"] = str(Ic)
    kwargs["junction_cutoff_radius"] = 1.1 * lattice_constant
    kwargs["junction_length_dependence"] = "power_law"
    kwargs["island_diameter"] = "0 nm"

    if geometry == "square":
        a = lattice_constant
        width = height = 100
        xs = ys = np.linspace(-width / 2, width / 2, int(width / a) + 1)
        X, Y = np.meshgrid(xs, ys)
        island_positions = np.stack([X.ravel(), Y.ravel()], axis=1)
    else:
        raise NotImplementedError

    kwargs["island_positions"] = island_positions
    model = model_cls(**kwargs)
    model.run_multistart(number_of_starts=number_of_starts)
