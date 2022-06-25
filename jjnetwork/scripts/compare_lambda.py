import os
import json

import numpy as np
import superscreen as sc

from ..em import ureg, current_loop_field
from ..geometry import circle, triangulate, polygon_centroids, triangle_areas
from ..susceptibility import TwoLoopModel


def Lambda_array(Ic):
    return (ureg("Phi_0") / (2 * np.pi * ureg("mu_0") * Ic)).to("um")


def Lambda_to_Ic(Lambda):
    return (ureg("Phi_0") / (2 * np.pi * ureg("mu_0") * Lambda)).to("uA")


def process_solution(solution: sc.Solution, metadata: dict):
    ureg = solution.device.ureg
    pl_outer = circle(metadata["pl_radius"], points=1001)
    length_units = ureg(metadata["length_units"])
    pl_center = np.atleast_2d(metadata["pl_center"])
    fc_center = np.atleast_2d(metadata["fc_center"])
    fc_current = ureg(metadata["fc_current"])
    fc_radius = metadata["fc_radius"]
    pl_points, pl_triangles = triangulate(pl_outer, min_triangles=5000)
    pl_centroids = polygon_centroids(pl_points, pl_triangles)
    pl_areas = triangle_areas(pl_points, pl_triangles) * length_units**2
    pl_centroids = np.append(pl_centroids, np.zeros_like(pl_centroids[:, :1]), axis=1)
    pl_centroids += pl_center
    length_scale = length_units.to("m").magnitude

    print("Calculating bare mutual inductance...")
    fc_field = current_loop_field(
        pl_centroids * length_scale,
        loop_center=fc_center * length_scale,
        loop_radius=fc_radius * length_scale,
        current=fc_current.to("A").magnitude,
    )[:, 2]
    fc_field = fc_field * ureg("tesla")
    bare_flux = np.einsum("i, i ->", fc_field, pl_areas).to("Phi_0")
    bare_mutual_bs = (bare_flux / fc_current).to("Phi_0 / A")
    print(f"Bare mutual inductance (Biot-Savart): {bare_mutual_bs:.3e~P}")
    field = solution.field_at_position(
        pl_centroids,
        units="Phi_0/um**2",
        with_units=True,
        return_sum=False,
    )["base"]
    flux = np.sum(field * pl_areas)
    susc = (flux / fc_current).to("Phi_0 / A")
    return susc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Lambda",
        type=float,
        default=None,
        help="Lambda in length_units.",
    )
    parser.add_argument(
        "--Lambda-geomspace",
        type=float,
        nargs=2,
        help="Start, stop, num_points for Lambda.",
    )
    parser.add_argument(
        "--Lambda-index",
        type=int,
        default=0,
        help="Index of Lambda in geomspace.",
    )
    parser.add_argument(
        "--superscreen-points",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--optimesh_steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--geometry",
        type=str,
        choices=("square", "triangular"),
        default="square",
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
        "--directory",
        type=str,
        default="./results",
        help="Output directory.",
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
        "--patch-radius-factor",
        type=float,
        default=3,
        help=(
            "Radius of the patch of islands to be modeled, "
            "in units of the loop_radius."
        ),
    )

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)
    number_of_starts = kwargs.pop("starts")
    geometry = kwargs.pop("geometry")
    lattice_constant = kwargs.pop("lattice_constant")

    if geometry == "triangular":
        raise NotImplementedError

    if geometry == "square":
        a = lattice_constant
        width = height = 100
        xs = ys = np.linspace(-width / 2, width / 2, int(width / a) + 1)
        X, Y = np.meshgrid(xs, ys)
        island_positions = np.stack([X.ravel(), Y.ravel()], axis=1)

    length_units = ureg(kwargs["length_units"])

    Lambda = kwargs.pop("Lambda")
    Lambda_geomspace = kwargs.pop("Lambda_geomspace")
    index = kwargs.pop("Lambda_index")
    if Lambda is None:
        start, stop, N = Lambda_geomspace
        Lambdas = np.geomspace(start, stop, int(N))
        Lambda = Lambdas[index]
    Lambda = Lambda * length_units
    Ic = Lambda_to_Ic(Lambda)
    kwargs["junction_I0"] = str(Ic)
    kwargs["junction_d0"] = f"{lattice_constant} {str(length_units)}"
    kwargs["junction_cutoff_radius"] = 1.1 * lattice_constant
    kwargs["island_diameter"] = "0 nm"
    kwargs["junction_length_dependence"] = "power_law"
    kwargs["island_positions"] = island_positions

    fc_current = ureg(kwargs["fc_current"])
    fc_radius = kwargs["fc_radius"]
    fc_center = np.atleast_2d(kwargs["fc_center"])
    pl_radius = kwargs["pl_radius"]
    pl_center = np.atleast_2d(kwargs["pl_center"])
    patch_radius_factor = kwargs["patch_radius_factor"]
    min_points = kwargs.pop("superscreen_points")
    optimesh_steps = kwargs.pop("optimesh_steps")

    # First, run the SuperScreen simulation.
    layer = sc.Layer("base", Lambda=0, z0=0)
    film = sc.Polygon(
        "film",
        layer="base",
        points=sc.geometry.circle(fc_radius * patch_radius_factor),
    )
    bounding_box = sc.Polygon(
        "bounding_box",
        layer="base",
        points=sc.geometry.circle(fc_radius * patch_radius_factor * 1.1, points=101),
    )

    device = sc.Device(
        "array",
        layers=[layer],
        films=[film],
        abstract_regions=[bounding_box],
        length_units=kwargs["length_units"],
    )

    def field_coil_field(x, y, z, *, fc_center, fc_radius, fc_current):
        if z.shape[0] != x.shape[0]:
            z = z * np.ones_like(x)
        x = np.squeeze(x)
        y = np.squeeze(y)
        z = np.squeeze(z)
        positions = np.stack([x, y, z], axis=1)
        length_scale = ureg("um").to("m").magnitude
        return current_loop_field(
            positions * length_scale,
            loop_center=np.atleast_2d(fc_center) * length_scale,
            loop_radius=fc_radius * length_scale,
            current=fc_current,
        )[:, 2]

    applied_field = sc.Parameter(
        field_coil_field,
        fc_center=fc_center,
        fc_radius=fc_radius,
        fc_current=fc_current.to("A").m,
    )

    device.make_mesh(min_points=min_points, optimesh_steps=optimesh_steps)

    device.layers["base"].Lambda = sc.Constant(Lambda.to("um").m)
    solution = sc.solve(
        device,
        applied_field=applied_field,
        field_units="tesla",
    )[-1]
    solution.to_file(os.path.join(kwargs["directory"], "sc_solution"), to_zip=True)

    for include_screening in (False, True):
        kwargs["include_screening"] = include_screening
        model = TwoLoopModel(**kwargs)
        model.run_multistart(number_of_starts=number_of_starts)

    json_path = os.path.join(
        model.basedir, f"{model.solve_iteration - 1:03}", "metadata.json"
    )

    with open(json_path, "r") as f:
        metadata = json.load(f)

    susc = process_solution(solution, metadata)
    print(f"SuperScreen susceptibility: {susc}.")
