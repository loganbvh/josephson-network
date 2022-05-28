from typing import Union

import networkx as nx
import numpy as np
import pint
from tqdm import tqdm

from . import em
from .network import JosephsonNetwork
from .geometry import (
    circle,
    triangulate,
    triangle_areas,
    polygon_centroids,
)
from .junctions import josephson_energy_power_law, josephson_energy_exponential

ureg = em.ureg

EJ_funcs = {
    "power_law": josephson_energy_power_law,
    "exponential": josephson_energy_exponential,
}


class TransportModel(JosephsonNetwork):

    META_ATTRS = [
        "Bz",
        "junction_length_dependence",
        "junction_d0",
        "junction_I0",
    ] + JosephsonNetwork.META_ATTRS

    def __init__(
        self,
        *,
        junction_d0: str,
        junction_I0: str,
        Bz: Union[str, float, pint.Quantity] = "0 tesla",
        junction_length_dependence: str = "power_law",
        **kwargs,
    ):
        self.junction_d0 = junction_d0
        self.junction_I0 = junction_I0
        assert junction_length_dependence in EJ_funcs
        self.junction_length_dependence = junction_length_dependence
        if isinstance(Bz, str):
            Bz = ureg(Bz)
        if isinstance(Bz, float):
            Bz = Bz * ureg("tesla")
        self.Bz = Bz

        super().__init__(**kwargs)

    def vector_potential(self, positions: np.ndarray) -> float:
        A = em.uniform_Bz_vector_potential(positions, self.Bz)
        return A.to("tesla * meter").magnitude

    def josephson_energy(self, junction_length: float) -> float:
        d0 = ureg(self.junction_d0).to("m").magnitude
        I0 = ureg(self.junction_I0).to("A").magnitude
        ej_func = EJ_funcs[self.junction_length_dependence]
        return ej_func(junction_length, d0=d0, I0=I0)


def image_current_loop(
    graph: nx.Graph,
    pl_radius: float,
    pl_centers: np.ndarray,
    pl_points: int = 201,
    pl_triangles: int = 5000,
    length_units: str = "um",
    units: str = "Phi_0",
    with_units: bool = True,
) -> np.ndarray:
    assert pl_centers.ndim == 2
    assert pl_centers.shape[1] == 3
    length_units = ureg(length_units)
    pl_outer = circle(pl_radius, points=pl_points)
    pl_points, pl_triangles = triangulate(pl_outer, min_triangles=pl_triangles)
    pl_centroids = polygon_centroids(pl_points, pl_triangles)
    pl_areas = triangle_areas(pl_points, pl_triangles) * length_units**2
    pl_centroids = np.append(pl_centroids, np.zeros_like(pl_centroids[:, :1]), axis=1)

    flux = []
    for r0 in tqdm(pl_centers):
        positions = pl_centroids + r0
        field = em.calculate_field_from_graph(
            positions, graph, length_units=length_units
        )[:, 2]
        flux.append(np.einsum("i, i ->", field, pl_areas).to(units).magnitude)
    flux = np.array(flux)
    if with_units:
        flux = flux * ureg(units)
    return flux
