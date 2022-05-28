import json
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from .. import em
from .. import graph_utils as gu
from ..io import NumpyJSONEncoder
from ..network import JosephsonNetwork
from ..geometry import (
    circle,
    close_curve,
    triangulate,
    triangle_areas,
    polygon_centroids,
)
from ..junctions import josephson_energy_power_law, josephson_energy_exponential

ureg = em.ureg

EJ_funcs = {
    "power_law": josephson_energy_power_law,
    "exponential": josephson_energy_exponential,
}


class TwoLoopModel(JosephsonNetwork):
    """A Josephson network model where the SQUID field coil and pickup loop are
    represented by 1D circular loops.
    """

    META_ATTRS = [
        "fc_center",
        "fc_radius",
        "fc_current",
        "pl_center",
        "pl_radius",
        "patch_radius",
        "junction_length_dependence",
        "junction_d0",
        "junction_I0",
    ] + JosephsonNetwork.META_ATTRS

    def __init__(
        self,
        *,
        fc_center: Sequence[float],
        fc_radius: float,
        fc_current: str,
        pl_center: Sequence[float],
        pl_radius: float,
        patch_radius_factor: float,
        junction_d0: str,
        junction_I0: str,
        junction_length_dependence: str = "power_law",
        **kwargs,
    ):
        # Field coil info
        self.fc_center = np.atleast_2d(fc_center)
        self.fc_radius = fc_radius
        self.fc_current = ureg(fc_current)
        # Pickup loop info
        self.pl_center = np.atleast_2d(pl_center)
        self.pl_radius = pl_radius
        self.pl_centroids = None
        self.pl_areas = None
        # Remove points lying outside the patch radius
        self.patch_radius = fc_radius * patch_radius_factor
        island_positions = kwargs.pop("island_positions")
        island_positions = island_positions[
            la.norm(island_positions - self.fc_center[:, :2], axis=1)
            <= self.patch_radius
        ]
        kwargs["island_positions"] = island_positions
        print(f"Total patch size: {island_positions.shape[0]} islands.")
        self.junction_d0 = junction_d0
        self.junction_I0 = junction_I0
        assert junction_length_dependence in EJ_funcs
        self.junction_length_dependence = junction_length_dependence

        super().__init__(**kwargs)

    def josephson_energy(self, junction_length: float) -> float:
        d0 = ureg(self.junction_d0).to("m").magnitude
        I0 = ureg(self.junction_I0).to("A").magnitude
        ej_func = EJ_funcs[self.junction_length_dependence]
        return ej_func(junction_length, d0=d0, I0=I0)

    def build_model(
        self,
        pl_points: int = 201,
        pl_triangles: int = 5000,
    ) -> None:
        pl_outer = circle(self.pl_radius, points=pl_points)
        pl_points, pl_triangles = triangulate(pl_outer, min_triangles=pl_triangles)
        pl_centroids = polygon_centroids(pl_points, pl_triangles)
        pl_areas = triangle_areas(pl_points, pl_triangles) * self.length_units**2
        pl_centroids = np.append(
            pl_centroids, np.zeros_like(pl_centroids[:, :1]), axis=1
        )
        pl_centroids += self.pl_center
        self.pl_areas = pl_areas
        self.pl_centroids = pl_centroids
        length_scale = self.length_units.to("m").magnitude

        print("Calculating bare mutual inductance...")
        fc_field = em.current_loop_field(
            pl_centroids * length_scale,
            loop_center=self.fc_center * length_scale,
            loop_radius=self.fc_radius * length_scale,
            current=self.fc_current.to("A").magnitude,
        )[:, 2]
        fc_field = fc_field * ureg("tesla")
        bare_flux = np.einsum("i, i ->", fc_field, pl_areas).to("Phi_0")
        bare_mutual_bs = (bare_flux / self.fc_current).to("Phi_0 / A")
        print(f"Bare mutual inductance (Biot-Savart): {bare_mutual_bs:.3e~P}")

        pl_outer = (
            np.append(pl_outer, np.zeros_like(pl_outer[:, :1]), axis=1)
            * self.length_units
        )
        pl_outer += self.pl_center * self.length_units
        pl_vector_potential = em.current_loop_vector_potential(
            pl_outer.to("m").magnitude,
            loop_center=self.fc_center * length_scale,
            loop_radius=self.fc_radius * length_scale,
            current=self.fc_current.to("A").magnitude,
        ) * ureg("tesla * meter")
        d_pl = np.diff(close_curve(pl_outer), axis=0)
        pl_flux = np.trapz(np.sum(pl_vector_potential * d_pl, axis=1)).to("Phi_0")
        bare_mutual_A = (pl_flux / self.fc_current).to("Phi_0 / A")
        print(f"Bare mutual inductance (vector potential): {bare_mutual_A:.3e~P}")
        return super().build_model()

    def vector_potential(self, positions: np.ndarray) -> np.ndarray:
        length_scale = self.length_units.to("m").magnitude
        return em.current_loop_vector_potential(
            positions,
            loop_center=self.fc_center * length_scale,
            loop_radius=self.fc_radius * length_scale,
            current=self.fc_current.to("A").magnitude,
        )

    def post_process(self):
        """Calculates the flux through the pickup loop due to the supercurrents
        flowing in the network.
        """
        print("Calculating screening field...")
        graph = self.graph
        pl_areas = self.pl_areas
        screening_field = em.calculate_field_from_graph(self.pl_centroids, graph)[:, 2]
        screening_flux = np.einsum("i, i ->", screening_field, pl_areas).to("Phi_0")
        mutual = (screening_flux / self.fc_current).to("Phi_0 / A")
        print(f"Susceptibility: {mutual:.3e~P}")
        with open(self.json_file, "r") as f:
            metadata = json.load(f)
        metadata["susceptibility"] = mutual
        with open(self.json_file, "w") as f:
            json.dump(metadata, f, indent=4, sort_keys=True, cls=NumpyJSONEncoder)

        length_scale = self.length_units.to("m").magnitude
        fig, ax = gu.draw_graph(graph)
        fc = (
            close_curve(circle(self.fc_radius)) + self.fc_center[:, :2]
        ) * length_scale
        pl = (
            close_curve(circle(self.pl_radius)) + self.pl_center[:, :2]
        ) * length_scale
        ax.plot(fc[:, 0], fc[:, 1], "C1-", lw=3)
        ax.plot(pl[:, 0], pl[:, 1], "C2-", lw=3)
        ax.set_title(os.path.basename(self.outdir))
        fig.savefig(os.path.join(self.outdir, "graph.pdf"), bbox_inches="tight")
        plt.close(fig)

        df = gu.edge_data_to_df(graph)
        fig, axes = gu.draw_currents(df, linewidth=3, cmap="inferno")
        title = [
            self.outdir,
            (
                f"Junction I0: {self.junction_I0}, "
                f"FC current: {self.fc_current:.2f~P}, "
                f"Susceptibility: {mutual:.3e~P}"
            ),
        ]
        fig.suptitle("\n".join(title))
        fig.subplots_adjust(top=0.85)
        fig.savefig(os.path.join(self.outdir, "currents.pdf"), bbox_inches="tight")
        plt.close(fig)

        return super().post_process()
