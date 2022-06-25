import json
import os
from typing import Sequence, Union, Optional

import matplotlib.pyplot as plt
import pint
import numpy as np
import scipy.linalg as la
import superscreen as sc

from . import squids
from .. import em
from .. import graph_utils as gu
from ..io import NumpyJSONEncoder
from ..network import JosephsonNetwork
from ..junctions import josephson_energy_power_law, josephson_energy_exponential

ureg = em.ureg


class FieldCoil:
    def __init__(
        self,
        r_inner: Union[float, str, pint.Quantity],
        r_outer: Union[float, str, pint.Quantity],
    ):
        if isinstance(r_inner, str):
            r_inner = ureg(r_inner)
        if isinstance(r_outer, str):
            r_outer = ureg(r_outer)
        self.r_inner = r_inner
        self.r_outer = r_outer

    @property
    def d_inner(self) -> Union[float, pint.Quantity]:
        return 2 * self.r_inner

    @property
    def d_outer(self) -> Union[float, pint.Quantity]:
        return 2 * self.r_outer

    @property
    def d_effective(self) -> Union[float, pint.Quantity]:
        return np.sqrt((self.d_inner**2 + self.d_outer**2) / 2)

    @property
    def r_effective(self) -> Union[float, pint.Quantity]:
        return self.d_effective / 2


EJ_funcs = {
    "power_law": josephson_energy_power_law,
    "exponential": josephson_energy_exponential,
}

squid_map = {
    "ibm.small": squids.ibm.small,
    "ibm.medium": squids.ibm.medium,
    "ibm.large": squids.ibm.large,
    "ibm.xlarge": squids.ibm.xlarge,
    "huber": squids.huber,
    "hypres.small": squids.hypres.small,
}

field_coil_radii = {
    "ibm.small": FieldCoil("0.5 um", "1.0 um"),
    "ibm.medium": FieldCoil("1.0 um", "1.5 um"),
    "ibm.large": FieldCoil("2.5 um", "3.5 um"),
    "ibm.xlarge": FieldCoil("6.0 um", "8.0 um"),
    "huber": FieldCoil("5.5 um", "8.0 um"),
    "hypres.small": FieldCoil("1.5 um", "3.0 um"),
}


def Bz_from_graph(x, y, z, *, graph, units="mT"):
    z = z * np.ones_like(x)
    positions = np.stack([x, y, z], axis=1)
    field = em.calculate_field_from_graph(positions, graph, length_units="um")
    return field.to(units).magnitude[:, 2]


class SSMModel(JosephsonNetwork):
    """A Josephson network model where the SQUID is modeled as a SuperScreen device."""

    META_ATTRS = [
        "fc_current",
        "squid_type",
        "squid_position",
        "squid_points",
        "squid_iterations",
        "squid_fname",
    ] + JosephsonNetwork.META_ATTRS

    def __init__(
        self,
        *,
        squid_type: str,
        squid_position: Sequence[float],
        fc_current: str,
        patch_radius_factor: float,
        junction_d0: str,
        junction_I0: str,
        junction_length_dependence: str = "power_law",
        squid_fname: Optional[str] = None,
        squid_points: int = 5000,
        squid_iterations: int = 4,
        solve_dtype: str = "float64",
        **kwargs,
    ):
        self.squid_fname = squid_fname
        self.squid_type = squid_type
        if self.squid_fname is None:
            squid = squid_map[self.squid_type].make_squid()
        else:
            squid = sc.Device.from_file(self.squid_fname)
        self.squid_position = np.atleast_2d(squid_position)
        self.squid_points = squid_points
        self.squid_iterations = squid_iterations
        self.squid = squid.translate(*self.squid_position.squeeze())
        self.squid.solve_dtype = solve_dtype
        self.fc_current = ureg(fc_current)
        # Remove points lying outside the patch radius
        length_units = kwargs["length_units"]
        fc_radius = field_coil_radii[self.squid_type].r_effective.to(length_units).m
        self.patch_radius = fc_radius * patch_radius_factor
        island_positions = kwargs.pop("island_positions")
        island_positions = island_positions[
            la.norm(island_positions - self.squid_position[:, :2], axis=1)
            <= self.patch_radius
        ]
        kwargs["island_positions"] = island_positions
        self.junction_d0 = junction_d0
        self.junction_I0 = junction_I0
        assert junction_length_dependence in EJ_funcs
        self.junction_length_dependence = junction_length_dependence

        if self.squid.Del2 is None:
            self.squid.make_mesh(min_points=squid_points, optimesh_steps=30)

        circulating_currents = dict(fc_center=str(self.fc_current))
        I_fc = squid.ureg(circulating_currents["fc_center"])
        self.fc_solution = sc.solve(
            device=self.squid,
            circulating_currents=circulating_currents,
            iterations=self.squid_iterations,
        )[-1]
        pl_fluxoid = sum(self.fc_solution.hole_fluxoid("pl_center", units="Phi_0"))
        self.bare_mutual = (pl_fluxoid / I_fc).to("Phi_0/A")
        print(f"Bare mutual inductance: {self.bare_mutual:~.3fP}")

        super().__init__(**kwargs)

    def josephson_energy(self, junction_length: float) -> float:
        d0 = ureg(self.junction_d0).to("m").magnitude
        I0 = ureg(self.junction_I0).to("A").magnitude
        ej_func = EJ_funcs[self.junction_length_dependence]
        return ej_func(junction_length, d0=d0, I0=I0)

    def vector_potential(self, positions: np.ndarray) -> np.ndarray:
        positions = positions * 1e6
        return self.fc_solution.vector_potential_at_position(
            positions,
            units="T * m",
            with_units=False,
        )

    def post_process(self):
        """Calculates the flux through the pickup loop due to the supercurrents
        flowing in the network.
        """
        print("Calculating screening field...")
        graph = self.graph
        applied_field = sc.Parameter(Bz_from_graph, graph=graph)
        self.pl_solution = sc.solve(
            self.squid,
            applied_field=applied_field,
            circulating_currents=dict(fc_center=str(self.fc_current)),
            field_units="mT",
            iterations=self.squid_iterations,
        )[-1]
        screening_flux = sum(self.pl_solution.hole_fluxoid("pl_center")).to("Phi_0")
        fc_current = sc.ureg(str(self.fc_current))
        bare_mutual = sc.ureg(str(self.bare_mutual))
        mutual = (screening_flux / fc_current).to("Phi_0 / A")
        self.susceptibility = ureg(str(mutual - bare_mutual))
        print(f"Bare mutual inductance: {self.bare_mutual:.3e~P}")
        print(f"Mutual inductance: {mutual:.3e~P}")
        print(f"Susceptibility: {self.susceptibility:.3e~P}")
        with open(self.json_file, "r") as f:
            metadata = json.load(f)
        metadata["susceptibility"] = self.susceptibility
        with open(self.json_file, "w") as f:
            json.dump(metadata, f, indent=4, sort_keys=True, cls=NumpyJSONEncoder)

        squid_polygons = {**self.squid.films, **self.squid.holes}
        fig, ax = gu.draw_graph(graph, buffer=1.15)
        for polygon in squid_polygons.values():
            points = polygon.points
            points = (points * ureg(self.squid.length_units)).to("m").magnitude
            ax.plot(points[:, 0], points[:, 1], "C1-")
        ax.set_title(os.path.basename(self.basedir))
        fig.savefig(os.path.join(self.basedir, "graph.pdf"), bbox_inches="tight")
        plt.close(fig)

        df = gu.edge_data_to_df(graph)
        energy = sum(energy for _, _, energy in graph.edges.data("energy"))
        fig, axes = gu.draw_currents_combined(df=df, linewidth=3, cmap="inferno")
        title = [
            self.outdir,
            (
                f"Junction I0: {ureg(self.junction_I0):.3f~P}, "
                f"FC current: {self.fc_current:.3f~P}, "
                f"Susceptibility: {self.susceptibility:.3e~P}, "
                f"Energy: {energy:.4e} eV"
            ),
        ]
        for ax in axes:
            for polygon in squid_polygons.values():
                points = polygon.points
                ax.plot(points[:, 0], points[:, 1], "k-", alpha=0.5)
        fig.suptitle("\n".join(title))
        fig.subplots_adjust(top=0.85)
        fig.savefig(os.path.join(self.outdir, "currents.pdf"), bbox_inches="tight")
        plt.close(fig)
        self.fc_solution.to_file(os.path.join(self.outdir, "fc_solution"), to_zip=True)
        self.pl_solution.to_file(os.path.join(self.outdir, "pl_solution"), to_zip=True)

        return super().post_process()
