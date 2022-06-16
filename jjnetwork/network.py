from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
import functools
import json
import os
from typing import Any, Callable, Optional, Union
import warnings


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pint
import pyomo.environ as pyo
from pyomo import opt
from pyomo.common.tempfiles import TempfileManager
from scipy.spatial import distance
from tqdm import tqdm

from .em import ureg, Phi_0
from .geometry import contains_points
from .graph_utils import (
    basis_loops,
    draw_graph,
    draw_currents_combined,
    edge_data_to_df,
    find_all_cells,
    remove_isolated_islands,
    remove_overlapping_islands,
)
from .io import DTFORMAT, NumpyJSONEncoder
from .pyomo_model import (
    ModelInfo,
    calculate_loop_info,
    model_to_graph,
    graph_to_model,
    initialize_variables,
    LoopInfo,
    set_model_flexible,
    set_model_strict,
)


@dataclass
class TimingInfo:
    run_start: Optional[datetime] = None
    run_stop: Optional[datetime] = None
    build_start: Optional[datetime] = None
    build_stop: Optional[datetime] = None
    solve_start: Optional[datetime] = None
    solve_stop: Optional[datetime] = None

    @property
    def build_time(self) -> int:
        if self.build_start is None or self.build_stop is None:
            return None
        return (self.build_stop - self.build_start).total_seconds()

    @property
    def solve_time(self) -> int:
        if self.solve_start is None or self.solve_stop is None:
            return None
        return (self.solve_stop - self.solve_start).total_seconds()

    @property
    def total_time(self) -> int:
        if self.run_start is None or self.run_stop is None:
            return None
        return (self.run_stop - self.run_start).total_seconds()


def build_graph(
    island_positions: np.ndarray,
    neighbors: list[list[int]],
    josephson_energy_func: Callable,
    vector_potential_func: Callable,
    vector_potential_points: int = 101,
) -> nx.DiGraph:
    """Generates the network.

    Args:
        island_positions: Shape (n, 2) array of island (x, y) positions in meters.
        neighbors: Length n list of lists of neighbor indices.
        josephson_energy_func: A callable with signature
            ``josephson_energy_func(junction_length_in_meters)
            -> josephson_energy_in_joules``.
        vector potential_func: A callable with signature
            ``vector_potential(positions) -> A_vector_in_tesla_meter``, where
            ``positions`` is a shape (m, 3) array of (x, y, z) points at which
            to evaluate the vector potential.
        vector_potential_points: Number of points used to approximate the line
            integral of the vector potential along each junction.
    """
    distances = distance.cdist(island_positions, island_positions)
    graph = nx.DiGraph()
    # Populate islands (nodes)
    for i, p in tqdm(
        enumerate(island_positions),
        total=island_positions.shape[0],
        desc="Populating islands",
    ):
        graph.add_node(
            i,
            position=p,
        )
    # Popualate junctions (edges)
    for i, indices in enumerate(neighbors):
        for j in indices:
            if (i, j) not in graph.edges and (j, i) not in graph.edges:
                graph.add_edge(i, j)
    nodes = graph.nodes
    for i, j, attrs in tqdm(
        graph.edges.data(),
        total=graph.number_of_edges(),
        desc="Populating junctions",
    ):
        n1 = nodes[i]
        n2 = nodes[j]
        # Approximate the line integral of the vector potential
        # using the trapezoid rule.
        r1 = n1["position"]
        r2 = n2["position"]
        xs = np.linspace(r1[0], r2[0], vector_potential_points)
        ys = np.linspace(r1[1], r2[1], vector_potential_points)
        rs = np.stack([xs, ys, np.zeros_like(xs)], axis=1)
        dr = np.diff(rs, axis=0)
        A = vector_potential_func(rs[:-1])
        Aij = np.trapz(np.sum(A * dr, axis=1)) / Phi_0
        length = distances[i, j]
        EJ = josephson_energy_func(length)
        attrs.update(
            dict(
                length=length,
                EJ=EJ,
                Aij_applied=Aij,
            )
        )
    return graph


class JosephsonNetwork(ABC):
    """An abstract base class for Josephson network models.

    Args:
        directory: The base directory in which to save results.
        island_positions: An array of (x, y) island positions in ``length_units``.
        length_units: Pint-parseable string representing the units used for lengths.
    """

    ureg = ureg

    META_ATTRS = [
        "basedir",
        "outdir",
        "island_diameter",
        "junction_cutoff_radius",
        "source_nodes",
        "drain_nodes",
        "source_drain_current",
        "include_screening",
        "length_units",
        "rng_seed",
        "solve_iteration",
    ]

    def __init__(
        self,
        *,
        directory: os.PathLike,
        island_positions: np.ndarray,
        island_diameter: Union[float, str, pint.Quantity],
        junction_cutoff_radius: Union[float, str, pint.Quantity],
        source_points: Optional[np.ndarray] = None,
        drain_points: Optional[np.ndarray] = None,
        source_drain_current: Optional[Union[str, float]] = None,
        include_screening: bool = False,
        length_units: str = "um",
    ):
        directory = os.path.abspath(directory)
        # Ensure a unique directory name for each simulation
        i = 0
        run_start = datetime.now()
        basedir = os.path.join(directory, run_start.strftime(DTFORMAT) + f"_{i:03}")
        while os.path.exists(basedir):
            i += 1
            run_start = datetime.now()
            basedir = os.path.join(directory, run_start.strftime(DTFORMAT) + f"_{i:03}")
        self.basedir = os.path.abspath(basedir)
        os.makedirs(self.basedir)
        self.solve_iteration = 0
        self.timing = TimingInfo(run_start=run_start)
        self.include_screening = include_screening
        self.length_units = ureg(length_units)
        self.rng_seed = None

        island_positions = np.atleast_2d(island_positions)
        self.island_positions = island_positions

        if isinstance(island_diameter, str):
            island_diameter = ureg(island_diameter)
        if isinstance(island_diameter, pint.Quantity):
            island_diameter = island_diameter.to(self.length_units).magnitude
        self.island_diameter = island_diameter
        if isinstance(junction_cutoff_radius, str):
            junction_cutoff_radius = ureg(junction_cutoff_radius)
        if isinstance(junction_cutoff_radius, pint.Quantity):
            junction_cutoff_radius = junction_cutoff_radius.to(
                self.length_units
            ).magnitude
        self.junction_cutoff_radius = junction_cutoff_radius
        self.compute_neighbors()
        self.model_info = ModelInfo()
        self.pyomo_result = None
        self._basis_cycles = None
        self.find_cells = None

        if isinstance(source_drain_current, str):
            source_drain_current = ureg(self.source_drain_current)
        elif isinstance(source_drain_current, (int, float)):
            source_drain_current = source_drain_current * ureg("A")
        if source_points is None:
            assert drain_points is None
            assert source_drain_current is None
            source_drain_current = 0
            source_nodes = []
            drain_nodes = []
        else:
            assert drain_points is not None
            assert source_drain_current is not None
            source_drain_current = source_drain_current.to("A").magnitude
            source_nodes = contains_points(source_points, island_positions, index=True)
            drain_nodes = contains_points(drain_points, island_positions, index=True)
        self.source_drain_current = source_drain_current
        self.source_nodes = source_nodes
        self.drain_nodes = drain_nodes

    def basis_cycles(self) -> list[list[int]]:
        if self.graph is None:
            return None
        if self._basis_cycles is None:
            self._basis_cycles = basis_loops(self.graph)
        return self._basis_cycles

    @property
    def outdir(self) -> os.PathLike:
        outdir = os.path.join(self.basedir, f"{self.solve_iteration:03}")
        os.makedirs(outdir, exist_ok=True)
        return outdir

    @property
    def json_file(self) -> os.PathLike:
        return os.path.join(self.outdir, "metadata.json")

    def make_rng(self) -> np.random.Generator:
        self.rng_seed = np.random.SeedSequence().entropy
        return np.random.default_rng(self.rng_seed)

    def compute_neighbors(self) -> None:
        """Removes any overlapping or isolated islands."""
        # Replace any set of overlapping islands with a single island located at
        # the mean position of the set of islands.
        island_positions = self.island_positions
        island_positions = remove_overlapping_islands(
            island_positions,
            self.island_diameter,
        )
        print(
            f"Total network size after removing overlapping islands: "
            f"{island_positions.shape[0]} islands."
        )
        island_positions, neighbors = remove_isolated_islands(
            island_positions,
            self.junction_cutoff_radius,
        )
        print(
            f"Total network size after removing isolated islands: "
            f"{island_positions.shape[0]} islands."
        )
        self.island_positions = island_positions
        self.neighbors = neighbors
        assert len(self.neighbors) == len(self.island_positions)

    @abstractmethod
    def josephson_energy(self, junction_length: float) -> float:
        """Returns the Josephson energy in joules for a given junction length in meters."""
        pass

    @abstractmethod
    def vector_potential(self, positions: np.ndarray) -> np.ndarray:
        """Returns the magnetic vector potential in tesla * meter at an array of
        (x, y, z) positions in meters.
        """
        pass

    @property
    def graph(self) -> Optional[nx.DiGraph]:
        return self.model_info.graph

    @property
    def loops(self) -> Optional[LoopInfo]:
        return self.model_info.loops

    @property
    def model(self) -> Optional[pyo.ConcreteModel]:
        return self.model_info.model

    def metadata(self, **kwargs) -> dict[str, Any]:
        meta = kwargs.copy()
        meta.update({attr: getattr(self, attr) for attr in self.META_ATTRS})
        meta["timing"] = asdict(self.timing)
        return meta

    def build_model(self) -> None:
        """Generates the graph representing the Josephson network
        and defines the NLP problem.
        """
        island_positions = (self.island_positions * self.length_units).to("m").magnitude
        n_neighbors = len(self.neighbors)
        n_islands = len(island_positions)
        if n_neighbors != n_islands:
            raise ValueError(
                f"The number of list of neighbors ({n_neighbors}) does not equal "
                f"the number of islands ({n_islands})."
            )
        self.timing.build_start = datetime.now()
        graph = self.graph
        if graph is None:
            print("Building graph...")
            graph = build_graph(
                island_positions=island_positions,
                neighbors=self.neighbors,
                josephson_energy_func=self.josephson_energy,
                vector_potential_func=self.vector_potential,
            )
        print("Building model from graph...")
        self.model_info = graph_to_model(
            graph,
            source_nodes=self.source_nodes,
            drain_nodes=self.drain_nodes,
            source_drain_current=self.source_drain_current,
            include_screening=self.include_screening,
        )
        print("Drawing graph...")
        fig, ax = draw_graph(self.graph)
        ax.set_title(os.path.basename(self.basedir))
        fig.savefig(os.path.join(self.basedir, "graph.pdf"), bbox_inches="tight")
        plt.close(fig)

        metadata = self.metadata()
        with open(self.json_file, "w") as f:
            json.dump(metadata, f, indent=4, sort_keys=True, cls=NumpyJSONEncoder)

        self.timing.build_stop = datetime.now()
        print(
            f"Finished building model in python. Elapsed time: "
            f"{self.timing.build_time:.3f} seconds."
        )
        self.find_cells = functools.lru_cache(
            functools.partial(find_all_cells, graph=self.graph)
        )

    def solve(self, reinitialize: bool = True) -> None:
        """Solves the NLP problem."""
        self.timing.solve_start = datetime.now()
        model = self.model
        rng = self.make_rng()
        print(f"RNG seed: {self.rng_seed}")
        if reinitialize:
            print("Initializing variables...")
            initialize_variables(model, rng=rng)

        TempfileManager.tempdir = self.outdir

        print("Solving model...", flush=True)
        # See: https://coin-or.github.io/Ipopt/OPTIONS.html and
        # https://github.com/Pyomo/pyomo/issues/206#issuecomment-324332219
        solver_options = dict(
            OF_tol=1e-8,
            OF_print_level=5,
            OF_print_info_string="yes",
            OF_print_user_options="yes",
            OF_print_timing_statistics="yes",
            OF_nlp_scaling_method="gradient-based",
            OF_max_iter=5000,
            OF_linear_solver="ma57",
            OF_ma57_automatic_scaling="yes",
        )
        minlp = False
        if minlp:

            for loop in model.loops:
                model.vortices[loop].domain = pyo.Integers
                flux = int(np.ceil(np.abs(pyo.value(model.applied_flux[loop]))))
                model.vortices[loop].lb = -flux
                model.vortices[loop].ub = +flux

            solver = opt.SolverFactory("mindtpy")
            self.pyomo_result = solver.solve(
                model,
                mip_solver="gurobi",
                nlp_solver="ipopt",
                nlp_solver_args=dict(options=solver_options),
                strategy="OA",
                init_strategy="initial_binary",
                tee=True,
                solver_tee=True,
                time_limit=3600,
                heuristic_nonconvex=True,
                integer_tolerance=1e-3,
            )

        else:
            solver = opt.SolverFactory("ipopt")
            solver.options.update(solver_options)
            self.pyomo_result = solver.solve(model, tee=True)

        print(str(self.pyomo_result.solver))
        self.timing.solve_stop = datetime.now()

    def process_results(self) -> None:
        """Extracts results from the graph and saves them to disk."""
        print("Building graph from solved model...")
        self.model_info.graph = model_to_graph(self.model)

        print("Extracting results from graph...")
        graph = self.graph
        outdir = self.outdir
        self.model_info.loops = loop_info = calculate_loop_info(
            graph, basis_loops(graph)
        )

        loops = np.array(loop_info.nodes, dtype=object)
        current = np.array(loop_info.current)
        vortices = np.array(loop_info.vortices)
        print("Vortex info:")
        print(pd.DataFrame(vortices).describe())
        applied_flux = np.array(loop_info.applied_flux)
        gauge_invariant_phase = np.array(loop_info.gauge_invariant_phase)
        (nonzero_loops,) = np.where(np.abs(vortices) > 1e-2)
        loop_data = {"nodes": loops}
        loop_data.update(
            {
                "current": current,
                "gauge_invariant_phase": gauge_invariant_phase,
                "vortices": vortices,
                "applied_flux": applied_flux,
            }
        )
        df = edge_data_to_df(graph)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pd.HDFStore(os.path.join(outdir, "results.h5")) as store:
                store["edge_data"] = df
                store["loops"] = pd.DataFrame(loop_data)
                store["nonzero_loops"] = pd.Series(nonzero_loops)

        josephson_energy = sum(
            energy for _, _, energy in graph.edges.data("josephson_energy")
        )
        inductive_energy = sum(
            energy for _, _, energy in graph.edges.data("inductive_energy")
        )
        total_energy = sum(energy for _, _, energy in graph.edges.data("energy"))
        print(f"Josephson energy: {josephson_energy:.3e} eV")
        print(f"Inductive energy: {inductive_energy:.3e} eV")
        print(f"Total energy: {total_energy:.3e} eV")

        print("Drawing currents...")
        fig, axes = draw_currents_combined(df=df, linewidth=3, cmap="inferno")
        fig.suptitle(outdir)
        fig.savefig(os.path.join(outdir, "currents.pdf"), bbox_inches="tight")
        plt.close(fig)

        self.timing.run_stop = datetime.now()
        print(f"Solve time: {self.timing.solve_time:.3f} seconds.")
        print(f"Cumulative run time: {self.timing.total_time:.3f} seconds.")

        metadata = self.metadata()
        metadata["solver_info"] = [
            line for line in str(self.pyomo_result.solver).splitlines() if line
        ]
        metadata["solver_status"] = str(self.pyomo_result.solver.status)
        metadata["termination_condition"] = str(
            self.pyomo_result.solver.termination_condition
        )
        metadata["josephson_energy"] = f"{josephson_energy:.6e} eV"
        metadata["inductive_energy"] = f"{inductive_energy:.6e} eV"
        metadata["energy"] = f"{total_energy:.6e} eV"
        metadata["vortices"] = {}
        for n, v in enumerate(vortices):
            if np.abs(v) > 1e-2:
                metadata["vortices"][n] = v
        print(f"Total number of vortices: {len(nonzero_loops)}.")
        with open(self.json_file, "w") as f:
            json.dump(metadata, f, indent=4, sort_keys=True, cls=NumpyJSONEncoder)

    def post_process(self) -> None:
        """Can be implemented by subclasses to perform additional processing after
        ``self.process_results()`` has been called.
        """
        pass

    def run_multistart(
        self, number_of_starts: int = 10, resolve_with_current_conservation: bool = True
    ):
        if self.model is None:
            self.build_model()
        model = self.model
        curr_iterations = self.solve_iteration
        if resolve_with_current_conservation:
            number_of_starts *= 2
        while (self.solve_iteration - curr_iterations) < number_of_starts:
            set_model_flexible(model, include_screening=self.include_screening)
            self.single_solve()
            self.solve_iteration += 1
            if resolve_with_current_conservation:
                set_model_strict(model, include_screening=self.include_screening)
                self.single_solve(reinitialize=False)
                self.solve_iteration += 1

    def single_solve(self, reinitialize: bool = True) -> None:
        self.solve(reinitialize=reinitialize)
        self.process_results()
        self.post_process()

    def run(self) -> None:
        """Builds the model, solves it, and processes results"""
        self.build_model()
        self.solve()
        self.process_results()
        self.post_process()
        self.solve_iteration += 1
