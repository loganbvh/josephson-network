from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os
from typing import Any, Callable, Optional, Union

from gekko import GEKKO
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pint
import scipy.linalg as la
from tqdm import tqdm

from .em import ureg, Phi_0, eV
from .geometry import contains_points
from .graph_utils import (
    get_scalar,
    basis_loops,
    find_all_cells,
    draw_graph,
    draw_currents,
    edge_data_to_df,
    round_trip,
    remove_isolated_islands,
    remove_overlapping_islands,
)
from .io import DTFORMAT, NumpyJSONEncoder


@dataclass
class LoopInfo:
    """A container for data related to a single loop (closed path) in a network."""

    nodes: list[int]
    applied_flux: float
    current: Optional[float] = None
    gauge_invariant_phase: Optional[float] = None
    # total_flux: Optional[float] = None
    frustration: Optional[float] = None
    vortices: Optional[float] = None


@dataclass
class ModelInfo:
    graph: nx.DiGraph
    loops: list[LoopInfo]


@dataclass
class TimingInfo:
    run_start: Optional[datetime] = None
    build_start: Optional[datetime] = None
    run_stop: Optional[datetime] = None
    run_time: Optional[float] = None
    gekko_start: Optional[datetime] = None
    gekko_stop: Optional[datetime] = None


def build_graph(
    *,
    gekko_model: GEKKO,
    island_positions: np.ndarray,
    neighbors: list[list[int]],
    josephson_energy_func: Callable,
    vector_potential_func: Callable,
    vector_potential_points: int = 21,
    rng: Optional[np.random.Generator] = None,
    source_points: Optional[np.ndarray] = None,
    drain_points: Optional[np.ndarray] = None,
    source_drain_current: Optional[float] = None,
    phase_initializer: Optional[Callable] = None,
) -> ModelInfo:
    """Generates the graph structure and populates it with gekko objects.

    Args:
        gekko_model: gekko.GEKKO model object.
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
        phase_initializer: A callable with signature
            ``phase_initializer() -> guess_phase``, producing initial guesses for
            island phases.
    """
    print("Generating junctions and building model...")
    if rng is None:
        rng = np.random.default_rng()

    if phase_initializer is None:
        phase_initializer = lambda: 0.0  # noqa: E731

    m = gekko_model

    if source_points is None:
        assert drain_points is None
        assert source_drain_current is None
        source_nodes = []
        drain_nodes = []
    else:
        assert drain_points is not None
        assert source_drain_current is not None
        source_nodes = contains_points(source_points, island_positions, index=True)
        drain_nodes = contains_points(drain_points, island_positions, index=True)

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
            phase=m.Var(
                value=phase_initializer(),
                # lb=0,
                # ub=2 * np.pi,
                name=f"p{i}",
            ),
        )
    # Popualate junctions (edges)
    for i, indices in enumerate(neighbors):
        for j in indices:
            if (i, j) not in graph.edges and (j, i) not in graph.edges:
                graph.add_edge(i, j)
    nodes = graph.nodes
    # Store gekko objects as graph edge attributes.
    edge_attrs = {}
    for n, (i, j) in tqdm(
        enumerate(graph.edges),
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
        length = la.norm(r2 - r1)
        attrs = dict(
            length=length,
            EJ=josephson_energy_func(length),
            # delta[i, j] is the phase difference between islands i and j:
            # delta[i, j] = phase[j] - phase[i]
            delta=m.Intermediate(
                n1["phase"] - n2["phase"],
                name=f"dp{n}",
            ),
            A=Aij,
        )
        # attrs["delta"] = m.Var(value=2 * np.pi * Aij),
        # m.Equation(attrs["delta"] == (n2["phase"] - n1["phase"]))
        attrs["Ic"] = 2 * np.pi / Phi_0 * attrs["EJ"]
        attrs["theta"] = m.Intermediate(
            # attrs["delta"] - (2 * np.pi * attrs["A"]),
            2 * m.atan(m.tan((attrs["delta"] - (2 * np.pi * attrs["A"])) / 2)),
            name=f"dA{n}",
        )
        attrs["current"] = m.Intermediate(
            attrs["Ic"] * m.sin(attrs["theta"]),
            name=f"c{n}",
        )
        attrs["energy"] = m.Intermediate(
            attrs["EJ"] * (1 - m.cos(attrs["theta"])) / eV,
            name=f"e{n}",
        )
        edge_attrs[(i, j)] = attrs
    nx.set_edge_attributes(graph, edge_attrs)

    # Apply current conservation constraint
    # One equality constraint per node
    node_current_in = defaultdict(list)
    node_current_out = defaultdict(list)
    for i, j, current in graph.edges.data("current"):
        node_current_in[j].append(current)
        node_current_out[i].append(current)
    if source_drain_current is not None:
        for i in source_nodes:
            m.Equation(
                1e0 * (sum(node_current_in[i]) - sum(node_current_out[i]))
                == -1e0 * source_drain_current
            )
        for i in drain_nodes:
            m.Equation(
                1e0 * (sum(node_current_in[i]) - sum(node_current_out[i]))
                == +1e0 * source_drain_current
            )
    num_nodes = graph.number_of_nodes()
    for i in tqdm(graph.nodes, total=num_nodes, desc="Applying current conservation"):
        if i in source_nodes or i in drain_nodes:
            continue
        m.Equation(1e0 * sum(node_current_in[i]) == 1e0 * sum(node_current_out[i]))

    # Apply phase single-valuedness constraint
    # One equality constraint per basis cycle
    edges = graph.edges
    loops = basis_loops(graph)
    loop_info = []
    for n, loop in tqdm(
        enumerate(loops), total=len(loops), desc="Applying phase single-valuedness"
    ):
        loop_phases = []
        loop_deltas = []
        applied_flux = []
        # excess_flux = m.Var(value=0, lb=0, ub=1, name=f"ef{n}")
        for i, j in round_trip(loop):
            # Needed to get the correct sign for all edges in the directed graph.
            if (i, j) in edges:
                loop_deltas.append(+1 * edges[i, j]["delta"])
                loop_phases.append(+1 * edges[i, j]["theta"])
                applied_flux.append(+1 * edges[i, j]["A"])
            else:
                loop_deltas.append(-1 * edges[j, i]["delta"])
                loop_phases.append(-1 * edges[j, i]["theta"])
                applied_flux.append(-1 * edges[j, i]["A"])
        loop_deltas = sum(loop_deltas)
        applied_flux = sum(applied_flux)
        excess_flux = m.Var(
            value=0 * applied_flux,
            # lb=min(applied_flux, 0),
            # ub=max(0, applied_flux),
            name=f"ef{n}",
        )
        # vortices = m.Var(
        #     value=init_vorticity,
        #     # lb=-abs_ceil,
        #     # ub=+abs_ceil,
        #     integer=True,
        #     name=f"f{n}",
        # )
        loop_phase = sum(loop_phases)
        loop_info.append(
            LoopInfo(
                nodes=loop,
                applied_flux=applied_flux,
                gauge_invariant_phase=loop_phase,
                frustration=applied_flux,
                # total_flux=enclosed_flux,
                # vortices=vortices,
            )
        )
        # m.Equation(excess_flux + vortices == applied_flux)
        # m.Equation(loop_phase / (2 * np.pi) == vortices - applied_flux)
        m.Equation(sum(loop_phases) / (2 * np.pi) == excess_flux)
        # m.Equation(loop_deltas / (2 * np.pi) == vortices)
        # m.Equation(m.sin(loop_phase) == m.sin((2 * np.pi) * total_flux))

    # # Build Objective
    # # Not needed because number of DOF == 0.
    # m.Minimize(m.sum([energy for _, _, energy in graph.edges.data("energy")]))

    model_info = ModelInfo(graph=graph, loops=loop_info)

    msg = (
        f"Finished building model with {graph.number_of_nodes()} islands, "
        f"{graph.number_of_edges()} junctions, and {len(loops)} loops."
    )
    print(msg)
    return model_info


def calculate_loop_info(
    graph: nx.DiGraph, length: Optional[int] = None
) -> list[LoopInfo]:
    """Generates LoopInfo instances for all basis loops in a network."""
    if length is None:
        loops = basis_loops(graph)
    else:
        loops = find_all_cells(graph, length)
    edges = graph.edges
    loop_info = []
    for loop in loops:
        current = []
        applied_flux = []
        loop_theta = []
        loop_delta = []
        for i, j in round_trip(loop):
            if (i, j) in edges:
                current.append(+1 * get_scalar(edges[i, j]["current"]))
                applied_flux.append(+1 * get_scalar(edges[i, j]["A"]))
                loop_theta.append(+1 * get_scalar(edges[i, j]["theta"]))
                loop_delta.append(+1 * get_scalar(edges[i, j]["delta"]))
            else:
                current.append(-1 * get_scalar(edges[j, i]["current"]))
                applied_flux.append(-1 * get_scalar(edges[j, i]["A"]))
                loop_theta.append(-1 * get_scalar(edges[j, i]["theta"]))
                loop_delta.append(-1 * get_scalar(edges[j, i]["delta"]))
        loop_current = sum(current)
        loop_theta = sum(loop_theta)
        loop_delta = sum(loop_delta)
        frustration = applied_flux = sum(applied_flux)
        vortices = applied_flux + loop_theta / (2 * np.pi)
        # vortices = applied_flux + loop_theta / (2 * np.pi)
        loop_info.append(
            LoopInfo(
                nodes=loop,
                current=loop_current,
                applied_flux=applied_flux,
                gauge_invariant_phase=loop_theta,
                frustration=frustration,
                vortices=vortices,
            )
        )
    return loop_info


class JosephsonNetwork(ABC):
    """An abstract base class for Josephson network models.

    Args:
        directory: The base directory in which to save results.
        island_positions: An array of (x, y) island positions in ``length_units``.
        length_units: Pint-parseable string representing the units used for lengths.
        rng_seed: An integer used to seed the random number generator for island phase
            initialization. If set to zero, phases will be initialized to zero. If set
            to -1, rng_seed will be changed to the timestamp of the start of the
            simulation.
        gekko_local: Whether to run gekko locally.
        gekko_verbose: An integer indicating the gekko verbosity level.
    """

    ureg = ureg

    META_ATTRS = [
        "outdir",
        "source_points",
        "drain_points",
        "source_drain_current",
        "length_units",
        "rng_seed",
        "gekko_remote",
        "gekko_verbose",
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
        source_drain_current: Optional[str] = None,
        length_units: str = "um",
        rng_seed: int = -1,
        gekko_local: bool = True,
        gekko_verbose: int = 10,
    ):
        self.directory = os.path.abspath(directory)
        # Ensure a unique directory name for each simulation
        i = 0
        run_start = datetime.now()
        outdir = os.path.join(directory, run_start.strftime(DTFORMAT) + f"_{i:03}")
        while os.path.exists(outdir):
            i += 1
            run_start = datetime.now()
            outdir = os.path.join(directory, run_start.strftime(DTFORMAT) + f"_{i:03}")
        self.outdir = os.path.abspath(outdir)
        os.makedirs(self.outdir)
        self.json_file = os.path.join(outdir, "metadata.json")
        self.timing = TimingInfo(run_start=run_start)
        self.length_units = ureg(length_units)
        self.gekko_local = gekko_local
        self.gekko_verbose = gekko_verbose
        self.rng_seed = int(rng_seed)
        if self.rng_seed == -1:
            self.rng_seed = int(self.timing.run_start.timestamp())
        self.rng = np.random.default_rng(seed=self.rng_seed)
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
        self.source_points = source_points
        self.drain_points = drain_points
        self.source_drain_current = source_drain_current
        if self.source_drain_current is not None:
            self.source_drain_current = ureg(self.source_drain_current)
        self.neighbors = None
        self.model_info = None
        self.gekko_model = None
        self.compute_neighbors()

    def compute_neighbors(self) -> None:
        """Removes any overlapping or isolated islands."""
        # Replace any set of overlapping islands with a single island located at
        # the mean position of the set of islands.
        self.island_positions = remove_overlapping_islands(
            self.island_positions,
            self.island_diameter,
        )
        print(
            f"Total network size after removing overlapping islands: "
            f"{self.island_positions.shape[0]} islands."
        )
        self.island_positions, self.neighbors = remove_isolated_islands(
            self.island_positions,
            self.junction_cutoff_radius,
        )
        print(
            f"Total network size after removing isolated islands: "
            f"{self.island_positions.shape[0]} islands."
        )
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
    def gekko_remote(self) -> bool:
        return not self.gekko_local

    @property
    def graph(self) -> Optional[nx.DiGraph]:
        if self.model_info is None:
            return None
        return self.model_info.graph

    @property
    def loops(self) -> Optional[LoopInfo]:
        if self.model_info is None:
            return None
        return self.model_info.loops

    def metadata(self, **kwargs) -> dict[str, Any]:
        meta = kwargs.copy()
        meta.update({attr: getattr(self, attr) for attr in self.META_ATTRS})
        meta["timing"] = asdict(self.timing)
        return meta

    def solver_options(
        self, solver: str, mixed_integer: bool = False
    ) -> tuple[int, list[str]]:
        """Returns the appropriate solver ID and options for a given NLP problem type."""
        solver = solver.lower()
        assert solver in {"ipopt", "apopt"}
        if mixed_integer:
            assert solver == "apopt"
            solver_id = 1
            options = [
                "minlp_as_nlp 0",
                f"minlp_print_level {min(self.gekko_verbose, 10)}",
                "minlp_integer_tol 1.0e-3",
                "minlp_gap_tol 1.0e-3",
                "minlp_branch_method 3",
                # "minlp_integer_leaves 3",
                # "minlp_integer_max 1.0e3",
                # "nlp_maximum_iterations 1000",
                # "minlp_max_iter_with_int_sol 1000",
            ]
            return solver_id, options
        elif solver == "apopt":
            solver_id = 1
            options = [
                "minlp_as_nlp 1",
                f"minlp_print_level {min(self.gekko_verbose, 10)}",
            ]
            # solver_id = 2
            # options = None
        else:
            solver_id = 3
            options = [
                "nlp_scaling_method gradient-based",
                "ma57_automatic_scaling yes",
                f"print_level {max(self.gekko_verbose, 0)}",
                "least_square_init_duals yes",
                "least_square_init_primal yes",
            ]
            # solver_id = 2
            # options = None
        return solver_id, options

    def build_model(self) -> None:
        """Generates the graph representing the Josephson network
        and defines the NLP problem.
        """
        n_neighbors = len(self.neighbors)
        n_islands = len(self.island_positions)
        if n_neighbors != n_islands:
            raise ValueError(
                f"The number of list of neighbors ({n_neighbors}) does not equal "
                f"the number of islands ({n_islands})."
            )
        self.timing.build_start = datetime.now()

        self.gekko_model = GEKKO(
            name=os.path.basename(self.outdir), remote=self.gekko_remote
        )

        if self.rng_seed:
            init_phase = lambda: 1e-2 * (self.rng.random() - 0.5)  # noqa: E731
        else:
            init_phase = None  # noqa: E731

        island_positions = (self.island_positions * self.length_units).to("m").magnitude
        if self.source_points is None:
            source_points = self.source_points
            drain_points = self.drain_points
            source_drain_current = self.source_drain_current
        else:
            source_points = (self.source_points * self.length_units).to("m").magnitude
            drain_points = (self.drain_points * self.length_units).to("m").magnitude
            source_drain_current = self.source_drain_current.to("A").magnitude

        self.model_info = build_graph(
            gekko_model=self.gekko_model,
            island_positions=island_positions,
            neighbors=self.neighbors,
            josephson_energy_func=self.josephson_energy,
            vector_potential_func=self.vector_potential,
            source_points=source_points,
            drain_points=drain_points,
            source_drain_current=source_drain_current,
            phase_initializer=init_phase,
            rng=self.rng,
        )

        fig, ax = draw_graph(self.graph)
        ax.set_title(os.path.basename(self.outdir))
        fig.savefig(os.path.join(self.outdir, "graph.pdf"), bbox_inches="tight")
        plt.close(fig)

        metadata = self.metadata()
        with open(self.json_file, "w") as f:
            json.dump(metadata, f, indent=4, sort_keys=True, cls=NumpyJSONEncoder)

        print(
            f"Finished building model in python. Elapsed time: "
            f"{(datetime.now()-self.timing.build_start).total_seconds():.3f} seconds."
        )

    def solve(self) -> None:
        """Solves the NLP problem."""
        self.timing.gekko_start = datetime.now()
        m = self.gekko_model
        print("Solving model...\n", flush=True)
        m.options.IMODE = 1  # Steady-state simulation, number of DOF == 0
        m.options.DIAGLEVEL = 1
        m.options.MAX_ITER = 5000
        m.options.SCALING = 1
        m.options.MAX_MEMORY = 6
        m.options.REDUCE = 100
        m.options.RTOL = 1e-6
        m.options.OTOL = 1e-6
        m._path = self.outdir

        if self.gekko_remote:
            # Use IPOPT
            solver_id, solver_options = self.solver_options("ipopt")
            m.options.SOLVER = solver_id
            m.solver_options = solver_options
            print("Solving NLP problem with IPOPT...")
        else:
            # IPOPT not supported for local solve
            solver_id, solver_options = self.solver_options(
                "apopt", mixed_integer=False
            )
            m.options.SOLVER = solver_id
            m.solver_options = solver_options
            print("Solving NLP problem with APOPT...")

        m.solve(disp=True, debug=2)

        if False:
            solver_id, solver_options = self.solver_options("apopt", mixed_integer=True)
            m.options.SOLVER = solver_id
            m.solver_options = solver_options
            print("Solving MINLP problem with APOPT...")
            m.solve(disp=True, debug=2)

        self.timing.gekko_stop = datetime.now()

    def process_results(self) -> None:
        """Extracts results from the graph and saves them to disk."""
        graph = self.graph
        outdir = self.outdir
        print("Extracting and saving edge data...")

        with pd.HDFStore(os.path.join(self.outdir, "results.h5")) as store:
            df = edge_data_to_df(graph)
            store["edge_data"] = df

            self.model_info.loops = loop_info = calculate_loop_info(graph, 4)
            loops = np.array([loop.nodes for loop in loop_info])
            frustration = np.array([get_scalar(loop.frustration) for loop in loop_info])
            vortices = np.array([get_scalar(loop.vortices) for loop in loop_info])
            applied_flux = np.array(
                [get_scalar(loop.applied_flux) for loop in loop_info]
            )
            gauge_invariant_phase = np.array(
                [get_scalar(loop.gauge_invariant_phase) for loop in loop_info]
            )
            # total_flux = np.array([get_scalar(loop.total_flux) for loop in loop_info])
            (nonzero_loops,) = np.where(np.abs(vortices) > 1e-3)
            loop_data = {f"node{i}": loops[:, i] for i in range(loops.shape[1] - 1)}
            loop_data.update(
                {
                    # "total_flux": total_flux,
                    "gauge_invariant_phase": gauge_invariant_phase,
                    "frustration": frustration,
                    "vortices": vortices,
                    "applied_flux": applied_flux,
                }
            )
            df_loops = pd.DataFrame(loop_data)
            store["loops"] = df_loops
            store["nonzero_loops"] = pd.Series(nonzero_loops)

        total_energy = sum(
            get_scalar(energy) for _, _, energy in graph.edges.data("energy")
        )
        print(f"Total energy: {total_energy:.3e} eV.")

        fig, axes = draw_currents(df, linewidth=3, cmap="inferno")
        fig.suptitle(outdir)
        fig.savefig(os.path.join(outdir, "currents.pdf"), bbox_inches="tight")
        plt.close(fig)

        self.timing.run_stop = datetime.now()
        self.timing.run_time = (
            self.timing.run_stop - self.timing.run_start
        ).total_seconds()
        solve_time = (self.timing.gekko_stop - self.timing.gekko_start).total_seconds()
        print(f"Total solve time: {solve_time:.3f} seconds.")
        print(f"Total run time: {self.timing.run_time:.3f} seconds.")

        metadata = self.metadata()
        metadata["energy"] = f"{total_energy:.6e} eV"
        metadata["vortices"] = {}
        for n, loop_info in enumerate(self.loops):
            v = get_scalar(loop_info.vortices)
            if np.abs(v) > 1e-3:
                metadata["vortices"][n] = v

        with open(self.json_file, "w") as f:
            json.dump(metadata, f, indent=4, sort_keys=True, cls=NumpyJSONEncoder)

    def post_process(self) -> None:
        """Can be implemented by subclasses to perform additional processing after
        ``self.process_results()`` has been called.
        """
        pass

    def cleanup(self) -> None:
        """Removes unwanted solver-related files."""
        files_to_keep = {
            "infeasibilities.txt",
            "results.h5",
            "graph.pdf",
            "currents.pdf",
            "metadata.json",
            "ipopt.opt",
            "apopt_current_options.opt",
            "APOPT.out",
        }
        for name in os.listdir(self.outdir):
            if name in files_to_keep:
                continue
            print(f"Removing {name}...")
            try:
                os.remove(os.path.join(self.outdir, name))
            except Exception as e:
                print(f"Unable to remove {name}: {e}.")

    def run(self) -> None:
        """Builds the model, solves it, processes results, and cleans up."""
        self.build_model()
        self.solve()
        self.process_results()
        self.post_process()
        self.cleanup()
