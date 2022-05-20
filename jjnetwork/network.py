from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os
from typing import Any, Callable, Optional

from gekko import GEKKO
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.linalg as la
from tqdm import tqdm

from .em import ureg, Phi_0
from .graph_utils import (
    get_scalar,
    basis_loops,
    find_all_cells,
    draw_graph,
    draw_currents,
    edge_data_to_df,
    round_trip,
)
from .io import DTFORMAT, NumpyJSONEncoder


@dataclass
class LoopInfo:
    """A container for data related to a single loop (closed path) in a network."""

    nodes: list[int]
    applied_flux: float
    total_flux: Optional[float] = None
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
    if phase_initializer is None:
        phase_initializer = lambda: 0.0  # noqa: E731

    m = gekko_model

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
                lb=0,
                ub=2 * np.pi,
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
                n2["phase"] - n1["phase"],
                name=f"dp{n}",
            ),
            A=Aij,
        )
        attrs["Ic"] = 2 * np.pi / Phi_0 * attrs["EJ"]
        attrs["theta"] = m.Intermediate(
            attrs["delta"] - (2 * np.pi * attrs["A"]), name=f"dA{n}"
        )
        attrs["current"] = m.Intermediate(
            attrs["Ic"] * m.sin(attrs["theta"]),
            name=f"c{n}",
        )
        attrs["energy"] = m.Intermediate(
            attrs["EJ"] * (1 - m.cos(attrs["theta"])),
            name=f"e{n}",
        )
        edge_attrs[(i, j)] = attrs
    nx.set_edge_attributes(graph, edge_attrs)

    # Apply current conservation constraint
    # One equality constraint per island
    node_current_in = defaultdict(list)
    node_current_out = defaultdict(list)
    for i, j, current in graph.edges.data("current"):
        node_current_in[j].append(current)
        node_current_out[i].append(current)
    num_nodes = graph.number_of_nodes()
    for i in tqdm(graph.nodes, total=num_nodes, desc="Applying current conservation"):
        m.Equation(sum(node_current_in[i]) == sum(node_current_out[i]))

    # Apply phase single-valuedness constraint
    # One equality constraint per basis cycle
    edges = graph.edges
    loops = basis_loops(graph)
    loop_info = []
    for n, loop in tqdm(
        enumerate(loops), total=len(loops), desc="Applying phase single-valuedness"
    ):
        enclosed_flux = m.Var(value=0, name=f"f{n}")
        loop_phases = []
        applied_flux = []
        for i, j in round_trip(loop):
            # Needed to get the correct sign for all edges in the directed graph.
            if (i, j) in edges:
                loop_phases.append(edges[i, j]["theta"])
                applied_flux.append(edges[i, j]["A"])
            else:
                loop_phases.append(-1 * edges[j, i]["theta"])
                applied_flux.append(-1 * edges[j, i]["A"])
        loop_info.append(
            LoopInfo(
                nodes=loop,
                applied_flux=sum(applied_flux),
                total_flux=enclosed_flux,
            )
        )
        m.Equation(sum(loop_phases) == (2 * np.pi) * enclosed_flux)

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
        applied_flux = []
        loop_theta = []
        for i, j in round_trip(loop):
            if (i, j) in edges:
                loop_theta.append(get_scalar(edges[i, j]["theta"]))
                applied_flux.append(get_scalar(edges[i, j]["A"]))
            else:
                loop_theta.append(-get_scalar(edges[j, i]["theta"]))
                applied_flux.append(-get_scalar(edges[j, i]["A"]))
        total_flux = sum(loop_theta) / (2 * np.pi)
        applied_flux = sum(applied_flux)
        frustration = np.fmod(total_flux, 1)
        loop_info.append(
            LoopInfo(
                nodes=loop,
                applied_flux=applied_flux,
                total_flux=total_flux,
                frustration=frustration,
                vortices=(frustration - total_flux),
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

    META_ATTRS = [
        "outdir",
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
        length_units: str = "um",
        rng_seed: int = -1,
        gekko_local: bool = True,
        gekko_verbose: int = 5,
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
        self.neighbors = None
        self.model_info = None
        self.gekko_model = None
        self.compute_neighbors()

    @abstractmethod
    def compute_neighbors(self) -> None:
        """Define neighboring or adjacent islands (nodes).

        This method must update ``self.neighbors`` and can optionally update
        ``self.island_positions``. This method must ensure that
        ``len(self.neighbors) == len(self.island_positions)``.
        """
        pass

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
                "minlp_integer_max 1.0e3",
                "minlp_branch_method 1",
                "nlp_maximum_iterations 1000",
                "minlp_max_iter_with_int_sol 1000",
            ]
            return solver_id, options
        elif solver == "apopt":
            solver_id = 1
            options = [
                "minlp_as_nlp 1",
                f"minlp_print_level {min(self.gekko_verbose, 10)}",
            ]
        else:
            solver_id = 3
            options = [
                "nlp_scaling_method gradient-based",
                "ma57_automatic_scaling yes",
                f"print_level {max(self.gekko_verbose, 0)}",
            ]
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
            init_phase = lambda: 1e-2 * self.rng.random()  # noqa: E731
        else:
            init_phase = lambda: 0  # noqa: E731

        island_positions = (self.island_positions * self.length_units).to("m").magnitude
        self.model_info = build_graph(
            gekko_model=self.gekko_model,
            island_positions=island_positions,
            neighbors=self.neighbors,
            josephson_energy_func=self.josephson_energy,
            vector_potential_func=self.vector_potential,
            phase_initializer=init_phase,
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
            total_flux = np.array([get_scalar(loop.total_flux) for loop in loop_info])
            (nonzero_loops,) = np.where(np.abs(vortices) > 1e-3)
            loop_data = {f"node{i}": loops[:, i] for i in range(loops.shape[1] - 1)}
            loop_data.update(
                {
                    "total_flux": total_flux,
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
        print(f"Total energy: {total_energy:.3e} joules.")

        fig, axes = draw_currents(df, linewidth=3, cmap="inferno")
        fig.suptitle(outdir)
        fig.savefig(os.path.join(outdir, "currents.pdf"), bbox_inches="tight")
        plt.close(fig)

        self.timing.run_stop = datetime.now()
        self.timing.run_time = (
            self.timing.run_stop - self.timing.run_start
        ).total_seconds()
        solve_time = (self.timing.gekko_stop - self.timing.gekko_stop).total_seconds()
        print(f"Total solve time: {solve_time:.3f} seconds.")
        print(f"Total run time: {self.timing.run_time:.3f} seconds.")

        metadata = self.metadata()
        metadata["energy"] = total_energy
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
