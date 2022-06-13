from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
import pyomo.environ as pyo
import scipy.linalg as la
from tqdm import tqdm

from .graph_utils import basis_loops, round_trip
from .em import (
    Phi_0,
    eV,
    # mutual_vector_potential_matrix,
)


@dataclass
class LoopInfo:
    """A container for data related to all loops (closed paths) in a network."""

    nodes: list[int] = field(default_factory=list)
    applied_flux: list[float] = field(default_factory=list)
    current: list[float] = field(default_factory=list)
    gauge_invariant_phase: list[float] = field(default_factory=list)
    vortices: list[float] = field(default_factory=list)


@dataclass
class ModelInfo:
    graph: Optional[nx.DiGraph] = None
    loops: Optional[LoopInfo] = None
    model: Optional[pyo.ConcreteModel] = None


def nodes_out_init(m, node):
    for i, j in m.edges:
        if i == node:
            yield j


def nodes_in_init(m, node):
    for i, j in m.edges:
        if j == node:
            yield i


def principal_value(theta, offset=0):
    return offset + 2 * np.arctan(np.tan((theta + offset) / 2))


def principal_value_pyo(theta, offset=0):
    return offset + 2 * pyo.atan(pyo.tan((theta + offset) / 2))


def theta_rule(m, *edge):
    i, j = edge
    return principal_value_pyo(m.phase[i] - m.phase[j] - 2 * np.pi * m.Aij[edge])


def loop_theta_rule(m, loop):
    thetas = []
    for i, j in round_trip(m.loop_nodes[loop]):
        theta = m.theta[i, j] if (i, j) in m.edges else -m.theta[j, i]
        thetas.append(theta)
    return sum(thetas)


def applied_flux_rule(m, loop):
    Aijs = []
    for i, j in round_trip(m.loop_nodes[loop]):
        Aij = m.Aij[i, j] if (i, j) in m.edges else -m.Aij[j, i]
        Aijs.append(Aij)
    return sum(Aijs)


def critical_current_rule(m, *edge):
    return 2 * np.pi / Phi_0 * m.EJ[edge]


def current_rule(m, *edge):
    return m.current_scale * m.Ic[edge] * pyo.sin(m.theta[edge])


def energy_rule(m, *edge):
    return 2 * np.pi * m.EJ[edge] * (1 - pyo.cos(m.theta[edge])) / eV


def node_current_rule(m, node):
    return sum(m.current[i, node] for i in m.nodes_in[node]) - sum(
        m.current[node, j] for j in m.nodes_out[node]
    )


def current_conservation_rule(m, node):
    return m.node_current[node] == 0


def flux_quantization_rule(m, loop):
    return m.loop_theta[loop] / (2 * np.pi) == m.vortices[loop] - m.applied_flux[loop]


def source_rule(m, node):
    return m.node_current[node] == -m.current_scale * m.source_drain_current


def drain_rule(m, node):
    return m.node_current[node] == +m.current_scale * m.source_drain_current


def induced_Aij_rule(m, *edge):
    return m.Aij_induced[edge] == sum(
        m.MAij[edge + source_edge] * m.current[source_edge] / m.current_scale
        for source_edge in m.edges
    )


def total_Aij_rule(m, *edge):
    return m.Aij_applied[edge] + m.Aij_induced[edge]


junction_network = pyo.AbstractModel()

junction_network.current_scale = pyo.Param(initialize=1e6, mutable=True)
junction_network.energy_scale = pyo.Param(initialize=1e4, mutable=True)

junction_network.nodes = pyo.Set()
junction_network.source_nodes = pyo.Set(within=junction_network.nodes)
junction_network.drain_nodes = pyo.Set(within=junction_network.nodes)
junction_network.boundary_nodes = (
    junction_network.source_nodes | junction_network.drain_nodes
)
junction_network.edges = pyo.Set(dimen=2)
junction_network.nodes_out = pyo.Set(junction_network.nodes, initialize=nodes_out_init)
junction_network.nodes_in = pyo.Set(junction_network.nodes, initialize=nodes_in_init)
junction_network.loops = pyo.Set()
junction_network.loop_nodes = pyo.Set(junction_network.loops)

junction_network.phase = pyo.Var(junction_network.nodes, within=pyo.Reals, initialize=0)
junction_network.xs = pyo.Param(junction_network.nodes, within=pyo.Reals)
junction_network.ys = pyo.Param(junction_network.nodes, within=pyo.Reals)
junction_network.source_drain_current = pyo.Param(default=0, within=pyo.Reals)

junction_network.EJ = pyo.Param(junction_network.edges, within=pyo.NonNegativeReals)

junction_network.MAij = pyo.Param(
    junction_network.edges * junction_network.edges,
    within=pyo.Reals,
    default=0,
)
junction_network.Aij_applied = pyo.Param(junction_network.edges, within=pyo.Reals)
# junction_network.Aij_induced = pyo.Var(
#     junction_network.edges, within=pyo.Reals, initialize=0
# )
junction_network.Aij_induced = pyo.Param(
    junction_network.edges, within=pyo.Reals, initialize=0
)
junction_network.Aij = pyo.Expression(junction_network.edges, rule=total_Aij_rule)

junction_network.theta = pyo.Expression(junction_network.edges, rule=theta_rule)
junction_network.Ic = pyo.Expression(junction_network.edges, rule=critical_current_rule)
junction_network.current = pyo.Expression(junction_network.edges, rule=current_rule)
junction_network.energy = pyo.Expression(junction_network.edges, rule=energy_rule)

# junction_network.Aij_induced_constraint = pyo.Constraint(
#     junction_network.edges,
#     rule=induced_Aij_rule,
# )

junction_network.node_current = pyo.Expression(
    junction_network.nodes,
    rule=node_current_rule,
)

junction_network.current_conservation = pyo.Constraint(
    junction_network.nodes - junction_network.boundary_nodes,
    rule=current_conservation_rule,
)
junction_network.source_constraint = pyo.Constraint(
    junction_network.source_nodes,
    rule=source_rule,
)
junction_network.drain_constraint = pyo.Constraint(
    junction_network.drain_nodes,
    rule=drain_rule,
)
junction_network.applied_flux = pyo.Expression(
    junction_network.loops,
    rule=applied_flux_rule,
)
junction_network.vortices = pyo.Var(
    junction_network.loops,
    within=pyo.Reals,
)
junction_network.loop_theta = pyo.Expression(
    junction_network.loops,
    rule=loop_theta_rule,
)
junction_network.flux_quantization = pyo.Constraint(
    junction_network.loops,
    rule=flux_quantization_rule,
)


def squared_current_nonconservation(m):
    """Squared current conservation violation for all nodes
    other than source-drain nodes.
    """
    return pyo.summation(
        m.node_current,
        m.node_current,
        # index=junction_network.nodes - junction_network.boundary_nodes,
    )


def objective_flexible(m):
    return pyo.summation(m.energy) * (
        m.energy_scale + squared_current_nonconservation(m)
    )


def objective_strict(m):
    return pyo.summation(m.energy)


junction_network.objective_flexible = pyo.Objective(
    rule=objective_flexible,
    sense=pyo.minimize,
)

junction_network.objective_strict = pyo.Objective(
    rule=objective_strict,
    sense=pyo.minimize,
)


def graph_to_model(
    graph: nx.DiGraph,
    source_nodes: Optional[np.ndarray] = None,
    drain_nodes: Optional[np.ndarray] = None,
    source_drain_current: Optional[float] = None,
    # include_screening: bool = True,
) -> ModelInfo:
    """Populates a ``junction_network`` model from a directed graph."""
    if source_drain_current is None:
        source_drain_current = 0
    loops = basis_loops(graph)
    node_attrs = list(graph.nodes[next(iter(graph.nodes))])
    edge_attrs = list(graph.edges[next(iter(graph.edges))])
    positions = np.stack([p for _, p in sorted(graph.nodes.data("position"))], axis=0)
    loop_indices = list(range(len(loops)))
    edges = list(graph.edges)
    model_data = {
        "nodes": {None: list(graph.nodes)},
        "edges": {None: edges},
        "loops": {None: loop_indices},
        "source_nodes": {None: source_nodes},
        "drain_nodes": {None: drain_nodes},
        "source_drain_current": {None: source_drain_current},
    }
    model_data["loop_nodes"] = {i: loop[:-1] for i, loop in enumerate(loops)}
    model_data["xs"] = dict(enumerate(positions[:, 0]))
    model_data["ys"] = dict(enumerate(positions[:, 1]))
    for name in node_attrs:
        model_data[name] = {i: val for i, val in graph.nodes.data(name)}
    for name in edge_attrs:
        model_data[name] = {(i, j): val for i, j, val in graph.edges.data(name)}

    # if include_screening:
    #     print("Calculating mutual vector potential matrix...")
    #     MAij = mutual_vector_potential_matrix(graph).to("Phi_0 / A").magnitude
    #     model_data["MAij"] = {
    #         edges[kl] + edges[ij]: A for (kl, ij), A in np.ndenumerate(MAij)
    #     }

    print("Constructing Pyomo model...")
    model = junction_network.create_instance({None: model_data})

    # if not include_screening:
    #     for edge in model.edges:
    #         model.Aij_induced[edge].fix(0)
    #     model.Aij_induced_constraint.deactivate()

    current_scale = pyo.value(model.current_scale)
    loop_info = LoopInfo()
    edges = graph.edges
    for ell, loop in enumerate(tqdm(loops, desc="Populating loops")):
        current = []
        loop_thetas = []
        for i, j in round_trip(loop):
            # Needed to get the correct sign for all edges in the directed graph.
            if (i, j) in edges:
                current.append(model.current[i, j])
                loop_thetas.append(model.theta[i, j])
            else:
                current.append(-model.current[j, i])
                loop_thetas.append(-model.theta[j, i])
        loop_theta = sum(loop_thetas)
        applied_flux = model.applied_flux[ell]
        loop_info.nodes.append(loop)
        loop_info.current.append(sum(current) / current_scale)
        loop_info.applied_flux.append(applied_flux)
        loop_info.gauge_invariant_phase.append(loop_theta)
        loop_info.vortices.append(model.vortices[ell])
    return ModelInfo(graph, loop_info, model=model)


def model_to_graph(model: pyo.ConcreteModel) -> nx.DiGraph:
    """Populates a directed graph from a ``junction_network`` model."""

    def get(x):
        return pyo.value(x)

    current_scale = get(model.current_scale)

    graph = nx.DiGraph()
    for node in model.nodes:
        graph.add_node(
            node,
            position=np.array([get(model.xs[node]), get(model.ys[node])]),
            phase=get(model.phase[node]),
            current=(
                sum(get(model.current[i, node]) for i in model.nodes_in[node])
                - sum(get(model.current[node, j]) for j in model.nodes_out[node])
            ),
        )
    for i, j in model.edges:
        graph.add_edge(
            i,
            j,
            length=la.norm(
                get(graph.nodes[j]["position"]) - get(graph.nodes[i]["position"])
            ),
            EJ=get(model.EJ[i, j]),
            Ic=get(model.Ic[i, j]),
            Aij=get(model.Aij[i, j]),
            Aij_applied=get(model.Aij_applied[i, j]),
            Aij_induced=get(model.Aij_induced[i, j]),
            theta=get(model.theta[i, j]),
            current=get(model.current[i, j]) / current_scale,
            energy=get(model.energy[i, j]),
        )
    return graph


def calculate_loop_info(graph: nx.DiGraph, loops: list[list[int]]) -> LoopInfo:
    """Generates LoopInfo for all specified loops in a network."""
    edges = graph.edges
    loop_info = LoopInfo()
    for loop in loops:
        current = []
        applied_flux = []
        loop_thetas = []
        for i, j in round_trip(loop):
            if (i, j) in edges:
                current.append(edges[i, j]["current"])
                applied_flux.append(edges[i, j]["Aij"])
                loop_thetas.append(edges[i, j]["theta"])
            else:
                current.append(-edges[j, i]["current"])
                applied_flux.append(-edges[j, i]["Aij"])
                loop_thetas.append(-edges[j, i]["theta"])
        loop_theta = sum(loop_thetas)
        applied_flux = sum(applied_flux)
        loop_info.nodes.append(loop)
        loop_info.current.append(sum(current))
        loop_info.applied_flux.append(applied_flux)
        loop_info.gauge_invariant_phase.append(loop_theta)
        loop_info.vortices.append(sum(loop_thetas) / (2 * np.pi) + applied_flux)
    return loop_info


def initialize_variables(
    model: pyo.ConcreteModel,
    rng: Optional[np.random.Generator] = None,
) -> None:
    if rng is None:
        rng = np.random.default_rng()

    # applied_flux = np.array(
    #     [pyo.value(model.applied_flux[loop]) for loop in model.loops]
    # )
    # max_flux = np.max(np.abs(applied_flux))
    # median_flux = np.median(np.abs(applied_flux))
    # mean_flux = np.mean(np.abs(applied_flux))
    # print(f"Max applied flux: {max_flux:.5f} Phi_0")
    # print(f"Median applied flux: {median_flux:.5f} Phi_0")
    # print(f"Mean applied flux: {mean_flux:.5f} Phi_0")

    max_Aij = max(abs(model.Aij_applied[edge]) for edge in model.edges)
    print(f"Max Aij: {max_Aij:.4f}")

    for node in model.nodes:
        model.phase[node].value = rng.normal(loc=0, scale=2 * np.pi * max_Aij)


def set_model_flexible(model: pyo.ConcreteModel) -> None:
    max_Ic = max(pyo.value(model.Ic[edge]) for edge in model.edges)
    model.objective_flexible.activate()
    model.objective_strict.deactivate()
    model.current_conservation.deactivate()
    model.current_scale.value = 1 / max_Ic
    E0 = Phi_0 * max_Ic / (2 * np.pi) / eV
    model.energy_scale.value = 1 / E0


def set_model_strict(model: pyo.ConcreteModel) -> None:
    max_Ic = max(pyo.value(model.Ic[edge]) for edge in model.edges)
    model.objective_flexible.deactivate()
    model.objective_strict.activate()
    model.current_conservation.activate()
    model.current_scale.value = 1e2 / max_Ic
    E0 = Phi_0 * max_Ic / (2 * np.pi) / eV
    model.energy_scale.value = 1 / E0
