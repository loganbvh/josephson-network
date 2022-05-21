from numbers import Number
from typing import Any, Sequence, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import spatial
import scipy.linalg as la

from .geometry import (
    close_curve,
    is_ccw,
    polygon_areas,
    unit_vector,
)


def get_scalar(value):
    """Converts a GEKKO variable into a Python scalar."""
    if isinstance(value, Number):
        return value
    try:
        return value[0]
    except TypeError:
        return value


def nearest_neighbors(points: np.ndarray, k_nn: int) -> tuple[np.ndarray, np.ndarray]:
    """Find the ``k_nn`` nearest neighbors of each point in ``points``.

    Here we do not consider a point to be its own neighbor.

    Args:
        points: Shape (n, 2) array of (x, y) positions.
        k_nn: The number of nearest neighbors to find.

    Returns:
        Shape (n, k_nn) array of neighbor indices, shape (n, k_nn) array of distances.
    """
    tree = spatial.KDTree(points)
    # k_nn includes self as a neighbor, so we want k_nn+1 in total.
    distances, neighbors = tree.query(points, k=k_nn + 1)
    # Don't consider self a neighbor.
    distances = distances[:, 1:]
    neighbors = neighbors[:, 1:]
    assert np.all(distances > 0)
    return neighbors, distances


def nearest_neighbors_ball(
    points: np.ndarray, radius: float
) -> tuple[list[list[int]], np.ndarray]:
    """Finds all neighbors within a given radius.

    Here we do not consider a point to be its own neighbor.

    Args:
        points: Shape (n, 2) array of (x, y) positions.
        radius: The cutoff radius to be considered a neighbor.

    Returns:
        List of lists of neighbor indices, list of list of neighbor distances.
    """
    tree = spatial.KDTree(points)
    neighbors = tree.query_ball_tree(tree, r=radius)
    # Don't consider self a neighbor.
    neighbors = [[j for j in n if j != i] for i, n in enumerate(neighbors)]
    distances = []
    for point, neigh in zip(points, neighbors):
        distances.append(la.norm(point - points[neigh], axis=1))
    for d in distances:
        if len(d):
            assert np.all(d > 0), d
    return neighbors, distances


def round_trip(seq: Sequence[Any]) -> list[Any]:
    """Given a sequence, returns a list containing successive pairs of
    elements, with the final pair being (seq[-1], seq[0]).

    round_trip([0, 1, 2]) --> [(0, 1), (1, 2), (2, 0)]
    """
    seq = list(seq)
    if seq[0] == seq[-1]:
        seq = seq[:-1]
    return list(zip(seq, seq[1:])) + [(seq[-1], seq[0])]


def basis_loops(graph: nx.Graph, ensure_ccw: bool = True) -> list[list[int]]:
    """Return all basis cycles for a graph, ignoring edge directions.

    Args:
        graph: The graph to analyze.

    Returns:
        List of lists of node indices
    """
    if isinstance(graph, nx.DiGraph):
        graph = graph.to_undirected(as_view=True)
    loops = nx.cycle_basis(graph, root=0)
    assert all(len(loop) > 1 for loop in loops)
    loops = [close_curve(loop) for loop in loops]
    if ensure_ccw:
        points = np.stack([p for _, p in sorted(graph.nodes.data("position"))], axis=0)
        ccw_loops = []
        for loop in loops:
            ccw = is_ccw(points[loop])
            if not ccw:
                loop = loop[::-1]
                assert is_ccw(points[loop])
            ccw_loops.append(loop)
        loops = ccw_loops
    return loops


def find_paths(graph: nx.Graph, node: int, length: int) -> list[list[int]]:
    """Returns all paths of a given length starting at a given node."""
    # https://stackoverflow.com/questions/57661314/
    # fastest-way-to-find-all-cycles-of-length-n-containing-a-specified-node-in-a-dire
    if length == 0:
        return [[node]]
    return [
        [node] + path
        for neighbor in graph.neighbors(node)
        for path in find_paths(graph, neighbor, length - 1)
        if node not in path[1:-1]
    ]


def find_cycles(graph: nx.Graph, node: int, length: int) -> list[tuple[int]]:
    """Returns all closed paths (cycles) of a given length starting at a given node,
    ignoring edge directions.
    """
    # https://stackoverflow.com/questions/57661314/
    # fastest-way-to-find-all-cycles-of-length-n-containing-a-specified-node-in-a-dire
    graph = graph.to_undirected(as_view=True)
    paths = find_paths(graph, node, length)
    return [
        tuple(path)
        for path in paths
        if (path[-1] == node) and sum(x == node for x in path) == 2
    ]


def find_all_cycles(graph: nx.Graph, length: int) -> list[list[tuple[int]]]:
    """Finds all closed paths of a given length in a graph, ignoring edge directions."""
    graph = graph.to_undirected(as_view=True)
    cycles = []
    for node in graph.nodes:
        cycles.extend(find_cycles(graph, node, length))
    return cycles


def find_all_cells(graph: nx.Graph, length: int = 4) -> np.ndarray:
    """Finds all cycles of a given length in the graph and orients them counterclockwise."""
    loops = find_all_cycles(graph, length)
    seen = set()
    unique_loops = []
    for loop in loops:
        if frozenset(loop) not in seen:
            unique_loops.append(loop)
            seen.add(frozenset(loop))
    # find_all_cycles returns round-trip indices
    cells = np.array(unique_loops)
    points = np.stack([p for _, p in sorted(graph.nodes.data("position"))], axis=0)
    # Ensure triangles are CCW
    areas = polygon_areas(points, cells[:, :-1])
    clockwise = areas < 0
    cells[clockwise, :] = cells[clockwise, ::-1]
    areas = polygon_areas(points, cells[:, :-1])
    assert np.all(areas > 0)
    return cells


def remove_overlapping_islands(
    island_positions: np.ndarray,
    island_diameter: float,
    min_distance: float = 0.001,
) -> np.ndarray:
    """Given an array of (x, y) island center cooridnates, removes all islands
    whose centers are within ``island_diameter + min_distances`` of each other, replacing
    the group of overlapping islands with a single island located at the group's center.
    """
    iteration = 0
    while True:
        overlapping, _ = nearest_neighbors_ball(
            island_positions, island_diameter + min_distance
        )
        overlapping = [
            [i] + neighbors for i, neighbors in enumerate(overlapping) if neighbors
        ]
        if len(overlapping) == 0:
            break
        islands_to_remove = np.unique(np.concatenate(overlapping))
        if iteration == 0:
            print(f"Found {islands_to_remove.shape[0]} overlapping islands.")
            print("Removing overlapping islands...")
        remove = np.zeros(island_positions.shape[0], dtype=bool)
        remove[islands_to_remove] = True
        new_positions = island_positions[~remove]

        new_centers = [
            island_positions[islands].mean(axis=0) for islands in overlapping
        ]
        new_centers = np.unique(np.stack(new_centers, axis=0), axis=0)
        new_positions = np.concatenate([new_positions, new_centers], axis=0)
        print(
            f"Iteration {iteration}: Removed {islands_to_remove.shape[0]}, "
            f"added {new_centers.shape[0]}. "
        )
        island_positions = new_positions
        iteration += 1
    return island_positions


def remove_isolated_islands(
    island_positions: np.ndarray,
    cutoff_radius: float,
) -> tuple[np.ndarray, list[list[int]]]:
    neighbors, _ = nearest_neighbors_ball(island_positions, cutoff_radius)
    # Remove isolated islands (or isolated pairs of islands).
    isolated = np.array([len(n) < 2 for n in neighbors])
    num_isolated = np.sum(isolated)
    while num_isolated:
        print(
            f"Removing {num_isolated} isolated island(s): " f"{np.where(isolated)[0]}."
        )
        island_positions = island_positions[~isolated]
        neighbors, _ = nearest_neighbors_ball(
            island_positions,
            cutoff_radius,
        )
        isolated = np.array([len(n) < 2 for n in neighbors])
        num_isolated = np.sum(isolated)
    assert not (np.sum(isolated))
    return island_positions, neighbors


def extract_edge_data(graph: nx.Graph) -> dict[tuple[int], float]:
    edge_data = {}
    nodes = graph.nodes
    for i, j, data in graph.edges.data():
        n1 = nodes[i]
        n2 = nodes[j]
        r1 = n1["position"]
        r2 = n2["position"]
        center = (r1 + r2) / 2
        vector = r2 - r1
        edge_data[(i, j)] = {
            "node1_x": r1[0],
            "node1_y": r1[1],
            "node2_x": r2[0],
            "node2_y": r2[1],
            "node1_phase": get_scalar(n1["phase"]),
            "node2_phase": get_scalar(n2["phase"]),
            "center_x": center[0],
            "center_y": center[1],
            "vector_x": vector[0],
            "vector_y": vector[1],
        }
        edge_data[(i, j)].update({k: get_scalar(v) for k, v in data.items()})
    return edge_data


def edge_data_to_df(graph: nx.Graph) -> pd.DataFrame:
    edge_data = extract_edge_data(graph)
    df = pd.DataFrame(edge_data).T
    df.index.names = ["node1", "node2"]
    return df.sort_index()


def load_h5(filename: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    df_edge = pd.read_hdf(filename, "edge_data")
    df_loop = pd.read_hdf(filename, "loops")
    vortices = pd.read_hdf(filename, "nonzero_loops").values
    return df_edge, df_loop, vortices


def make_graph_from_df(df: pd.DataFrame) -> nx.DiGraph:
    df = df.sort_index()
    graph = nx.DiGraph()
    graph.add_edges_from(df.index)
    for i, j in graph.edges:
        row = dict(df.loc[i, j])
        node_attrs = {}
        for node, label in zip([i, j], [1, 2]):
            node_attrs[node] = dict(
                position=np.array([row[f"node{label}_x"], row[f"node{label}_y"]]),
                phase=np.array([row[f"node{label}_phase"], row[f"node{label}_phase"]]),
            )
        nx.set_node_attributes(graph, node_attrs)
        edge_attrs = ["length", "EJ", "A", "delta", "theta", "current", "energy"]
        nx.set_edge_attributes(graph, {(i, j): {key: row[key] for key in edge_attrs}})
    return graph


def load_graph_h5(filename: str) -> pd.DataFrame:
    df, _, _ = load_h5(filename)
    return make_graph_from_df(df)


def draw_graph(
    graph: nx.Graph, ax: Optional[plt.Axes] = None, buffer: float = 1.1, **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    ax.set_aspect("equal")
    pos = nx.get_node_attributes(graph, "position")
    coords = np.stack([p for _, p in sorted(graph.nodes.data("position"))], axis=0)
    kwargs["arrows"] = False
    kwargs["node_size"] = kwargs.get("node_size", 1)
    nx.draw(graph, pos, ax=ax, **kwargs)
    x0, y0 = np.mean(coords, axis=0)
    dx, dy = buffer * np.ptp(coords, axis=0)
    ax.set_xlim(x0 - dx / 2, x0 + dx / 2)
    ax.set_ylim(y0 - dy / 2, y0 + dy / 2)
    return fig, ax


def draw_currents(
    df: pd.DataFrame, figsize: tuple[float, float] = (12, 4), **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    node1_positions = np.stack([df["node1_x"], df["node1_y"]], axis=1) * 1e6
    node2_positions = np.stack([df["node2_x"], df["node2_y"]], axis=1) * 1e6
    edge_positions = (node1_positions + node2_positions) / 2
    edge_vectors = np.stack([df["vector_x"], df["vector_y"]], axis=1) * 1e6
    unit_vectors = unit_vector(edge_vectors)
    currents = df["current"].values * 1e9
    unit_vectors *= np.sign(currents)[:, np.newaxis]
    currents = np.abs(currents)
    Ic = df["Ic"].values * 1e9
    fig, (ax, bx) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    for a in (ax, bx):
        a.set_aspect("equal")
        a.set_xlabel("$x$ [$\\mu$m]")
        a.set_ylabel("$y$ [$\\mu$m]")

    im = ax.quiver(
        edge_positions[:, 0],
        edge_positions[:, 1],
        np.abs(currents) * unit_vectors[:, 0],
        np.abs(currents) * unit_vectors[:, 1],
        currents,
        **kwargs,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Current, $|I|$ [nA]")

    im = bx.quiver(
        edge_positions[:, 0],
        edge_positions[:, 1],
        np.abs(currents) * unit_vectors[:, 0],
        np.abs(currents) * unit_vectors[:, 1],
        currents / Ic,
        **kwargs,
    )
    cbar = fig.colorbar(im, ax=bx)
    cbar.set_label("$|I| / I_c$")
    return fig, (ax, bx)
