from typing import Sequence, Union

import networkx as nx
import numpy as np
import pint
import scipy.linalg as la

# from scipy import interpolate
# from scipy.spatial import distance
from scipy import special

# from .geometry import unit_vector
# from .graph_utils import get_node_positions, round_trip


ureg = pint.UnitRegistry()
mu_0 = ureg("mu_0").to_base_units().magnitude
Phi_0 = ureg("Phi_0").to_base_units().magnitude
eV = ureg("eV").to_base_units().magnitude


def biot_savart(
    eval_positions: np.ndarray,
    *,
    current_positions: np.ndarray,
    current_vectors: np.ndarray,
    currents: np.ndarray,
) -> np.ndarray:
    """Calculates the vector magnetic field [Bx, By, Bz] at ``eval_positions``
    due to a discrete set of 1D current elements.

    Input units are meters and Amperes, output units are Tesla.

    Args:
        eval_positions: Shape (n, 3) array of (x, y, z) positions at which to
            evaluate the field.
        current_positions: Shape (m, 3) array of (x, y, z) positions for the
            current elements.
        current_vectors: Shape (m, 3) array of (dx, dy, dy) distance vectors
            indicating the direction and length of the current elements.
        currents: Shape (m, ) or (m, 1) array of current magnitudes for each
            current element.

    Returns:
        Shape (n, 3) pint.Quantity array of the vector magnetic field
        at ``eval_positions``.
    """
    eval_positions = np.atleast_2d(eval_positions)
    current_positions = np.atleast_2d(current_positions)
    current_vectors = np.atleast_2d(current_vectors)
    currents = np.atleast_2d(currents)
    assert eval_positions.shape[-1] == 3, eval_positions.shape
    assert current_positions.shape[-1] == 3, current_positions.shape
    assert current_vectors.shape[-1] == 3, current_vectors.shape
    assert currents.shape[-1] == 1, currents.shape

    dx = np.subtract.outer(eval_positions[:, 0], current_positions[:, 0])
    dy = np.subtract.outer(eval_positions[:, 1], current_positions[:, 1])
    dz = np.subtract.outer(eval_positions[:, 2], current_positions[:, 2])
    rprime = np.stack([dx, dy, dz], axis=-1)
    denom = (la.norm(rprime, axis=-1) ** 3)[:, :, np.newaxis]
    integrand = np.cross(current_vectors, rprime) / denom
    integral = np.einsum("jk,ijk -> ik", currents, integrand)
    return mu_0 / (4 * np.pi) * integral * ureg("tesla")


def uniform_Bz_vector_potential(
    positions: np.ndarray,
    Bz: Union[float, str, pint.Quantity],
) -> np.ndarray:
    """Calculates the magnetic vector potential [Ax, Ay, Az] at ``positions``
    due uniform magnetic field along the z-axis with strength ``Bz``.

    Args:
        positions: Shape (n, 3) array of (x, y, z) positions in meters at which to
            evaluate the vector potential.
        Bz: The strength of the uniform field, as a pint-parseable string,
            a pint.Quantity, or a float with units of Tesla.

    Returns:
        Shape (n, 3) array of the vector potential [Ax, Ay, Az] at ``positions``
        in units of Tesla * meter.
    """
    assert isinstance(Bz, (float, str, pint.Quantity)), type(Bz)
    positions = np.atleast_2d(positions)
    assert positions.shape[1] == 3, positions.shape
    if not isinstance(positions, pint.Quantity):
        positions = positions * ureg("meter")
    if isinstance(Bz, str):
        Bz = ureg(Bz)
    if isinstance(Bz, float):
        Bz = Bz * ureg("tesla")
    Ax = -Bz * positions[:, 1] / 2
    Ay = Bz * positions[:, 0] / 2
    A = np.stack([Ax, Ay, np.zeros_like(Ax)], axis=1)
    return A.to("tesla * meter")


def current_loop_vector_potential(
    positions: np.ndarray,
    *,
    loop_center: Sequence[float] = (0, 0, 0),
    loop_radius: float = 1e-6,
    current: float = 1e-3,
):
    """Calculates the magnetic vector potential [Ax, Ay, Az] at ``positions``
    due to a 1D current loop.

    Input units are meters and Amperes, output units are Tesla * meter.

    Args:
        positions: Shape (n, 3) array of (x, y, z) positions at which to
            evaluate the vector potential.
        loop_center: (x, y, z) coordinates of the current loop center.
        loop_radius: radius of the current loop.
        current: Magnitude of the current flowing in the loop.

    Returns:
        Shape (n, 3) array of the vector potential [Ax, Ay, Az] at ``positions``.
    """
    # http://www.physics.usu.edu/Wheeler/EMarchive/Jch5Notes.pdf
    positions = np.atleast_2d(positions)
    loop_center = np.atleast_2d(loop_center)
    a = loop_radius

    positions = positions - loop_center
    # This is a pint-friendly vector norm.
    rs = np.sqrt(np.sum(np.square(positions), axis=1))
    # rs = la.norm(positions, axis=1)
    thetas = np.arccos(positions[:, 2] / rs)
    sin_thetas = np.sin(thetas)
    # m == k**2, see docs for scipy.special.ellipk
    denom = rs**2 + a**2 + 2 * a * rs * sin_thetas
    m = 4 * a * rs * sin_thetas / denom
    K = special.ellipk(m)
    E = special.ellipe(m)
    mag = -mu_0 * current * a / (np.pi * m) * (((m - 2) * K + 2 * E)) / np.sqrt(denom)
    # \vec{A} is directed along the azimuthal direction,
    # so here we generate the azimuthal unit vector.
    # Azimuthal angle + pi / 2 to get azimuthal direction.
    phis = np.arctan2(positions[:, 1], positions[:, 0]) + np.pi / 2
    direc = np.stack([np.cos(phis), np.sin(phis), np.zeros_like(phis)], axis=1)
    return mag[:, np.newaxis] * direc


def current_loop_field(
    positions: np.ndarray,
    *,
    loop_center: Sequence[float] = (0, 0, 0),
    loop_radius: float = 1e-6,
    current: float = 1e-3,
    num_segments: int = 101,
):
    """Calculates the vector magnetic field [Bx, By, Bz] at ``positions``
    due to a 1D current loop.

    Input units are meters and Amperes, and output units are Tesla.

    Args:
        positions: Shape (n, 3) array of (x, y, z) positions at which to
            evaluate the vector potential.
        loop_center: (x, y, z) coordinates of the current loop center.
        loop_radius: radius of the current loop.
        current: Magnitude of the current flowing in the loop.
        num_segments: Number of current elements used to model the loop.

    Returns:
        Shape (n, 3) array of the magnetic field [Bx, By, Bz] at ``positions``.
    """
    positions = np.atleast_2d(positions)
    loop_center = np.atleast_2d(loop_center)
    # Create loop positions
    thetas = np.linspace(0, 2 * np.pi, num_segments)
    circ = np.stack([np.cos(thetas), np.sin(thetas), np.zeros_like(thetas)], axis=1)
    loop = loop_radius * circ + loop_center
    dloop = np.diff(loop, axis=0)
    loop = loop[:-1]
    return (
        biot_savart(
            positions,
            current_positions=loop,
            current_vectors=dloop,
            currents=current,
        )
        .to("tesla")
        .magnitude
    )


def calculate_field_from_graph(
    positions: np.ndarray,
    graph: nx.DiGraph,
    length_units: str = "um",
) -> np.ndarray:
    positions = np.atleast_2d(positions)
    if not isinstance(positions, pint.Quantity):
        positions = positions * ureg(length_units)
    positions = positions.to("m").magnitude
    # Extract edge data from the graph
    nodes = graph.nodes
    edge_centers = []
    edge_vectors = []
    edge_currents = []
    for i, j, data in graph.edges.data():
        n1 = nodes[i]
        n2 = nodes[j]
        r1 = n1["position"]
        r2 = n2["position"]
        edge_centers.append((r1 + r2) / 2)
        edge_vectors.append(r2 - r1)
        edge_currents.append(data["current"])
    edge_centers = np.stack(edge_centers, axis=0)
    edge_vectors = np.stack(edge_vectors, axis=0)
    edge_currents = np.array(edge_currents)[:, np.newaxis]
    # Add a zero z coordinate
    edge_centers = np.append(edge_centers, np.zeros_like(edge_centers[:, :1]), axis=1)
    edge_vectors = np.append(edge_vectors, np.zeros_like(edge_vectors[:, :1]), axis=1)
    return biot_savart(
        positions,
        current_positions=edge_centers,
        current_vectors=edge_vectors,
        currents=edge_currents,
    )


# def calculate_vector_potential_from_graph(
#     positions: np.ndarray,
#     graph: nx.DiGraph,
#     length_units: str = "um",
# ) -> np.ndarray:
#     positions = np.atleast_2d(positions)
#     if not isinstance(positions, pint.Quantity):
#         positions = positions * ureg(length_units)
#     positions = positions.to("m").magnitude
#     # Extract edge data from the graph
#     nodes = graph.nodes
#     edge_centers = []
#     edge_vectors = []
#     edge_currents = []
#     for i, j, data in graph.edges.data():
#         n1 = nodes[i]
#         n2 = nodes[j]
#         r1 = n1["position"]
#         r2 = n2["position"]
#         edge_centers.append((r1 + r2) / 2)
#         edge_vectors.append(r2 - r1)
#         edge_currents.append(data["current"])
#     edge_centers = np.stack(edge_centers, axis=0)
#     edge_vectors = np.stack(edge_vectors, axis=0)
#     edge_currents = np.array(edge_currents)[:, np.newaxis]
#     # Add a zero z coordinate
#     edge_centers = np.append(edge_centers, np.zeros_like(edge_centers[:, :1]), axis=1)
#     edge_vectors = np.append(edge_vectors, np.zeros_like(edge_vectors[:, :1]), axis=1)
#     unit_vectors = unit_vector(edge_vectors)
#     rho = distance.cdist(positions, edge_centers)
#     rho[rho == 0] = np.nan
#     integrand = (edge_currents * unit_vectors) / rho[:, :, np.newaxis]
#     return mu_0 / (4 * np.pi) * np.nansum(integrand, axis=1) * ureg("T * m")


# def mutual_vector_potential_matrix(graph: nx.DiGraph, integral_n: int = 200):
#     nodes = graph.nodes
#     edge_centers = []
#     edge_vectors = []
#     positions = get_node_positions(graph)
#     edge_indices = np.array(graph.edges)
#     for i, j in graph.edges:
#         r1 = nodes[i]["position"]
#         r2 = nodes[j]["position"]
#         edge_centers.append((r1 + r2) / 2)
#         edge_vectors.append(r2 - r1)
#     edge_centers = np.stack(edge_centers, axis=0)
#     edge_vectors = np.stack(edge_vectors, axis=0)
#     # Add a zero z coordinate
#     edge_centers = np.append(edge_centers, np.zeros_like(edge_centers[:, :1]), axis=1)
#     edge_vectors = np.append(edge_vectors, np.zeros_like(edge_vectors[:, :1]), axis=1)
#     positions = np.append(positions, np.zeros_like(positions[:, :1]), axis=1)
#     positions_i = positions[edge_indices[:, 0]]
#     positions_j = positions[edge_indices[:, 1]]
#     # Integrate A along the path from i to j
#     index = np.linspace(0, 1, integral_n)
#     xs_ij = interpolate.interp1d(
#         [0, 1],
#         np.stack((positions_i[:, 0], positions_j[:, 0]), axis=0),
#         axis=0,
#     )(index)
#     ys_ij = interpolate.interp1d(
#         [0, 1],
#         np.stack((positions_i[:, 1], positions_j[:, 1]), axis=0),
#         axis=0,
#     )(index)
#     positions_ij = np.stack([xs_ij, ys_ij, np.zeros_like(xs_ij)], axis=-1)
#     drs = np.diff(positions_ij, axis=0)
#     MAij = np.zeros((edge_centers.shape[0], edge_centers.shape[0]))
#     for rs, dr in zip(positions_ij, drs):
#         rho = distance.cdist(edge_centers, rs)
#         MAij += np.einsum("ijk, ik -> ij", edge_vectors / rho[:, :, np.newaxis], dr)
#     # TODO: check whether this needs to be transposed...
#     return (ureg("mu_0") * ureg("1 meter") / (4 * np.pi) * MAij).to("Phi_0 / A")


# def mutual_inductance_matrix(graph: nx.DiGraph, loops: list[list[int]]):
#     num_loops = len(loops)
#     edges = np.array(graph.edges)
#     num_edges = edges.shape[0]
#     MAij = mutual_vector_potential_matrix(graph)
#     MAij = MAij.to("Phi_0 / A").magnitude

#     def edge_sign(edge):
#         # 1 if edge in edges else -1
#         return -1 + 2 * np.equal(edge, edges).all(axis=1).any()

#     Mij = np.zeros((num_loops, num_loops))
#     edge_indices = {tuple(edge): i for i, edge in enumerate(graph.edges)}
#     edge_indices.update({tuple(edge)[::-1]: i for i, edge in enumerate(graph.edges)})
#     round_trips = [round_trip(loop) for loop in loops]
#     loop_edge_indices = [[edge_indices[edge] for edge in loop] for loop in round_trips]
#     edge_signs = [np.apply_along_axis(edge_sign, 1, loop) for loop in round_trips]
#     for ell in range(num_loops):
#         currents = np.zeros((num_edges, 1))
#         ell_indices = loop_edge_indices[ell]
#         currents[ell_indices, 0] = edge_signs[ell]
#         Aij = MAij @ currents
#         for k in range(num_loops):
#             k_signs = edge_signs[k][:, np.newaxis]
#             k_indices = loop_edge_indices[k]
#             Mij[k, ell] = np.sum(k_signs * Aij[k_indices])
#     return Mij
