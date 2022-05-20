from typing import Sequence

import networkx as nx
import numpy as np
import pint
from scipy import special
import scipy.linalg as la

ureg = pint.UnitRegistry()
mu_0 = ureg("mu_0").to_base_units().magnitude
Phi_0 = ureg("Phi_0").to_base_units().magnitude


def biot_savart(
    eval_positions: np.ndarray,
    *,
    current_positions: np.ndarray,
    current_vectors: np.ndarray,
    currents: np.ndarray,
) -> np.ndarray:
    """Calculates the vector magnetic field [Bx, By, Bx] at ``eval_positions``
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
    assert eval_positions.shape[-1] == 3
    assert current_positions.shape[-1] == 3
    assert current_vectors.shape[-1] == 3
    assert currents.shape[-1] == 1

    dx = np.subtract.outer(eval_positions[:, 0], current_positions[:, 0])
    dy = np.subtract.outer(eval_positions[:, 1], current_positions[:, 1])
    dz = np.subtract.outer(eval_positions[:, 2], current_positions[:, 2])
    rprime = np.stack([dx, dy, dz], axis=-1)
    denom = (la.norm(rprime, axis=-1) ** 3)[:, :, np.newaxis]
    integrand = np.cross(current_vectors, rprime) / denom
    integral = np.einsum("jk,ijk -> ik", currents, integrand)
    return mu_0 / (4 * np.pi) * integral * ureg("tesla")


def current_loop_vector_potential(
    positions: np.ndarray,
    *,
    loop_center: Sequence[float] = (0, 0, 0),
    loop_radius: float = 1e-6,
    current: float = 1e-3,
):
    """Calculates the magnetic vector potential [Ax, Ay] at ``positions``
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
    # Extract edge data from the grap
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
    edge_currents = np.array(edge_currents)
    # Add a zero z coordinate
    edge_centers = np.append(edge_centers, np.zeros_like(edge_centers[:, :1]), axis=1)
    edge_vectors = np.append(edge_vectors, np.zeros_like(edge_vectors[:, :1]), axis=1)
    return biot_savart(
        positions,
        current_positions=edge_centers,
        current_vectors=edge_vectors,
        currents=edge_currents,
    )
