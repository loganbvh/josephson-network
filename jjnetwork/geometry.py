from meshpy import triangle
import numpy as np
from scipy import spatial
import scipy.linalg as la


def unit_vector(vector):
    """Normalizes ``vector``."""
    return vector / la.norm(vector, axis=-1)[:, np.newaxis]


def circle(radius: float, points: int = 100, center=(0, 0)) -> np.ndarray:
    """Returns the coordinates for a circle with a given radius, centered at the
    specified center.

    Args:
        radius: Radius of the circle
        points: Number of points in the circle
        center: Coordinates of the center of the circle

    Returns:
        A shape ``(points, 2)`` array of (x, y) coordinates
    """
    x0, y0 = center
    theta = np.linspace(0, 2 * np.pi, points, endpoint=False)
    xs = radius * np.cos(theta)
    ys = radius * np.sin(theta)
    coords = np.stack([xs, ys], axis=1) + np.array([[x0, y0]])
    return coords


def close_curve(points: np.ndarray) -> np.ndarray:
    """Close a curve (making the start point equal to the end point),
    if it is not already closed.

    Args:
        points: Shape ``(m, n)`` array of ``m`` coordinates in ``n`` dimensions.

    Returns:
        ``points`` with the first point appended to the end if the start point
        was not already equal to the end point.
    """
    if not np.allclose(points[0], points[-1]):
        points = np.concatenate([points, points[:1]], axis=0)
    return points


def triangulate(coords, min_triangles=None, convex_hull=True, **kwargs):
    """Generate a triangular (Delaunay) mesh for a geometry specified by ``coords``.

    Keyword arguments are passed to ``meshpy.triangle.build()``.

    Args:
        coords: Shape (c, 2) array of (x, y) coordinates
        min_triangles: Minimum number of triangles in the resulting mesh.
        convex_hull: If True, a mesh is generated for the convex hull of ``coords``.

    Returns:
        mesh vertices [shape (n, 2)] and triangle indices [shape (m, 3)]
    """
    # Coords is a shape (n, 2) array of vertex coordinates.
    coords = np.asarray(coords)
    # Remove duplicate coordinates, otherwise triangle.build() will segfault.
    # By default, np.unique() does not preserve order, so we have to remove
    # duplicates this way:
    _, ix = np.unique(coords, return_index=True, axis=0)
    coords = coords[np.sort(ix)]
    # Facets is a shape (m, 2) array of edge indices.
    # coords[facets] is a shape (m, 2, 2) array of edge coordinates:
    # [(x0, y0), (x1, y1)]
    if convex_hull:
        facets = spatial.ConvexHull(coords).simplices
    else:
        indices = np.arange(coords.shape[0], dtype=int)
        facets = np.stack([indices, np.roll(indices, -1)], axis=1)

    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(coords)
    mesh_info.set_facets(facets)
    kwargs = kwargs.copy()

    if min_triangles is None:
        mesh = triangle.build(mesh_info=mesh_info, **kwargs)
        points = np.array(mesh.points)
        triangles = np.array(mesh.elements)
    else:
        max_vol = 1
        num_triangles = 0
        while num_triangles < min_triangles:
            kwargs["max_volume"] = max_vol
            mesh = triangle.build(
                mesh_info=mesh_info,
                **kwargs,
            )
            points = np.array(mesh.points)
            triangles = np.array(mesh.elements)
            num_triangles = triangles.shape[0]
            max_vol *= 0.9
    return points, triangles


def polygon_centroids(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Returns x, y coordinates for polygon centroids (centers of mass).

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices.
        polygons: Shape (m, p) array of polygon indices.

    Returns:
        Shape (m, 2) array of triangle centroid (center of mass) coordinates
    """
    return points[triangles].mean(axis=1)


def triangle_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Calculates the area of each triangle.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices
        triangles: Shape (m, 3) array of triangle indices

    Returns:
        Shape (m, ) array of triangle areas
    """
    xy = points[triangles]
    # s1 = xy[:, 2, :] - xy[:, 1, :]
    # s2 = xy[:, 0, :] - xy[:, 2, :]
    # s3 = xy[:, 1, :] - xy[:, 0, :]
    # which can be simplified to
    # s = xy[:, [2, 0, 1]] - xy[:, [1, 2, 0]]  # 3D
    s = xy[:, [2, 0]] - xy[:, [1, 2]]  # 2D
    a = np.linalg.det(s)
    return a * 0.5


def polygon_areas(points: np.ndarray, polygons: np.ndarray) -> np.ndarray:
    """Calculates the (signed) area of each polygon.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices
        polygons: Shape (m, p) array of polygon indices

    Returns:
        Shape (m, ) array of polygon areas
    """
    xy = points[polygons]
    x = xy[:, :, 0]
    y = xy[:, :, 1]
    return 0.5 * (
        np.einsum("ij,ij -> i", y, np.roll(x, 1, axis=-1))
        - np.einsum("ij,ij -> i", x, np.roll(y, 1, axis=-1))
    )


def rotation_matrix(angle_radians: float) -> np.ndarray:
    """Returns a 2D rotation matrix."""
    c = np.cos(angle_radians)
    s = np.sin(angle_radians)
    return np.array([[c, -s], [s, c]])


def rotate(coords: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotates an array of (x, y) coordinates counterclockwise by
    the specified angle.

    Args:
        coords: Shape (n, 2) array of (x, y) coordinates.
        angle_degrees: The angle by which to rotate the coordinates.

    Returns:
        Shape (n, 2) array of rotated coordinates (x', y')
    """
    coords = np.asarray(coords)
    assert coords.ndim == 2
    assert coords.shape[1] == 2
    R = rotation_matrix(np.radians(angle_degrees))
    return (R @ coords.T).T


def is_ccw(points):
    """Check if connected 2D points are counterclockwise."""

    points = np.asarray(points)

    if len(points.shape) != 2 or points.shape[1] != 2:
        raise ValueError("CCW is only defined for 2D")
    xd = np.diff(points[:, 0])
    # sum along axis=1 with a dot product
    yd = np.dot(
        np.column_stack((points[:, 1], points[:, 1]))
        .reshape(-1)[1:-1]
        .reshape((-1, 2)),
        [1, 1],
    )
    area = np.sum(xd * yd) * 0.5
    ccw = area < 0
    return ccw
