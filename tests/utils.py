import numpy as np
from scipy.spatial import Delaunay
from coxeter.shape_classes import ConvexPolyhedron


def compute_inertia_mc(vertices, num_samples=1e6):
    """Compute inertia tensor via Monte Carlo integration.

    Using Monte Carlo integration provides a means to test the results of an
    analytical calculation.

    Args:
        num_samples (int): The number of samples to use.

    Returns:
        float: The 3x3 inertia tensor.
    """
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)

    points = np.random.rand(int(num_samples), 3) * (maxs - mins) + mins

    hull = Delaunay(vertices)
    inside = hull.find_simplex(points) >= 0

    Ixx = np.mean(points[inside][:, 1]**2 + points[inside][:, 2]**2)
    Iyy = np.mean(points[inside][:, 0]**2 + points[inside][:, 2]**2)
    Izz = np.mean(points[inside][:, 0]**2 + points[inside][:, 1]**2)
    Ixy = np.mean(-points[inside][:, 0] * points[inside][:, 1])
    Ixz = np.mean(-points[inside][:, 0] * points[inside][:, 2])
    Iyz = np.mean(-points[inside][:, 1] * points[inside][:, 2])

    poly = ConvexPolyhedron(vertices)

    inertia_tensor = np.array([[Ixx, Ixy, Ixz],
                               [Ixy, Iyy, Iyz],
                               [Ixz, Iyz, Izz]]) * poly.volume

    return inertia_tensor
