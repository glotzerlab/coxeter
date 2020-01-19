# For calling up commonly used shapes
import numpy as np


# Truncation ranges from 0 (octahedra) to 2/3 (truncated octahedra) to 1
# (cuboctahedra)
def octahedron(side_length=1, truncation=0):
    pre = np.sqrt(2) / 2
    if truncation:

        trunc_value = truncation * np.sqrt(2) * side_length / 4

        return np.asarray(
            [[pre * side_length - trunc_value, trunc_value, 0],
             [pre * side_length - trunc_value, -trunc_value, 0],
             [pre * side_length - trunc_value, 0, trunc_value],
             [pre * side_length - trunc_value, 0, -trunc_value],
             [-pre * side_length + trunc_value, trunc_value, 0],
             [-pre * side_length + trunc_value, -trunc_value, 0],
             [-pre * side_length + trunc_value, 0, trunc_value],
             [-pre * side_length + trunc_value, 0, -trunc_value],
             [0, pre * side_length - trunc_value, trunc_value],
             [0, pre * side_length - trunc_value, -trunc_value],
             [trunc_value, pre * side_length - trunc_value, 0],
             [-trunc_value, pre * side_length - trunc_value, 0],
             [0, -pre * side_length + trunc_value, trunc_value],
             [0, -pre * side_length + trunc_value, -trunc_value],
             [trunc_value, -pre * side_length + trunc_value, 0],
             [-trunc_value, -pre * side_length + trunc_value, 0],
             [0, trunc_value, pre * side_length - trunc_value],
             [0, -trunc_value, pre * side_length - trunc_value],
             [trunc_value, 0, pre * side_length - trunc_value],
             [-trunc_value, 0, pre * side_length - trunc_value],
             [0, trunc_value, -pre * side_length + trunc_value],
             [0, -trunc_value, -pre * side_length + trunc_value],
             [trunc_value, 0, -pre * side_length + trunc_value],
             [-trunc_value, 0, -pre * side_length + trunc_value]])
    else:
        return np.asarray([[pre * side_length, 0, 0],
                           [-pre * side_length, 0, 0],
                           [0, pre * side_length, 0],
                           [0, -pre * side_length, 0],
                           [0, 0, pre * side_length],
                           [0, 0, -pre * side_length]])


def cube(side_length=1):
    return np.asarray([[side_length / 2, side_length / 2, side_length / 2],
                       [-side_length / 2, side_length / 2, side_length / 2],
                       [side_length / 2, -side_length / 2, side_length / 2],
                       [side_length / 2, side_length / 2, -side_length / 2],
                       [-side_length / 2, -side_length / 2, side_length / 2],
                       [-side_length / 2, side_length / 2, -side_length / 2],
                       [side_length / 2, -side_length / 2, -side_length / 2],
                       [-side_length / 2, -side_length / 2, -side_length / 2]])

# v1,v2,v3 should be flat arrays


def parallelepiped(v1, v2, v3):
    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)
    points = np.array([[0, 0, 0],
                       v1,
                       v2,
                       v3,
                       v1 + v2,
                       v1 + v3,
                       v2 + v3,
                       v1 + v2 + v3])
    com = points.mean(axis=0)
    return points - com


def rectangle(a, b, c):
    return np.asarray([[a / 2, b / 2, c / 2],
                       [-a / 2, b / 2, c / 2],
                       [a / 2, -b / 2, c / 2],
                       [a / 2, b / 2, -c / 2],
                       [-a / 2, -b / 2, c / 2],
                       [-a / 2, b / 2, -c / 2],
                       [a / 2, -b / 2, -c / 2],
                       [-a / 2, -b / 2, -c / 2]])

# This golden spiral code snippet is taken from
# http://www.softimageblog.com/archives/115


def poly_sphere(radius=1, points=100):
    pts = []
    inc = np.pi * (3 - np.sqrt(5))
    off = 2 / float(points)
    for k in range(0, int(points)):
        y = k * off - 1 + (off / 2)
        r = np.sqrt(1 - y * y)
        phi = k * inc
        pts.append([np.cos(phi) * r, y, np.sin(phi) * r])
    return radius * np.array(pts)


def rhombic_dodecahedron(a=1):
    return np.array([[a, a, a], [-a, a, a], [a, -a, a],
                     [a, a, -a], [a, -a, -a], [-a, a, -a],
                     [-a, -a, a], [-a, -a, -a], [2.0 * a, 0.0, 0.0],
                     [-2.0 * a, 0.0, 0.0], [0.0, 2.0 * a, 0.0],
                     [0.0, -2.0 * a, 0.0], [0.0, 0.0, 2.0 * a],
                     [0.0, 0.0, -2.0 * a]])

# A triangular prisim function
# Assumes isosceles triangle faces
# b is base width, h is height. h points in x direction


def triPrisim(b, h, l):
    return np.array([[-h / 3, b / 2, l / 2],
                     [-h / 3, -b / 2, l / 2],
                     [2 * h / 3, 0, l / 2],
                     [-h / 3, b / 2, -l / 2],
                     [-h / 3, -b / 2, -l / 2],
                     [2 * h / 3, 0, -l / 2]])

# makes a helix
# diameter: diameter of the spiral
# length: cylinder length of the spiral
# pitch: turns per unit length in the spiral
# num_points: how many points are in the returned representation
# returns
# points: points on the spiral
# segments: indices of points that are ajacent (connectivity information)


def helix(diameter, length, pitch, num_points):
    t = np.linspace(-0.5, 0.5, num=num_points, endpoint=True)
    x = (diameter / 2) * np.cos(length * pitch * 2 * np.pi * t)
    y = (diameter / 2) * np.sin(length * pitch * 2 * np.pi * t)
    z = t * length
    points = np.append(
        np.append(
            x.reshape((-1, 1)), y.reshape((-1, 1)), axis=1),
        z.reshape((-1, 1)), axis=1)
    segments = [[i, i + 1] for i in range(len(points) - 1)]
    return points, segments
