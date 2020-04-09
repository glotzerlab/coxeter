from itertools import chain

import numpy as np


class DegenerateTriangle(Exception):
    pass


def looped_pairs(iterable):
    """
    >>> list(looped_pairs([1,2,3]))
    [(1, 2), (2, 3), (3, 1)]
    """
    iterable = iter(iterable)
    first = last = next(iterable)
    for x in iterable:
        yield last, x
        last = x
    yield (last, first)


def triplets(iterable):
    """
    >>> list(triplets([1,2,3,4,5]))
    [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    """
    iterable = iter(iterable)
    backlog = (next(iterable), next(iterable))
    for item in iterable:
        yield (backlog[0], backlog[1], item)
        backlog = (backlog[1], item)


def near_zero(v):
    """
    >>> near_zero(0)
    True
    >>> near_zero(.1)
    False
    >>> near_zero(1E-10)
    True
    >>> near_zero(np.array([0, 0, 0]))
    True
    >>> near_zero(np.array([1E-10, 1E-10, 1E-10]))
    True
    >>> near_zero(np.array([1E-10, 1E-10, 7]))
    False
    """
    if isinstance(v, (float, int)):
        return v > -1E-6 and v < 1E-6
    else:
        return np.allclose(v, np.zeros(np.shape(v)))


def calculate_normal_3d(polygon):
    """Returns polygon normal vector for 3d polygon"""
    normal = np.array([0.0] * len(polygon[0]))
    for p1, p2 in looped_pairs(polygon):
        minus = p2 - p1
        plus = p2 + p1
        normal[0] += minus[1] * plus[2]
        normal[1] += minus[2] * plus[0]
        normal[2] += minus[0] * plus[1]
    if near_zero(normal):
        raise ValueError("No normal found")
    else:
        return normal


def calculate_normal_2d(polygon):
    """Returns 'normal' of 2d polygon (-1 if clockwise, 1 if not)"""
    sum = 0
    for (x1, y1), (x2, y2) in looped_pairs(polygon):
        sum += (x2 - x1) * (y2 + y1)
    if sum > 1E-6:
        return 1
    elif sum < -1E-6:
        return -1
    else:
        raise ValueError("No normal found")


def calculate_normal(polygon):
    """Returns polygon normal vector (or scalar if 2d)"""
    l = len(polygon[0])
    if l == 2:
        return calculate_normal_2d(polygon)
    elif l == 3:
        return calculate_normal_3d(polygon)
    else:
        raise TypeError("Unsupported number of dimensions: " + str(l))


def looped_slice(seq, start, count):
    """
    >>> list(looped_slice([1,2,3],0,3))
    [1, 2, 3]
    >>> list(looped_slice([1,2,3],2,3))
    [3, 1, 2]
    """
    l = len(seq)
    for i in range(start, start + count):
        yield seq[i % l]


def looped_slice_inv(seq, start, count):
    """
    >>> list(looped_slice_inv([1,2,3,4],0,3))
    [4]
    >>> list(looped_slice_inv([1,2,3,4],1,3))
    [1]
    >>> list(looped_slice_inv([1,2,3,4],2,3))
    [2]
    >>> list(looped_slice_inv([1,2,3,4],3,3))
    [3]
    """
    if start + count > len(seq):
        return seq[start + count - len(seq): start]
    else:
        return chain(seq[:start], seq[start + count:])


def any_point_in_triangle(triangle, points):
    a, b, c = triangle
    s = b - a
    t = c - a

    stack = [s, t]
    if len(s) == 3:
        stack.append(np.cross(s, t))
    mtrx = np.linalg.inv(np.vstack(stack).transpose())
    if len(s) == 3:
        mtrx = mtrx[:2]
    for point in points:
        ps, pt = np.dot(mtrx, point - a)
        if ps >= 0 and pt >= 0 and ps + pt <= 1:
            return True
    return False


def triangulate(polygon):
    """
    Converts a polygon to a set of triangles that cover the same area.

      * Convex and non-convex polygons are supported.
      * Polygon vertices must all be within a single plane, but the
        polygon itself may exist in 2 or 3 dimensional space
      * Clockwise and counter-clockwise winding supported.
      * Inverted polygons and polygons with holes are NOT supported.

    Args:
        polygon: A sequence of vertices making up the polygon, with each vertex
                 described as a sequence of coordinate components. The polygon
                 is implicitly closed: a polygon with N sides should have N
                 vertices.
    Returns:
        a generator of triangles, each specified in the same format as the
        input polygon
    """
    polygon = [np.array(x) for x in polygon]

    normal = calculate_normal(polygon)
    i = 0
    while len(polygon) > 2:
        if i >= len(polygon):
            raise ValueError("Triangulation failed")
        (a, b, c) = looped_slice(polygon, i, 3)
        triangle = (a, b, c)
        if ((a == b).all() or (b == c).all()):
            # Duplicate vertex, just skip
            del polygon[(i + 1) % len(polygon)]
            continue

        x = np.cross(c - b, b - a)
        dot = np.dot(normal, x)
        yld = False
        if dot > 1E-6:
            triangle = (a, b, c)
            if not any_point_in_triangle(triangle,
                                         looped_slice_inv(polygon, i, 3)):
                del polygon[(i + 1) % len(polygon)]
                yield triangle
                i = 0
                yld = True
        if not yld:
            i += 1
