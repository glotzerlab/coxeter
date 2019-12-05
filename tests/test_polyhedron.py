import pytest
import numpy as np
from euclid.shape_classes.polyhedron import Polyhedron
from euclid.polyhedron import ConvexPolyhedron


# Need to declare this outside the fixture so that it can be used in multiple
# fixtures (pytest does not allow fixtures to be called).
def get_cube_points():
    return np.asarray([[0, 0, 0],
                       [0, 1, 0],
                       [1, 1, 0],
                       [1, 0, 0],
                       [0, 0, 1],
                       [0, 1, 1],
                       [1, 1, 1],
                       [1, 0, 1]])


def get_oriented_cube_facets():
    return np.array([[0, 1, 2, 3],  # Bottom face
                     [4, 7, 6, 5],  # Top face
                     [0, 3, 7, 4],  # Left face
                     [1, 5, 6, 2],  # Right face
                     [3, 2, 6, 7],  # Front face
                     [0, 4, 5, 1]])  # Back face


def get_oriented_cube_normals():
    return np.asarray([[0, 0, -1],
                       [0, 0, 1],
                       [0, -1, 0],
                       [0, 1, 0],
                       [1, 0, 0],
                       [-1, 0, 0]])


@pytest.fixture
def cube_points():
    return get_cube_points()


@pytest.fixture
def cube():
    return Polyhedron(get_cube_points())


def test_surface_area(cube):
    """Test surface area calculation."""
    assert cube.surface_area == 6


def test_volume():
    cube = Polyhedron(get_cube_points(), facets=get_oriented_cube_facets(),
                      normals=get_oriented_cube_normals())
    assert cube.volume == 1


def test_merge_facets():
    cube = Polyhedron(get_cube_points())
    # cube._find_equations()
    # print('equations: ')
    # print(cube._equations)
    # cube._find_neighbors()
    # print('neighbors: ')
    # print(cube._connectivity_graph)
    # assert np.all(np.sum(cube._connectivity_graph, axis=1) == 3)
    cube.merge_facets()
    assert len(cube.facets) == 6
    # print("Vol: ", cube.volume)
    # print("Facets: ", get_oriented_cube_facets())
    # print("Normals: ", get_oriented_cube_normals())
    # c2 = ConvexPolyhedron(get_cube_points())
    # print("Computed facets: ", c2.facets)
    # print("Computed normals: ", c2.equations[:, :3])
    # assert 0
# def test_volume(cube_points):
    # """Test volume calculation."""
    # faces =
# def test_volume(cube):
    # """Test volume calculation."""
    # assert cube.volume == 1
