# Copyright (c) 2015-2024 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

"""Convenience utilities for shape classes.

This module contains useful functions in the construction and management of
various shapes. The functions here do not belong in the namespace of particular
shape classes, but are stored within the subpackage because their primary
utility is in the context of the shape classes.
"""

import numpy as np


def translate_inertia_tensor(displacement, inertia_tensor, volume):
    """Apply the generalized parallel axis theorem for 3D inertia tensors."""
    # Should be a vector, but we need to promote it to take the outer produce
    # and atleast_2d handles asarray as well.
    displacement = np.atleast_2d(displacement)
    inner = np.squeeze(np.dot(displacement, displacement.T))
    outer = np.dot(displacement.T, displacement)
    return inertia_tensor + volume * (inner * np.eye(3) - outer)


def rotate_order2_tensor(rotation, tensor):
    """Transform a tensor with a similarity transformation."""
    return rotation @ tensor @ rotation.T


def _generate_ax(ax=None, axes3d=False):
    """Create an instance of :class:`matplotlib.axes.Axes` if needed.

    If an instance of :class:`matplotlib.axes.Axes` is provided, it will be
    passed through.

    Args:
        ax (:class:`matplotlib.axes.Axes`):
            The axes on which to draw the polygon. An instance will be
            created if this is None (Default value: None).
        axes3d (bool):
            Whether to use 3D axes (Default value: False).

    Returns
    -------
        :class:`matplotlib.axes.Axes`: Axes to plot on.
    """
    if ax is None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exception:
            raise ImportError(
                "matplotlib must be installed for plotting."
            ) from exception
        fig = plt.figure()
        if not axes3d:
            ax = fig.subplots()
            ax.set_aspect("equal", "datalim")
        else:
            # This import registers the 3d projection
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            ax = fig.add_subplot(111, projection="3d")
    return ax


def _set_3d_axes_equal(ax, limits=None):
    """Make axes of 3D plot have equal scale.

    Ensuring equal scale means that spheres appear as spheres, cubes as
    cubes, etc. This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Args:
        ax (:class:`matplotlib.axes.Axes`):
            Axes object.
        limits (:math:`(3, 2)` :class:`numpy.ndarray`):
            Axis limits in the form :code:`[[xmin, xmax], [ymin, ymax],
            [zmin, zmax]]`. If :code:`None`, the limits are auto-detected
            (Default value = :code:`None`).
    """
    # Adapted from https://stackoverflow.com/a/50664367

    if limits is None:
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    else:
        limits = np.asarray(limits)
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
    return ax


def _map_dict_keys(data, key_mapping):
    """Rename a dict's keys based on a mapping dict.

    If an instance of :class:`matplotlib.axes.Axes` is provided, it will be
    passed through.

    Args:
        data (dict):
            A dict with keys to be remapped
        key_mapping (dict):
            A dict of keys that should be renamed. The keys of this dict should
            correspond with the keys of data that are to be changed, and the values
            should correspond with the desired new keys.

    Returns
    -------
        dict: A dict with the select keys renamed to the mapped values.
    """
    return {key_mapping.get(key, key): value for key, value in data.items()}


_hoomd_dict_mapping = {"inertia_tensor": "moment_inertia", "radius": "sweep_radius"}
