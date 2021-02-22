# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
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

    Returns:
        :class:`matplotlib.axes.Axes`: Axes to plot on.
    """
    if ax is None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib must be installed for plotting.")
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
