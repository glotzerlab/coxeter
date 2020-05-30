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
