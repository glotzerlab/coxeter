import numpy as np


def translate_inertia_tensor(displacement, I, volume):
    """Apply the generalized parallel axis theorem for 3D inertia tensors."""
    # Should be a vector, but we need to promote it to take the outer produce
    # and atleast_2d handles asarray as well.
    displacement = np.atleast_2d(displacement)
    inner = np.squeeze(np.dot(displacement, displacement.T))
    outer = np.dot(displacement.T, displacement)
    return I + volume*(inner*np.eye(3) - outer)


def rotate_order2_tensor(rotation, I):
    """Apply the transformation rule for tensor quantities (a similarity
    transformation)."""
    return rotation @ I @ rotation.T
