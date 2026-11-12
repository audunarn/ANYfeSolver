"""Private retained-cell kinetic factor at rest, without a dense mass root.

The spatial velocity is I*v - skew(U*lift)*omega, with body components
in U*R0. These are the preserved lifted-inertia equations; nodal rotation
traces have exactly zero inertia. No cell condensation or mass floor.
"""
import numpy as np
from ._ge_beam3_p5.arrays import _frames
from ._ge_beam3_p5.algebra import skew
from ._ge_beam3_p5_centered.mass import _CenteredRestInertia
from ._native_reference_modal import _owned


def current_lifted_kinetic_factor(reference, section_inertia, rotations, *, order):
    cells = _frames(rotations, 2, 'current cell rotations')
    source = _CenteredRestInertia(reference, section_inertia, order=order)
    root = np.linalg.cholesky(source.section_mass).T
    rows = []
    for cell, left, right, t, frame, offset, measure in source._stations:
        q = cells[cell]@frame; lift = cells[cell]@offset
        velocity = np.zeros((6, 24)); slot = slice(18+3*cell, 21+3*cell)
        velocity[:3, 6*left:6*left+3] = (1-t)*np.eye(3)
        velocity[:3, 6*right:6*right+3] = t*np.eye(3)
        velocity[:3, slot] = -skew(lift); velocity[3:, slot] = np.eye(3)
        rows.append(np.sqrt(measure)*root@np.kron(np.eye(2), q).T@velocity)
    return _owned(np.vstack(rows))
