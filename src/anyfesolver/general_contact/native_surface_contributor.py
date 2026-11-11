"""Material-surface contribution to the existing retained native/S3 assembly.

Development bridge: uses the existing owner's prepared structural domain and
constraints. Its old line contact is not assembled. This is not a new solver or
an admission of a single point as a distributed swept-surface discretization.
"""
from dataclasses import dataclass

import numpy as np

from anyfesolver._ge_beam3_p5_seeded.core import sha
from anyfesolver.corotational import _rotation_right_jacobian
from .native_retained_s3_static import NativeRetainedRoundS3Analysis
from .native_retained_line_s3_work import NativeRetainedLineS3Work
from .native_retained_station import _readonly
from .native_surface_s3_friction import NativeS3MaterialConfiguration, NativeS3MaterialPointFriction
from .native_surface_patch import NativeS3MaterialSurfacePatch


@dataclass(frozen=True)
class NativeS3MaterialAssembly:
    mechanical: object
    shell_displacement: np.ndarray
    residual: np.ndarray
    spatial_tangent: np.ndarray
    origin_chart_tangent: np.ndarray
    metrics: tuple
    native_responses: tuple
    material: object


class NativeS3MaterialContributor:
    """Bound one persistent material pair to an existing structural owner."""
    def __init__(self, owner, pair):
        if type(owner) is not NativeRetainedRoundS3Analysis or type(pair) not in (NativeS3MaterialPointFriction,NativeS3MaterialSurfacePatch):
            raise TypeError('exact retained native/S3 owner and material pair required')
        if type(owner.contact) is not NativeRetainedLineS3Work:
            raise ValueError('normal material-patch owner does not admit the existing friction transaction')
        owner._guard()
        element_id, element = owner.context.elements[0]
        if (pair.source.element_id != element_id or
                sha(pair.source.reference.canonical_data()) != sha(element.operator.reference.canonical_data())):
            raise ValueError('material contributor native reference does not bind owner')
        expected = owner.contact.shell
        actual = pair.target.adapter
        if (actual.body_id != expected.body_id or actual.element_id != expected.element_id
                or actual.node_ids != expected.node_ids or actual.side != expected.side
                or not np.array_equal(actual.reference_positions, expected.reference_positions)
                or actual.work_map.shape != expected.work_map.shape
                or (actual.work_map != expected.work_map).nnz):
            raise ValueError('material contributor physical S3 map does not bind owner')
        self.owner, self.pair = owner, pair
        self.identity = sha(dict(profile='NATIVE_S3_MATERIAL_CONTRIBUTOR_DEV_V1',
                                 structural_owner=owner.identity, pair=pair.definition_sha256))
        self._binding = (owner.identity, pair.definition_sha256)

    def _guard(self):
        self.owner._guard()
        if ((self.owner.identity, self.pair.definition_sha256) != self._binding
                or sha(self.pair._definition()) != self.pair.definition_sha256
                or self.identity != sha(dict(profile='NATIVE_S3_MATERIAL_CONTRIBUTOR_DEV_V1',
                    structural_owner=self.owner.identity, pair=self.pair.definition_sha256))):
            raise ValueError('material contributor definition changed')

    def evaluate(self, origin, history, native_increment, shell_increment, parameter):
        """One candidate from the physical origin; no material/history commit.

        All42native coordinates remain in the structural solve. Contact acts on
        physical translations/cell rotations, leaving algebraic resultants to
        their structural equations. No independent element condensation occurs.
        """
        owner, pair = self.owner, self.pair
        self._guard()
        # The eventual owner must issue and bind the complete accepted origin.
        # This local assembly bridge validates supports but cannot establish
        # that an independently supplied history belongs to that origin.
        owner.context.make(origin.mechanical.descriptor())
        if (origin.shell_displacement.shape != (owner.shell_stiffness.shape[0],)
                or not np.all(np.isfinite(origin.shell_displacement))
                or np.any(origin.shell_displacement[owner.shell_fixed])):
            raise ValueError('material contributor origin violates shell supports')
        increment = np.asarray(native_increment, dtype=float)
        delta_shell = np.asarray(shell_increment, dtype=float)
        if (increment.shape != (owner.context.count,) or delta_shell.shape != origin.shell_displacement.shape
                or not np.all(np.isfinite(delta_shell))
                or np.any(delta_shell[owner.shell_fixed]) or not np.isfinite(parameter)):
            raise ValueError('material contributor requires finite supported increments and load')
        mechanical = owner.context.advance(origin.mechanical, increment)
        shell = origin.shell_displacement+delta_shell
        nodes = owner.context.nodes[0]
        old = NativeS3MaterialConfiguration(origin.mechanical.positions[nodes],
            origin.mechanical.position_low[nodes], origin.mechanical.cell_rotations[0],
            origin.shell_displacement)
        local = increment[np.asarray(owner.context.slots[0], dtype=int)]
        translation = np.array([local[6*i:6*i+3] for i in range(3)])
        angular = local[18:24].reshape(2, 3)
        material = pair.evaluate_path(old, history, translation, angular, delta_shell,
                                      maximum_intervals=owner.maximum_swept_intervals)
        current = material.configuration
        if not all(np.array_equal(a, b) for a,b in (
                (current.positions, mechanical.positions[nodes]),
                (current.position_low, mechanical.position_low[nodes]),
                (current.cell_rotations, mechanical.cell_rotations[0]),
                (current.shell_displacement, shell))):
            raise ValueError('material path does not match authoritative native candidate')
        residual, tangent, responses = owner._assemble_structural(
            mechanical, shell, parameter, origin.native_histories)
        metrics = owner._add_spatial_contact(residual, tangent,
            material.contribution.residual, material.contribution.spatial_tangent())
        # evaluate accepts TOTAL origin-chart increments, whereas the native
        # structural/contact tangent differentiates spatial endpoint corrections.
        # For Exp(w+dw) = Exp(J_l(w)dw) Exp(w), K_chart = K_spatial J_l(w).
        # Translation and retained-resultant columns remain additive.
        origin_chart = tangent.copy()
        starts = [6*i+3 for i in range(len(owner.context.node_ids))]
        starts += [owner.context.nodal_count+3*i for i in range(2)]
        for first in starts:
            origin_chart[:,first:first+3] = tangent[:,first:first+3] @ _rotation_right_jacobian(
                -increment[first:first+3])
        return NativeS3MaterialAssembly(mechanical, _readonly(shell), _readonly(residual),
            _readonly(tangent), _readonly(origin_chart), metrics, tuple(responses), material)
