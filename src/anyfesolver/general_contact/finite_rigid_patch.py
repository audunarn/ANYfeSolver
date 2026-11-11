"""Opt-in finite V2D physical skins against a fixed planar rigid P1 mesh.

The existing shared-P1 penalty energy remains unchanged. Spatial-spin force
pullback B and additive-chart derivative A are deliberately distinct.
"""
import hashlib
import numpy as np
from scipy import sparse

from .core import ContactValidationReport
from .finite_rotation_skin import FiniteQ4SkinMap, FiniteSkinPose
from .triangulated_rigid_static import TriangulatedRigidManager, _prepare, _source_overlap_guard

PROFILE = 'S3_V2D_FIXED_PLANAR_RIGID_COROTATIONAL_PENALTY_STATIC_DEV_V0'


def admitted_definition(model, definition, settings):
    """Exact elastic owner, fixed target and existing penalty semantics only."""
    if (settings.shell_skin_kinematics != 'corotational_nodal_director_dev'
            or settings.general_s3_differentiation != 'jet2_s3_dev'
            or settings.enforcement_solver != 'nested_augmented'
            or settings.shell_pair_orientation != 'legacy'
            or settings.finite_frame_seed != 'common_reference_frame_dev'
            or settings.path_predictor != 'none'
            or settings.q4_source_pressure_basis != 'fine_nodal_q1'
            or settings.q4_source_pressure_knot_columns is not None
            or settings.q4_s3_shared_edge_tie_step_m is not None):
        return False
    try:
        from .finite_v2d_state import owner_contract
        owner_contract(model)
        adapters, _, _ = _prepare(model, definition, settings)
        if set(model.mesh.elements) != {a.element_id for a in adapters}:
            return False
    except (ValueError, TypeError, KeyError):
        return False
    return True


def validate_finite_rigid_patch(model, definition, settings, *, analysis_control='force'):
    if analysis_control not in {'force', 'displacement'} or not admitted_definition(model, definition, settings):
        return ContactValidationReport('unsupported', ('finite rigid patch requires exact elastic V2D owners, fixed planar P1 target and force/displacement penalty controls',), PROFILE)
    return ContactValidationReport('ok', (), PROFILE)


class _FixedTargetSkin:
    """Source owner maps plus exactly zero target variations; no fake DOFs."""
    def __init__(self, source, target):
        self.source = source
        self.target = np.asarray(target, float).copy()

    def pose(self, q):
        p = self.source.pose(q)
        zero = sparse.csr_matrix((3*len(self.target), self.source.n_dof))
        return FiniteSkinPose(np.vstack((p.positions, self.target)), p.directors,
            p.director_gradient, sparse.vstack((p.coordinate_map, zero)).tocsr(),
            sparse.vstack((p.spin_map, zero)).tocsr(),
            np.vstack((p.position_errors, np.zeros_like(self.target))))

    def spin_force_derivative(self, pose, local_force):
        count = len(self.source.reference)
        return self.source.spin_force_derivative(pose, np.asarray(local_force).reshape(-1, 3)[:count])


class FiniteRigidPatchManager(TriangulatedRigidManager):
    def __init__(self, model, definition, settings, initial_displacement=None, *, _pending_checkpoint=None):
        if not admitted_definition(model, definition, settings):
            raise ValueError('finite rigid patch definition is not admitted')
        self.model, self.definition, self.settings = model, definition, settings
        self.adapters, self.law, interaction = _prepare(model, definition, settings)
        self.law.stable_plane_gap=True
        from .s3_patch_static import _unique_work_map
        source, source_map = _unique_work_map(self.adapters, self.law.source_node_ids)
        self.reference = source, self.law.reference_target
        self.n_dof = model.mesh.dof_manager.total_dofs
        self.map = sparse.vstack((source_map, sparse.csr_matrix((3*len(self.law.target_node_ids), self.n_dof)))).tocsr()
        self.profile = PROFILE
        self.stiffness, self.gap_tolerance = interaction.normal_stiffness, interaction.gap_tolerance
        self.augmented = self.projected_work = False
        self.swept_trial_cutbacks = 0
        self.finite_skin = _FixedTargetSkin(FiniteQ4SkinMap(model, self.adapters, self.law.source_node_ids), self.reference[1])
        from .s3_pair_static import _normal
        self.reference_normals = (_normal(source[self.law.source_face_indices[0]]), _normal(self.reference[1][self.law.target_face_indices[0]]))
        from anyfesolver.corotational import _corotational_cache
        self.finite_elements = []
        for adapter in self.adapters:
            element = model.mesh.elements[adapter.element_id]
            reference = _corotational_cache(model, centered_shell=False).get(adapter.element_id)
            if reference is None or reference.category != 'shell':
                raise ValueError('actual finite V2D corotational reference required')
            length = min(np.linalg.norm(reference.coordinates[(i+1)%3]-reference.coordinates[i]) for i in range(3))
            if not np.isfinite(length) or length <= 0:
                raise ValueError('finite owner reference edge is invalid')
            dofs = np.array([model.mesh.dof_manager.get_node_dofs(n)[:6] for n in element.node_ids], dtype=np.intp)
            self.finite_elements.append((element, reference, dofs, length))
        self.last_finite_deformation_maxima = (0., 0., 0.)
        self._binding = self._contract_bytes()
        q = np.zeros(self.n_dof) if initial_displacement is None else np.asarray(initial_displacement, float)
        self.require_finite_deformation(q)
        self._pending_checkpoint = _pending_checkpoint
        self._committed_admitted = _pending_checkpoint is None
        lower = self.point_gap_lower(q)
        limit = (256*np.finfo(float).eps*max(1., np.max(np.abs(source)))
                 if _pending_checkpoint is None else self.gap_tolerance)
        if not np.isfinite(lower) or lower < -limit:
            raise ValueError('finite rigid initial or committed gap/chart is unresolved or overclosed')

    def admit_validated_committed_state(self, validated):
        """Called only after ordinary model, analysis and typed owner validation."""
        pending = self._pending_checkpoint
        if pending is None:
            raise ValueError('no pending finite rigid checkpoint')
        if (validated.payload['checkpoint_sha256'] != pending['checkpoint_sha256']
                or not np.array_equal(validated.displacements, pending['displacements'])):
            raise ValueError('validated finite rigid checkpoint differs from pending input')
        self.require_finite_deformation(validated.displacements)
        lower = self.point_gap_lower(validated.displacements)
        if not np.isfinite(lower) or lower < -self.gap_tolerance:
            raise ValueError('validated committed geometry is uncertified')
        self._committed_admitted = True
        self._pending_checkpoint = None

    def _contract_bytes(self):
        old = super()._contract_bytes()
        skin = self.finite_skin.source
        import json
        return old + json.dumps(dict(finite_field='nodal_exp_director_taylor32_v1',
            gap_arithmetic='affine_corner_exact_binary64_plane_v1',
            stable_plane_gap=bool(getattr(self.law,'stable_plane_gap',False)),
            force_work='spatial_spin_B_additive_chart_A_v1', limits=[1.2, .1, .01],
            reference=skin.reference.tolist(), directors=skin.directors.tolist(), offsets=skin.offsets.tolist(),
            target=self.finite_skin.target.tolist()), sort_keys=True).encode()

    def _positions(self, q):
        if self._contract_bytes() != self._binding:
            raise ValueError('finite rigid patch binding mutated')
        pose = self.finite_skin.pose(q)
        source = pose.positions[:len(self.law.source_node_ids)]
        _source_overlap_guard(source, self.law.source_face_indices)
        state = self.law.evaluate(source, self.reference[1])
        if np.any(state.reference_support <= 0) or not np.isfinite(state.reference_support).all():
            raise ValueError('finite rigid patch needs every positive supported P1 pressure row')
        self._reject_edge_only((source, self.reference[1]), state)
        return source, self.reference[1]

    def _gap_lower(self, positions, *, endpoint_errors=None):
        if endpoint_errors is None:
            raise ValueError('finite rigid gap requires explicit endpoint error boxes')
        from .finite_rigid_support import tool_normal_gap_lower
        source, target = positions
        es, et = endpoint_errors
        return min(tool_normal_gap_lower(source[si], source[si], target[ti],
            self.law.source_faces[i].reference_positions, self.law.target_faces[j].reference_positions,
            endpoint_errors=(es[si], es[si]))
            for i, si in enumerate(self.law.source_face_indices) for j, ti in enumerate(self.law.target_face_indices))

    def point_gap_lower(self, q):
        return self._gap_lower(self._positions(q), endpoint_errors=self._position_errors(q))

    def evaluate(self, q, *, tangent=True):
        if not self._committed_admitted:
            raise ValueError('finite rigid checkpoint owner validation is pending')
        self.require_finite_deformation(q)
        return super().evaluate(q, tangent=tangent)

    def _swept_admissible(self, start, end):
        if not self._committed_admitted:
            raise ValueError('finite rigid checkpoint owner validation is pending')
        from .finite_rigid_sweep import finite_rigid_path_admissible
        return finite_rigid_path_admissible(self, start, end)

    def restart_contract(self):
        if self._contract_bytes() != self._binding:
            raise ValueError('finite rigid restart binding mutated')
        from .finite_v2d_state import owner_contract, COORDINATES
        return dict(profile=PROFILE, rigid_mesh_contract_sha256=hashlib.sha256(self._binding).hexdigest(),
            finite_state_owners=owner_contract(self.model), element_state_coordinates=COORDINATES,
            force_work='spatial_spin_B_additive_chart_A_v1', finite_deformation_limits=[1.2, .1, .01])


from contextlib import contextmanager

@contextmanager
def validated_restart_scope(model, definition, settings, payload, *, analysis_control):
    """Private ordinary-driver join; construction never admits incoming state."""
    from .core import _ACTIVE_CONTACT
    report = validate_finite_rigid_patch(model, definition, settings, analysis_control=analysis_control)
    if not report.valid:
        raise ValueError('; '.join(report.issues))
    manager = FiniteRigidPatchManager(model, definition, settings,
        np.asarray(payload['displacements'], float), _pending_checkpoint=payload)
    token = _ACTIVE_CONTACT.set(manager)
    try:
        yield manager
    finally:
        _ACTIVE_CONTACT.reset(token)
