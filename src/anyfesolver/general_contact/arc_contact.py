"""Fixed-tool penalty transaction adapter for the existing arc continuation."""
from contextlib import contextmanager
from dataclasses import replace
import numpy as np
from .core import ContactDefinition, ContactSolverSettings, ContactValidationReport
from .triangulated_rigid import TriangulatedRigidContactBody
from .triangulated_rigid_static import TriangulatedRigidManager, validate_triangulated_rigid
from .runtime import _ACTIVE_CONTACT
from . import transient_results_archive as codec

PROFILE = 'S3_FIXED_TRIANGULATED_RIGID_ARC_PENALTY_DEV_V1'
STATE_SCHEMA = 'ANYSOLVER_ARC_CONTACT_STATE_V1'
PRELOAD_SCHEMA = 'ANYSOLVER_ARC_CONTACT_STATE_PRELOAD_V2'


def validate_arc_contact(model, definition, settings=None):
    if type(definition) is ContactDefinition and type(settings) is ContactSolverSettings:
        if settings.finite_frame_seed != 'common_reference_frame_dev':
            return ContactValidationReport('invalid', (
                'finite_frame_seed requires contact_path_program; unsupported in arc contact',), PROFILE)
        if settings.shell_skin_kinematics != 'reference_linear':
            return ContactValidationReport('invalid', (
                'shell_skin_kinematics requires contact_path_program; unsupported in arc contact',), PROFILE)
        if settings.path_predictor != 'none':
            return ContactValidationReport('invalid', (
                'path_predictor requires contact_path_program; unsupported in arc contact',), PROFILE)
        if settings.shell_pair_orientation != 'legacy':
            return ContactValidationReport('invalid', (
                'shell_pair_orientation requires contact_path_program; unsupported in arc contact',), PROFILE)
    if (type(definition) is ContactDefinition and type(settings) is ContactSolverSettings
            and settings.general_s3_differentiation == "jet2_q4_q4_dev"):
        return ContactValidationReport('invalid', (
            'jet2_q4_q4_dev requires a static deformable Q4/Q4 projected-penalty pair; unsupported in arc contact',), PROFILE)
    if type(definition) is not ContactDefinition or not definition.rigid_bodies or any(
            type(body) is not TriangulatedRigidContactBody for body in definition.rigid_bodies):
        return ContactValidationReport('unsupported', ('arc contact requires fixed triangulated rigid tools',), PROFILE)
    report = validate_triangulated_rigid(model, definition, settings, analysis_control='force')
    return replace(report, profile=PROFILE)


class ArcContactAdapter:
    """The penalty contributor has no pressure or material history to import."""
    def __init__(self, model, definition, settings, *, preload=False):
        report = validate_arc_contact(model, definition, settings)
        if not report.valid: raise ValueError('unsupported arc contact: ' + '; '.join(report.issues))
        self.manager = TriangulatedRigidManager(model, definition, settings)
        self.history = []
        self.preload_enabled = bool(preload)
        if self.preload_enabled:
            self._validate_preload_material_scope()
        self.base = None
        self.definition, self.settings = definition, settings
        self._bound_contract = self.manager.restart_contract()

    def _validate_preload_material_scope(self):
        # Replay below uses a common elastic origin, including unselected elements.
        from ..fe_core import Material
        for element in self.manager.model.mesh.elements.values():
            material = self.manager.model.materials[element.material_name]
            if (type(material) is not Material or
                    not (float(material.yield_stress) == 0. or float(material.yield_stress) == float('inf')) or
                    getattr(material,'hardening_curve',None) is not None):
                raise ValueError('arc preload replay requires whole assembled model elastic material')

    def _guard(self):
        if self.preload_enabled:
            self._validate_preload_material_scope()
        if self.manager.restart_contract() != self._bound_contract:
            raise ValueError('arc contact operator mutated')

    def evaluate(self, displacement, *, tangent=True):
        self._guard()
        return self.manager.evaluate(displacement, tangent=tangent)

    def restart_contract(self):
        self._guard()
        result = {'profile': PROFILE, 'contributor': self._bound_contract,
                'state_policy': 'stateless_penalty_accepted_pose_history_exact_bits_v1'}
        if self.preload_enabled:
            result['base_policy'] = 'ordinary_static_committed_preload_additive_support_v2'
            result['material_scope'] = 'whole_assembled_model_elastic_isotropic_v1'
        return result

    def admissible(self, start, end):
        self._guard()
        return self.manager._swept_admissible(start, end)

    def prepare(self, displacement, step_index, load_factor):
        self._guard()
        _, _, result = self.manager.evaluate(displacement, tangent=False)
        if not result.gap_tolerance_met: raise ValueError('arc endpoint gap exceeds tolerance')
        candidate = {'step_index': int(step_index), 'load_factor': float(load_factor),
                'displacements': np.array(displacement, dtype=float, copy=True),
                'contact': result.to_dict()}
        codec._encode(self.history + [candidate], codec._Budget(codec.MAX_BYTES), {})
        return candidate

    def commit(self, candidate):
        # All potentially failing geometry and codec validation occurs before
        # the structural commit; append is the only contact commit operation.
        self.history.append(candidate)

    def encoded_state(self):
        state = {'schema': STATE_SCHEMA,
                'history': codec._encode(self.history, codec._Budget(codec.MAX_BYTES), {})}
        if self.preload_enabled:
            state['schema'] = PRELOAD_SCHEMA
            state['base'] = codec._encode(self.base, codec._Budget(codec.MAX_BYTES), {})
        return state

    def restore(self, state, path, final_displacements):
        keys = {'schema','history','base'} if self.preload_enabled else {'schema','history'}
        schema = PRELOAD_SCHEMA if self.preload_enabled else STATE_SCHEMA
        if type(state) is not dict or set(state) != keys or state['schema'] != schema:
            raise ValueError('incompatible arc contact state')
        if self.preload_enabled:
            self.base = codec._decode(state['base'], codec._Budget(codec.MAX_BYTES), {})
            self.validate_base_inventory()
        history = codec._decode(state['history'], codec._Budget(codec.MAX_BYTES), {})
        if type(history) is not list or len(history) != path['step_index']:
            raise ValueError('incomplete arc contact accepted history')
        previous = self.base_pose()
        for index, row in enumerate(history, 1):
            if type(row) is not dict or set(row) != {'step_index','load_factor','displacements','contact'}:
                raise ValueError('invalid arc contact record')
            pose = row['displacements']
            if type(pose) is not np.ndarray or pose.shape != (self.manager.n_dof,) or not np.all(np.isfinite(pose)):
                raise ValueError('invalid arc contact accepted pose')
            step = path['steps'][index-1]
            if row['step_index'] != index or row['load_factor'] != step['load_factor']:
                raise ValueError('arc contact record differs from accepted structural path')
            if not self.admissible(previous, pose): raise ValueError('uncertified restored arc contact path')
            observed = self.prepare(pose, index, row['load_factor'])
            if codec._canonical(codec._encode(observed,codec._Budget(codec.MAX_BYTES),{})) != codec._canonical(
                    codec._encode(row,codec._Budget(codec.MAX_BYTES),{})):
                raise ValueError('arc contact accepted state failed replay')
            previous = pose
        if history and previous.tobytes() != np.asarray(final_displacements).tobytes():
            raise ValueError('arc contact final pose differs from structural checkpoint')
        if not history and (np.asarray(final_displacements).tobytes() != self.base_pose().tobytes() or path['load_factor'] != 0.):
            raise ValueError('empty contact history requires virgin zero endpoint')
        self.history = history

    def base_pose(self):
        return np.zeros(self.manager.n_dof) if self.base is None else self.base['displacements']

    def validate_base_inventory(self):
        base = self.base
        if type(base) is not dict or set(base) != {'displacements','element_states','history','summary'}:
            raise ValueError('invalid committed arc preload base')
        pose = base['displacements']
        if type(pose) is not np.ndarray or pose.shape != (self.manager.n_dof,) or not np.all(np.isfinite(pose)):
            raise ValueError('invalid arc preload pose')
        if type(base['element_states']) is not dict or type(base['history']) is not list or not base['history']:
            raise ValueError('invalid arc preload history/material origin')
        summary = base['summary']
        if (type(summary) is not dict or type(summary.get('step_count')) is not int or
                type(summary.get('steps')) is not list or
                summary['step_count'] != len(base['history']) or
                len(summary['steps']) != len(base['history'])):
            raise ValueError('preload history/summary inventory mismatch')
        for index,(row,step) in enumerate(zip(base['history'],summary['steps']),1):
            if (type(row) is not dict or type(step) is not dict or
                    row.get('step_index') != index or step.get('step_index') != index or
                    row.get('load_factor') != step.get('load_factor')):
                raise ValueError('preload history/summary index or factor mismatch')
        previous = np.zeros(self.manager.n_dof)
        factor = 0.
        for index,row in enumerate(base['history'],1):
            if type(row) is not dict or set(row) != {'step_index','load_factor','displacements','contact'}:
                raise ValueError('invalid preload accepted record')
            current = row['displacements']
            if type(row['step_index']) is not int or type(row['load_factor']) is not float or row['step_index'] != index or not factor < row['load_factor'] <= 1.:
                raise ValueError('invalid preload accepted factors')
            if not self.admissible(previous,current): raise ValueError('uncertified preload accepted sweep')
            observed = self.prepare(current,row['step_index'],row['load_factor'])
            if codec._canonical(codec._encode(observed,codec._Budget(codec.MAX_BYTES),{})) != codec._canonical(codec._encode(row,codec._Budget(codec.MAX_BYTES),{})):
                raise ValueError('preload accepted contact replay mismatch')
            previous, factor = current,row['load_factor']
        if factor != 1. or previous.tobytes() != pose.tobytes():
            raise ValueError('incomplete preload committed base')

    def preload(self, model, *, summary, **kwargs):
        # Compose ordinary static transactions under the caller's existing lease.
        from .core import contact_scope
        from ..nonlinear_static import _solve_static_nonlinear_under_lease, _run_with_nonlinear_state_cleanup
        with contact_scope(model,self.definition,self.settings,analysis_control='force'):
            result = _run_with_nonlinear_state_cleanup(_solve_static_nonlinear_under_lease,
                model,record_increment_snapshots=True,**kwargs)
        if result.status == 'completed':
            history = [self.prepare(snapshot.displacements,snapshot.step_index,snapshot.load_factor)
                       for snapshot in result.snapshots]
            from ..nonlinear_state import NonlinearStateStore, StateMaterializationPolicy
            states = result.element_states
            if isinstance(states,NonlinearStateStore):
                states = states.materialize_owned(policy=StateMaterializationPolicy.SAVED_STATE)
            base = {'displacements':np.array(result.displacements,copy=True),
                    'element_states':states,'history':history,'summary':summary(result)}
            encoded = codec._encode(base,codec._Budget(codec.MAX_BYTES),{})
            self.base = codec._decode(encoded,codec._Budget(codec.MAX_BYTES),{})
            self.validate_base_inventory()
        return result

    def validate_path(self, path, reduce_pose, full_pose, metric, arc_tolerance):
        previous_q = reduce_pose(self.base_pose(), 0.)
        previous_lambda = 0.
        last_delta = None
        for row, step in zip(self.history, path['steps']):
            pose = row['displacements']; lam = row['load_factor']
            q = reduce_pose(pose, lam)
            reconstructed = full_pose(q, lam)
            guard = 512*np.finfo(float).eps*max(1.,float(np.linalg.norm(pose)))
            if np.linalg.norm(reconstructed-pose) > guard:
                raise ValueError('arc contact history violates structural constraints')
            delta_q = q-previous_q; delta_lambda = lam-previous_lambda
            norm2 = metric(delta_q,delta_lambda,delta_q,delta_lambda)
            radius2 = step['arc_radius']**2
            if abs(norm2-radius2) > arc_tolerance*radius2+guard**2:
                raise ValueError('arc contact history violates spherical constraint')
            for actual, saved in ((np.linalg.norm(pose),step['displacement_norm']),
                    (np.sqrt(norm2),step['path_increment_norm']),
                    (delta_lambda,step['load_increment'])):
                if abs(actual-saved) > guard:
                    raise ValueError('arc contact history differs from structural step metric')
            previous_q,previous_lambda=q,lam;last_delta=delta_q
        if last_delta is not None:
            if np.linalg.norm(last_delta-np.asarray(path['previous_dq'])) > guard or \
                    abs(delta_lambda-path['previous_dlambda']) > guard:
                raise ValueError('arc contact predictor history differs from accepted path')


@contextmanager
def arc_contact_scope(model, definition, settings, *, preload=False):
    adapter = ArcContactAdapter(model, definition, settings, preload=preload)
    token = _ACTIVE_CONTACT.set(adapter)
    try: yield adapter
    finally: _ACTIVE_CONTACT.reset(token)
