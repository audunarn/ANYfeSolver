"""SOURCE_ONLY_UNRUN: supported native tangent evidence at saved g13/g14/g15.

No continuation, fold validation, stability or capacity acceptance is performed.
The prior screen's remaining allowance does not authorize this assessment.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from anyfesolver import stiffened_shell as ss
from anyfesolver import _stiffened_shell_displacement_corrector as corrector
from anyfesolver._stiffened_shell_arc_metric import (
    ChordDomainError, require_director_chart, require_metric_owner)
from anyfesolver._stiffened_shell_rotation_charts import physical_frames
from anyfesolver._stiffened_shell_bl5_local_experimental import BL5TrialDomainError
from anyfesolver._ge_beam3_pose_joint import _exp_terms
from anyfesolver._stiffened_shell_ge_finite_experimental import (
    _material_observables, _RotationStepTooLarge)
from examples import stiffened_shell_native_capacity as case


MAX_MIXED_ROWS = 2048
EQUILIBRIUM_LIMIT = 1e-8
COMPATIBILITY_LIMIT = 1e-9
CONTROL_LIMIT = 1e-9
FD_STEPS = (1e-5,1e-6)  # Fixed registered pair; no retry/refinement sweep.
ELASTIC_FD_LIMIT = 1e-6
SAVED = {
    13: (49821,'7ecc0cc526d7adf2fba5d0059c381c94ee130a90379fec7383a6406ae08fcebf',
         190891,'3b8a60a70f761b62ebcb0017d03b4d69d09781a8941765e4f5aa8f6ed0675afd'),
    14: (50115,'f632e5fdac61c9c160949bab025c20a1f37b9becd27122dbc7b5130316dae69e',
         190768,'c82662e7625084a10285e1da0755cff20745439c15181532014c42596f01bef0'),
    15: (50410,'0c4ed52276fe0d0a21c20ae034f05d86b2bf16eda7216697a1fd12e79adc8f35',
         191114,'765f06accd324691121a92312c6da6a0915581924dd698395b9a09e9ef256e2d'),
}


class AssessmentRefusal(RuntimeError):
    """Supported domain or numerical evidence is inconclusive; never a fold."""
    def __init__(self,code,details=None):
        super().__init__(code)
        self.code = code
        self.details = {} if details is None else details


def directional_error_witness(difference,rows,equation_free,constraint_start,full_direction):
    """Relative mechanical FD error against the declared full unit-factor load.

    Compatibility keeps its original reference row scales; it has no load
    denominator. This is separate from endpoint and linear-border residuals.
    """
    if not all(np.isfinite(v).all() for v in (difference,rows,full_direction)):
        raise AssessmentRefusal('nonfinite_directional_or_plateau_witness')
    reference = float(np.linalg.norm((rows*full_direction)[equation_free]))
    if not np.isfinite(reference) or reference <= 0.:
        raise AssessmentRefusal('nonzero_declared_full_load_fd_reference_required')
    absolute = float(np.linalg.norm((rows*difference)[equation_free]))
    compatibility = float(np.max(np.abs(rows[constraint_start:]*difference[constraint_start:]),initial=0.))
    if not np.isfinite([absolute,compatibility,absolute/reference]).all():
        raise AssessmentRefusal('nonfinite_directional_or_plateau_witness')
    return dict(full_unit_factor_load_reference_norm=reference,
        equilibrium_absolute_work_scaled_error=absolute,
        equilibrium_relative_full_load_error=absolute/reference,
        compatibility_reference_ratio=compatibility)


def supported_branch_tangent(matrix,direction,free,*,tip_dof,physical_translation_dofs,
                             length,column_scales,row_scales,constraint_start,
                             orientation=1):
    """One nonsymmetric sparse border: J t - p a=0, e_tip t/L=1.

    t is per unit xi=z_tip/L. a=d(lambda)/d(xi); lambda is the actual
    proportional load factor. Rows and columns retain their original ordering.
    This operation does not establish a simple fold or forward slope accuracy.
    """
    if not sparse.issparse(matrix) or matrix.shape[0] != matrix.shape[1]:
        raise AssessmentRefusal('bounded_sparse_mixed_operator_required')
    n = matrix.shape[0]
    if not 1 <= n <= MAX_MIXED_ROWS or not 0 <= constraint_start <= n:
        raise AssessmentRefusal('mixed_row_bound_or_constraint_numbering')
    free = np.asarray(free)
    physical = np.asarray(physical_translation_dofs)
    if (free.ndim != 1 or not np.issubdtype(free.dtype,np.integer) or
        len(np.unique(free)) != len(free) or np.any(free < 0) or np.any(free >= n) or
        physical.ndim != 1 or not np.issubdtype(physical.dtype,np.integer) or
        np.any(physical < 0) or np.any(physical >= constraint_start)):
        raise AssessmentRefusal('complete_free_and_physical_numbering_required')
    if orientation != 1 or type(orientation) is not int:
        raise AssessmentRefusal('fixed_positive_physical_orientation_required')
    if type(tip_dof) is not int or tip_dof not in physical or tip_dof not in free:
        raise AssessmentRefusal('free_physical_translation_progress_required')
    if not np.all(np.isin(np.arange(constraint_start,n),free)):
        raise AssessmentRefusal('constraint_multipliers_must_remain_free')
    if not np.isfinite(length) or length <= 0.:
        raise AssessmentRefusal('physical_length_required')
    p = np.asarray(direction,dtype=float)
    s = np.asarray(column_scales,dtype=float)
    d = np.asarray(row_scales,dtype=float)
    if (any(v.shape != (n,) or not np.isfinite(v).all() for v in (p,s,d)) or
        np.any(s <= 0.) or np.any(d <= 0.) or not np.isfinite(matrix.data).all()):
        raise AssessmentRefusal('finite_original_operator_and_declared_unit_scales_required')
    if np.any(p[np.setdiff1d(np.arange(n),physical)]) or not np.any(p[np.intersect1d(free,physical)]):
        raise AssessmentRefusal('nonzero_free_physical_dead_load_required')
    gradient = np.zeros(n); gradient[tip_dof] = 1./length
    scaled = sparse.diags(d)@matrix@sparse.diags(s)
    try:
        y,amplitude = corrector.bordered_path_correction(
            scaled,np.zeros(n),d*p,gradient*s,-1.,free)
    except corrector.DisplacementCorrectorFailure as error:
        raise AssessmentRefusal('singular_or_nonfinite_supported_border',
                                dict(type=type(error).__name__,message=str(error))) from error
    tangent = s*y
    original = np.asarray(matrix@tangent-p*amplitude)
    eq_free = free[free < constraint_start]
    compatibility = original[constraint_start:]
    equation_ratio = float(np.linalg.norm((d*original)[eq_free]))
    compatibility_ratio = float(np.max(np.abs(d[constraint_start:]*compatibility),initial=0.))
    control_error = float(gradient@tangent-1.)
    evidence = dict(original_equation_residual=original[:constraint_start].tolist(),
        original_compatibility_residual=compatibility.tolist(),
        equation_free_rows=eq_free.tolist(),original_row_denominators=(1./d).tolist(),
        equation_reference_work_ratio=equation_ratio,
        compatibility_reference_ratio=compatibility_ratio,physical_control_residual=control_error)
    if (not np.isfinite(tangent).all() or not np.isfinite(amplitude) or
        not np.isfinite(original).all() or
        not np.isfinite([equation_ratio,compatibility_ratio,control_error,amplitude/length]).all() or
        not np.isfinite(1./d).all() or
        equation_ratio > EQUILIBRIUM_LIMIT or compatibility_ratio > COMPATIBILITY_LIMIT or
        abs(control_error) > CONTROL_LIMIT):
        raise AssessmentRefusal('original_equation_compatibility_or_control_residual',evidence)
    return dict(tangent=tangent,load_tangent=float(amplitude),
        d_factor_d_tip_over_length=float(amplitude),d_factor_d_tip=float(amplitude/length),
        derivative_units=dict(progress='dimensionless z_tip/reference_length',
                              d_factor_d_tip='per supplied length unit'),
        residuals=evidence,slope_sign_certified=False,
        forward_slope_accuracy='unestablished; small backward residual is insufficient')


def supported_sign_assessment(records):
    """No raw load reversal is admitted as a supported tangent sign witness."""
    signs = []
    for row in records:
        if row.get('status') != 'SUPPORTED_TANGENT_EVIDENCE' or 'd_factor_d_tip_over_length' not in row:
            return dict(status='INCONCLUSIVE',simple_fold_proved=False,first_fold_proved=False,
                        stability_proved=False,capacity_proved=False,reason='missing_supported_tangent')
        if not np.isfinite(row['d_factor_d_tip_over_length']):
            return dict(status='INCONCLUSIVE',simple_fold_proved=False,first_fold_proved=False,
                        stability_proved=False,capacity_proved=False,reason='nonfinite_supported_tangent')
        signs.append(int(np.sign(row['d_factor_d_tip_over_length'])))
    nonzero = [(i,s) for i,s in enumerate(signs) if s]
    brackets = [(a[0],b[0]) for a,b in zip(nonzero,nonzero[1:]) if a[1] != b[1]]
    return dict(status='CANDIDATE' if brackets else 'NO_SUPPORTED_TANGENT_SIGN_BRACKET',
        reported_signs=signs,bracket_record_indices=brackets,slope_signs_certified=False,
        simple_fold_proved=False,first_fold_proved=False,stability_proved=False,capacity_proved=False,
        continuity_proved=False,reason='forward slope accuracy and fold transversality/rank are not proved')


def load_saved(folder,generation):
    binary_size,binary_sha,packet_size,packet_sha = SAVED[generation]
    raw = (Path(folder)/f'accepted-g{generation}.bin').read_bytes()
    metadata = (Path(folder)/f'accepted-g{generation}.json').read_bytes()
    if (len(raw) != binary_size or sha256(raw).hexdigest() != binary_sha or
        len(metadata) != packet_size or sha256(metadata).hexdigest() != packet_sha):
        raise ValueError('saved fold-candidate artifact differs from its frozen binding')
    packet = json.loads(metadata)
    if (packet['generation'] != generation or packet['accepted_status'] != 'accepted_prefix' or
        packet['checkpoint_sha256'] != binary_sha or packet['checkpoint_size_bytes'] != binary_size or
        packet['observation']['state']['generation'] != generation):
        raise ValueError('saved fold-candidate packet identity differs')
    return raw,deepcopy(packet)


def _poses_and_directions(owner,state,tangent,metric):
    require_director_chart(state.q[:owner._shell_count])
    nodal = owner._shell_count+6*len(owner._beam_ids)
    poses = np.concatenate((physical_frames(state.q[:owner._shell_count],state.shell_anchors),state.frames))
    translations = tangent[:nodal].reshape(-1,6)[:,:3].copy()
    spins = tangent[:nodal].reshape(-1,6)[:,3:].copy()
    for i,chart in enumerate(state.q[:owner._shell_count].reshape(-1,6)):
        spins[i] = _exp_terms(chart[3:])[1]@spins[i]
    norm2 = metric.tangent_norm_squared(poses,translations,spins)
    if not np.isfinite(norm2) or norm2 <= 0.:
        raise AssessmentRefusal('unmeasured_physical_tangent')
    return poses,float(np.sqrt(norm2))


def assess_endpoint(raw,packet):
    """Fresh private evaluation; no trial state or material history is published."""
    owner = ss.create_analysis(definition=case.definition())
    before = None
    try:
        restored = owner.restore(raw)
        before = deepcopy(case._plain(owner.recover()))
        if (restored.status != 'accepted_prefix' or before != packet['observation']['state'] or
            owner.checkpoint() != raw):
            raise ValueError('fresh restore differs from saved complete public endpoint')
        if not case.material_activity(owner.recover())['elastic_fold_interpretation_eligible']:
            raise AssessmentRefusal('material_history_update_or_boundary_not_elastic')
        driver = owner._finite_driver
        if (driver._external is not None or driver._programme is not None or
            driver._adaptive_state is not None or owner._definition.shell.j2_curve is not None or
            owner._definition.shell_reference_policy != 'flat'):
            raise AssessmentRefusal('unsupported_loaded_material_or_chart_domain')
        preflight = owner.preflight()
        if preflight['identity'] != case.VERIFIED_FORCE_PREFIX_MODEL:
            raise AssessmentRefusal('unsupported_model_identity')
        order = tuple(preflight['arc_metric_node_order'])
        metric = case.metric_for(order,case.Specimen()); require_metric_owner(owner,metric)
        parent = driver._state
        state = driver._increment_origin(parent)
        n = owner._K.shape[0]; size = n+len(state.multipliers)
        if size > MAX_MIXED_ROWS:
            raise AssessmentRefusal('mixed_row_bound')
        shell_origins,beam_origins = driver._shell_origins,driver._beam_origins
        full = driver._load(case.VERIFIED_FORCE_PREFIX_ACTIONS)
        applied = np.asarray(owner.recover().applied_dead_load)
        factor = packet['observation']['proportional_factor']
        amplitude = owner.recover().amplitude
        if not np.array_equal(applied,.5*full+amplitude*full) or factor != .5+amplitude:
            raise ValueError('actual accepted load is not the frozen proportional law')
        require_director_chart(state.q[:owner._shell_count])
        evaluation = driver._assemble(state,0.,shell_origins,beam_origins,applied_load=applied)
        if (not np.isfinite(evaluation[0]).all() or not np.isfinite([evaluation[2],evaluation[3]]).all() or
            evaluation[2] > EQUILIBRIUM_LIMIT or evaluation[3] > COMPATIBILITY_LIMIT):
            raise AssessmentRefusal('restored_original_equilibrium_or_compatibility',
                                    dict(equilibrium=evaluation[2],constraint=evaluation[3]))
        if any(v != 0. for row in _material_observables(evaluation[5]) for v in row):
            raise AssessmentRefusal('trial_material_update_not_elastic')
        energy = float(np.max((abs(owner._K)@owner._coordinate_scale)*owner._coordinate_scale))
        if not np.isfinite(energy) or energy <= 0.:
            raise AssessmentRefusal('unresolved_reference_work_scale')
        columns = np.r_[owner._coordinate_scale,energy/owner._constraint_scale]
        rows = columns/energy
        free = np.r_[owner._free,np.arange(n,size)]
        nodal = owner._shell_count+6*len(owner._beam_ids)
        physical = np.array([d for d in range(nodal) if d%6 < 3],dtype=int)
        tip = 6*owner._shell_index[18]+2
        direction = np.r_[full,np.zeros(size-n)]
        answer = supported_branch_tangent(evaluation[1],direction,free,tip_dof=tip,
            physical_translation_dofs=physical,length=metric.length,column_scales=columns,
            row_scales=rows,constraint_start=n,orientation=1)
        tangent = answer.pop('tangent')
        poses,norm = _poses_and_directions(owner,state,tangent,metric)
        # Actual update columns: additive shell, spatial-left beam/cell,
        # retained resultants and multipliers. Hold the actual load fixed for
        # the mechanical derivative; its analytic unit-factor column is -p.
        expected_derivative = evaluation[1]@tangent
        eq_free = free[free < n]
        differences,derivatives = [],[]
        for h in FD_STEPS:
            finite = []
            for sign in (-1.,1.):
                moved = driver._advance(state,tangent,sign*h)
                require_director_chart(moved.q[:owner._shell_count])
                moved_poses = np.concatenate((physical_frames(moved.q[:owner._shell_count],moved.shell_anchors),moved.frames))
                displacements = (moved.q[:nodal]-state.q[:nodal]).reshape(-1,6)[:,:3]
                metric.evaluate(displacements,moved_poses,poses)
                trial = driver._assemble(moved,0.,shell_origins,beam_origins,
                    applied_load=applied)
                if any(v != 0. for row in _material_observables(trial[5]) for v in row):
                    raise AssessmentRefusal('directional_material_update_not_elastic')
                finite.append(np.array(trial[0],copy=True))
            observed = (finite[1]-finite[0])/(2*h)
            difference = observed-expected_derivative
            witness = directional_error_witness(difference,rows,eq_free,n,direction)
            if deepcopy(case._plain(owner.recover())) != before or owner.checkpoint() != raw:
                raise RuntimeError('directional assessment changed accepted public state/checkpoint')
            differences.append(dict(step_in_tip_over_length=h,
                original_derivative_difference=difference.tolist(),
                original_observed_derivative=observed.tolist(),
                **witness,
                registered_elastic_fd_limit_met=(witness['equilibrium_relative_full_load_error'] <= ELASTIC_FD_LIMIT and
                                                witness['compatibility_reference_ratio'] <= ELASTIC_FD_LIMIT),
                material_update_nonzero=False,full_public_state_unchanged=True,checkpoint_bytes_unchanged=True))
            derivatives.append(observed)
        plateau = derivatives[0]-derivatives[1]
        plateau_witness = directional_error_witness(plateau,rows,eq_free,n,direction)
        fd_pass = all(row['registered_elastic_fd_limit_met'] for row in differences) and (
            plateau_witness['equilibrium_relative_full_load_error'] <= ELASTIC_FD_LIMIT and
            plateau_witness['compatibility_reference_ratio'] <= ELASTIC_FD_LIMIT)
        answer.update(status='SUPPORTED_TANGENT_EVIDENCE',generation=restored.generation,
            proportional_factor=factor,tip_z_m=packet['observation']['tip_z_m'],
            tip_dof=tip,physical_tip_orientation=1,reference_length_m=metric.length,
            reference_work_nm=energy,free_rows=free.tolist(),constraint_start=n,
            native_unknowns_retained=True,tangent_per_tip_over_length=tangent.tolist(),
            unknown_reference_scales=columns.tolist(),physical_metric_tangent_norm=norm,
            original_endpoint_residual=np.asarray(evaluation[0]).tolist(),
            endpoint_equilibrium_ratio=float(evaluation[2]),endpoint_constraint_max=float(evaluation[3]),
            material_history_and_trial_update_nonzero=False,
            yield_branch_and_positive_margin_evidence='UNESTABLISHED',
            elastic_branch_proved=False,
            directional_difference=dict(actual_update='_advance',checks=differences,
                load_held_fixed=True,original_expected_mechanical_derivative=expected_derivative.tolist(),
                original_analytic_unit_factor_load_derivative=(-direction).tolist(),
                original_affine_path_derivative=(expected_derivative-direction*answer['load_tangent']).tolist(),
                original_plateau_difference=plateau.tolist(),plateau=plateau_witness,
                registered_elastic_fd_limit=ELASTIC_FD_LIMIT,registered_pair_and_plateau_met=fd_pass,
                complete_jacobian_proved=False),
            full_public_recovery_unchanged=True,checkpoint_bytes_unchanged=True,
            derivative_units=dict(d_factor_d_tip_over_length='dimensionless',d_factor_d_tip='1/m'))
        if not answer['directional_difference']['registered_pair_and_plateau_met']:
            answer['status'] = 'INCONCLUSIVE_DIRECTIONAL_CONSISTENCY'
        return answer
    finally:
        try:
            if before is not None and (deepcopy(case._plain(owner.recover())) != before or owner.checkpoint() != raw):
                raise RuntimeError('assessment changed complete accepted public recovery/checkpoint')
        finally:
            owner.close()


def assess_saved_prefixes(folder):
    """Only three bound endpoints, three sparse borders and twelve FD evaluations."""
    records = []
    for generation in (13,14,15):
        raw,packet = load_saved(folder,generation)
        try:
            row = assess_endpoint(raw,packet)
        except (AssessmentRefusal,ChordDomainError,BL5TrialDomainError,_RotationStepTooLarge) as error:
            row = dict(status='INCONCLUSIVE',generation=generation,error_type=type(error).__name__,
                       reason=str(error),details=case._plain(getattr(error,'details',{})))
        repeated_raw,repeated_packet = load_saved(folder,generation)
        if repeated_raw != raw or repeated_packet != packet:
            raise ValueError('saved input changed during assessment')
        records.append(row)
    return dict(status='BOUNDED_DEVELOPMENT_ASSESSMENT',records=records,
        signed_tangent_assessment=supported_sign_assessment(records),scientific_acceptance=False,
        G4D_physical_acceptance='OPEN',same_law_external_and_refinement='OPEN',F03='OPEN',
        full_jacobian_and_forward_slope_accuracy='OPEN',new_execution_authority_required=True)


if __name__ == '__main__':
    raise SystemExit('SOURCE_ONLY_UNRUN: separate numerical assessment authority is required')
