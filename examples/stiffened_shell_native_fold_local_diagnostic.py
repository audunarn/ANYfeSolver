"""Passive single-owner Q4 diagnostic at a frozen saved endpoint.

This adds no execution authority and never validates a fold or repairs physics.
"""
from copy import deepcopy
from math import fsum
from threading import get_ident
from types import FunctionType

import numpy as np
from anyfesolver import stiffened_shell as ss
from anyfesolver import _stiffened_shell_ge_finite_experimental as finite
from anyfesolver import _stiffened_shell_q4_algorithmic_pullback as q4
from anyfesolver import _ge_beam3_variational_shell as shell_map
from anyfesolver import _ge_beam3_g3c_so3_numerics as stable_log
from anyfesolver.control import cancellation_safe_point
from examples import stiffened_shell_native_capacity as case
from examples import stiffened_shell_native_fold_assessment as fold


Q4_COUNT = 16
SHELL_ROWS = 162
MIXED_ROWS = 552
BASELINE_NUMERICS = 'Q4_LOCAL_ORIGINAL_EXP_LOG_CHART_V2'
LOG_ONLY_CANDIDATE = 'PRIVATE_Q4_SAVED_G13_STABLE_PRINCIPAL_LOG_ONLY_V1'
REGISTERED_PAIR = 'SAVED_G13_REGISTERED_1E5_1E6_V1'
INTERMEDIATE_PAIR = 'SAVED_G13_PROSPECTIVE_3E6_1E6_V1'
_active_capture = None


def _fd_steps(pair):
    """Fixed private assessment choices; selecting one grants no run authority."""
    if type(pair) is not str:
        raise ValueError('explicit known private difference-pair identity required')
    if pair == REGISTERED_PAIR:
        return fold.FD_STEPS
    if pair == INTERMEDIATE_PAIR:
        return (3e-6, 1e-6)
    raise ValueError('explicit known private difference-pair identity required')


def _copy_array(value,shape):
    result = np.array(value,dtype=float,copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('finite bounded actual Q4 capture required')
    return result


class Q4LocalCapture:
    """Three narrowly scoped aliases; original evaluations and returns are used.

    Only a single synchronous owner in an isolated process is supported. No
    profile/trace hook, evaluator replay, cache replacement or product edit.
    """
    def __init__(self,owner,*,numerical_candidate=BASELINE_NUMERICS):
        if numerical_candidate not in (BASELINE_NUMERICS,LOG_ONLY_CANDIDATE):
            raise ValueError('explicit known private Q4 numerical identity required')
        self.numerical_candidate = numerical_candidate
        self.owner = owner
        self.parts = {id(part[1]):(part[0],part[2]) for part in owner._shell_parts}
        self.origins = {id(part[1]):origin for part,origin in
                        zip(owner._shell_parts,owner._finite_driver._shell_origins)}
        if len(self.parts) != Q4_COUNT or any(len(slots) != 24 for _,slots in self.parts.values()):
            raise fold.AssessmentRefusal('exact_sixteen_Q4_capture_domain_required')
        if len(self.origins) != Q4_COUNT:
            raise RuntimeError('complete same-owner accepted Q4 predecessors required')
        self.records = []
        self.current = None
        self.entered = False
        self.in_deformation = False

    def __enter__(self):
        global _active_capture
        if _active_capture is not None:
            raise RuntimeError('nested or concurrent Q4 capture is unsupported')
        if finite.q4_algorithmic_response is not q4.response:
            raise RuntimeError('Q4 driver alias already replaced or incompatible')
        self.original = (finite.q4_algorithmic_response,q4.deformation,q4.spatial_response)
        self.selected_deformation = self.original[1]
        if self.numerical_candidate == LOG_ONLY_CANDIDATE:
            original = self.original[1]
            if (original is not shell_map.deformation or not isinstance(original,FunctionType) or
                original.__globals__.get('so3_log') is not shell_map.so3_log or
                original.__globals__.get('so3_exp') is not shell_map.so3_exp):
                raise RuntimeError('known unchanged Q4 deformation kernel required for Log-only candidate')
            # Reuse the exact existing code/closure/defaults. Only this private
            # function's global context selects Log; no product global changes.
            context = dict(original.__globals__)
            context['so3_log'] = self._candidate_log
            selected = FunctionType(original.__code__,context,original.__name__,
                                    original.__defaults__,original.__closure__)
            selected.__kwdefaults__ = None if original.__kwdefaults__ is None else dict(original.__kwdefaults__)
            self.selected_deformation = selected
        self.thread = get_ident()
        self.wrappers = (self._response,self._deformation,self._spatial)
        _active_capture = self
        finite.q4_algorithmic_response,q4.deformation,q4.spatial_response = self.wrappers
        self.entered = True
        return self

    def __exit__(self,*exception):
        global _active_capture
        actual = (finite.q4_algorithmic_response,q4.deformation,q4.spatial_response)
        changed = any(a is not b for a,b in zip(actual,self.wrappers))
        finite.q4_algorithmic_response,q4.deformation,q4.spatial_response = self.original
        _active_capture = None
        self.entered = False
        if changed:
            raise RuntimeError('Q4 capture alias changed during owned observation')
        return False

    def _require_call(self):
        if not self.entered or get_ident() != self.thread or _active_capture is not self:
            raise RuntimeError('foreign or asynchronous Q4 capture call')

    def _response(self,model,element,displacement,origin,layers,**options):
        self._require_call()
        if model is not self.owner._shell_model or id(element) not in self.parts or self.current is not None:
            raise RuntimeError('foreign or reentrant Q4 response call')
        eid,slots = self.parts[id(element)]
        if origin is not self.origins[id(element)]:
            raise RuntimeError('Q4 accepted predecessor identity changed')
        if any(row['element_id'] == eid for row in self.records):
            raise RuntimeError('repeated Q4 element in one actual assembly')
        record = dict(element_id=eid,slots=np.array(slots,dtype=int,copy=True),
            displacement=_copy_array(displacement,(24,)),layers=layers,
            stable_rotation=options.get('stable_rotation',False),
            material_capture_requested=options.get('capture_material_trial',False),
            same_accepted_predecessor_identity=True,
            predecessor_capture='identity witness only; constitutive branch and margin unestablished')
        record.update(numerical_candidate=self.numerical_candidate,baseline_numerics=BASELINE_NUMERICS,
            candidate_log_calls=0,original_exp_and_spatial_chart=True,
            mechanical_operator_compatibility_proved=False)
        self.current = record
        try:
            result = self.original[0](model,element,displacement,origin,layers,**options)
            if 'local_coordinates' not in record or 'spatial_residual' not in record:
                raise RuntimeError('actual Q4 observation seam was not called')
            record['returned_residual'] = _copy_array(result[0],(24,))
            record['returned_tangent'] = _copy_array(result[1],(24,24))
            record['candidate_present'] = result[2] is not None
            self.records.append(record)
            return result  # Exact tuple/object identity, including candidate and payload.
        finally:
            self.current = None

    def _deformation(self,*args,**options):
        self._require_call()
        if self.current is None or 'local_coordinates' in self.current:
            raise RuntimeError('foreign or repeated Q4 deformation call')
        if self.numerical_candidate == LOG_ONLY_CANDIDATE and options.get('stable_rotation',False) is not False:
            raise RuntimeError('Log-only candidate requires original Exp and chart numerical route')
        self.in_deformation = True
        try:
            result = self.selected_deformation(*args,**options)
        finally:
            self.in_deformation = False
        if self.numerical_candidate == LOG_ONLY_CANDIDATE and self.current['candidate_log_calls'] != 4:
            raise RuntimeError('exact four actual Q4 principal Log calls required')
        self.current.update(local_coordinates=_copy_array(result[0],(24,)),
            deformation_differential=_copy_array(result[1],(24,24)),
            deformation_second=_copy_array(result[2],(24,24,24)),
            reference=_copy_array(args[0],(4,3)),
            stable_translation=options.get('stable_translation',False),
            rotation_anchors=None if options.get('rotation_anchors') is None else
                _copy_array(options['rotation_anchors'],(4,3,3)))
        return result

    def _candidate_log(self,matrix):
        self._require_call()
        if self.current is None or not self.in_deformation:
            raise RuntimeError('Log-only candidate outside actual owned Q4 deformation')
        self.current['candidate_log_calls'] += 1
        if self.current['candidate_log_calls'] > 4:
            raise RuntimeError('bounded Q4 candidate Log call count exceeded')
        return stable_log.so3_log(matrix)  # Existing value/first/second helper only.

    def _spatial(self,displacement,force,tangent,differential,second,**options):
        self._require_call()
        if self.current is None or 'local_force' in self.current:
            raise RuntimeError('foreign or repeated Q4 spatial pullback call')
        if self.numerical_candidate == LOG_ONLY_CANDIDATE and options.get('stable_rotation',False) is not False:
            raise RuntimeError('Log-only candidate requires the original spatial chart route')
        local_force = _copy_array(force,(24,))
        local_tangent = _copy_array(tangent,(24,24))
        derivative = _copy_array(differential,(24,24))
        result = self.original[2](displacement,force,tangent,differential,second,**options)
        self.current.update(local_force=local_force,local_tangent=local_tangent,
            spatial_residual=_copy_array(result[0],(24,)),spatial_tangent=_copy_array(result[1],(24,24)),
            derived_work_covector=_copy_array(derivative.T@local_force,(24,)),
            derived_work_covector_provenance='diagnostic D.T*f transform; not observed private intermediate')
        return result

    def assemble(self,state,shell_origins,beam_origins,applied,token=None):
        self._require_call()
        cancellation_safe_point(token,'before bounded Q4 diagnostic assembly')
        self.records = []
        result = self.owner._finite_driver._assemble(state,0.,shell_origins,beam_origins,applied_load=applied)
        if len(self.records) != Q4_COUNT:
            raise RuntimeError('actual bounded assembly did not visit all sixteen Q4 elements')
        captured = deepcopy(self.records)
        cancellation_safe_point(token,'after bounded Q4 diagnostic assembly')
        return result,captured


def _sparse_owned(matrix):
    value = matrix.tocsc(copy=True)
    if not np.isfinite(value.data).all():
        raise ValueError('finite actual mixed tangent required')
    return dict(format='csc',shape=list(value.shape),data=value.data.copy(),
                indices=value.indices.copy(),indptr=value.indptr.copy())


def _publish_evaluation(result,evaluation,on_evaluation):
    """Retain an owned witness and deliver a different owned plain payload."""
    retained = deepcopy(case._plain(evaluation))
    result['evaluations'].append(retained)
    if on_evaluation is not None:
        on_evaluation(deepcopy(retained))


def _normal_or_zero(value):
    value = float(value)
    if not np.isfinite(value) or (value != 0. and abs(value) < np.finfo(float).tiny):
        raise fold.AssessmentRefusal('unsupported_nonfinite_or_subnormal_Q4_bound_domain')
    return value


def _positive_up(operation,a,b):
    """One positive operation rounded outward; exact zero stays zero."""
    a,b = _normal_or_zero(a),_normal_or_zero(b)
    if a < 0. or b < 0. or (operation == 'divide' and b == 0.):
        raise ValueError('positive finite bound operation required')
    if operation == 'add': value = a+b
    elif operation == 'multiply': value = a*b
    elif operation == 'divide': value = a/b
    else: raise ValueError('unknown bounded positive operation')
    value = _normal_or_zero(value)
    if value == 0.:
        exact_zero = (a == 0. and b == 0.) if operation == 'add' else (a == 0. or (operation == 'multiply' and b == 0.))
        if not exact_zero:
            raise fold.AssessmentRefusal('unsupported_underflow_Q4_bound_domain')
        return 0.
    return _normal_or_zero(np.nextafter(value,np.inf))


def _sum_rows(records,size):
    terms = [[] for _ in range(size)]
    for record in records:
        for slot,value in zip(record['slots'],record['returned_residual']):
            if not 0 <= slot < SHELL_ROWS:
                raise ValueError('Q4 contribution must occupy declared shell rows')
            terms[slot].append(_normal_or_zero(value))
    try:
        values = np.array([fsum(row) for row in terms])
        magnitudes = [fsum(abs(value) for value in row) for row in terms]
    except OverflowError as error:
        raise fold.AssessmentRefusal('unsupported_overflow_Q4_sum_domain') from error
    for value in values: _normal_or_zero(value)
    # Windows binary64 fsum final-rounding model, not an unrestricted exact sum.
    unit = np.finfo(float).eps/2
    absolute = np.array([_positive_up('divide',value,1-unit) for value in magnitudes])
    counts = np.array([len(row) for row in terms])
    return values,absolute,counts


def q4_difference_witness(baseline,minus,plus,tangent,h,rows,free,n,direction,*,
                          difference_pair=REGISTERED_PAIR):
    """Q4 returned-force decomposition; element arithmetic remains unbounded.

    Standard gamma_n summation bounds assume finite normal binary64 arithmetic
    and no overflow/underflow. These bound only sums of already returned floats.
    They do not bound local constitutive, deformation or pullback evaluation.
    """
    size = len(tangent)
    if h not in _fd_steps(difference_pair):
        raise ValueError('only the fixed registered step pair is permitted')
    qplus,aplus,cplus = _sum_rows(plus,size)
    qminus,aminus,cminus = _sum_rows(minus,size)
    expected = np.zeros(size)
    local = []
    byminus = {row['element_id']:row for row in minus}
    byplus = {row['element_id']:row for row in plus}
    for base in baseline:
        eid,slots = base['element_id'],base['slots']
        m,p = byminus[eid],byplus[eid]
        if not np.array_equal(slots,m['slots']) or not np.array_equal(slots,p['slots']):
            raise ValueError('Q4 local slots changed across actual evaluations')
        actual = (p['returned_residual']-m['returned_residual'])/(2*h)
        predicted = base['returned_tangent']@tangent[slots]
        np.add.at(expected,slots,predicted)
        local.append(dict(element_id=eid,slots=slots.copy(),
            original_observed_derivative=actual,original_expected_derivative=predicted,
            original_derivative_difference=actual-predicted,
            local_coordinate_derivative=(p['local_coordinates']-m['local_coordinates'])/(2*h),
            expected_local_coordinate_derivative=base['deformation_differential']@tangent[slots],
            local_force_derivative=(p['local_force']-m['local_force'])/(2*h)))
    observed = (qplus-qminus)/(2*h)
    difference = observed-expected
    unit = np.finfo(float).eps/2
    # Rounding of each sum and the subtraction/division, conditional on the
    # stated floating-point model; the returned floats are the inputs here.
    # n<=16 makes n*u and 1-n*u exactly representable; outward division gives
    # gamma_n. Expressing a two-operation envelope in its computed result
    # requires gamma_2/(1-gamma_2)=2*u/(1-4*u), not 2*u/(1-u).
    central_factor = _positive_up('divide',2*unit,1-4*unit)
    bound = np.zeros(size)
    for i in range(size):
        if cplus[i] > Q4_COUNT or cminus[i] > Q4_COUNT:
            raise ValueError('bounded Q4 contribution count exceeded')
        _normal_or_zero(qplus[i]-qminus[i]); _normal_or_zero(observed[i])
        gp = _positive_up('divide',int(cplus[i])*unit,1-int(cplus[i])*unit)
        gm = _positive_up('divide',int(cminus[i])*unit,1-int(cminus[i])*unit)
        numerator = _positive_up('add',_positive_up('multiply',gp,aplus[i]),
                                     _positive_up('multiply',gm,aminus[i]))
        summation = _positive_up('divide',numerator,2*h)
        central = _positive_up('multiply',central_factor,abs(observed[i]))
        bound[i] = _positive_up('add',summation,central)
    if not all(np.isfinite(v).all() for v in (observed,expected,difference,bound)):
        raise fold.AssessmentRefusal('nonfinite_Q4_difference_or_conditional_bound')
    eq_free = free[free < n]
    witness = fold.directional_error_witness(difference,rows,eq_free,n,direction)
    bound_witness = fold.directional_error_witness(bound,rows,eq_free,n,direction)
    return dict(status='INCONCLUSIVE_LOCAL_Q4_DIAGNOSTIC',local=local,
        Q4_only_original_observed_derivative=observed,Q4_only_original_expected_derivative=expected,
        Q4_only_original_difference=difference,Q4_only_witness=witness,
        Q4_plus_sum=qplus,Q4_minus_sum=qminus,
        Q4_plus_absolute_sum_upper=aplus,Q4_minus_absolute_sum_upper=aminus,
        Q4_plus_contribution_counts=cplus,Q4_minus_contribution_counts=cminus,
        Q4_only_conditional_assembly_difference_bound=bound,
        Q4_only_conditional_assembly_bound_witness=bound_witness,
        bound_assumptions='Windows IEEE754 binary64 half-even fsum final-rounding model; normal finite operations; no overflow/underflow; returned Q4 floats fixed',
        bound_vector_outward_positive=True,aggregate_bound_witness_is_interval_bound=False,
        expected_tangent_product_arithmetic_bound='UNESTABLISHED',
        unproved_element_arithmetic_bound=True,operator_defect_excluded=False,
        operator_defect_proved=False,criterion=fold.ELASTIC_FD_LIMIT)


def shell_translation_sampling_witness(minus_q,plus_q,tangent,h,jacobian,rows,free,n,direction,
                                       original_difference,*,difference_pair=REGISTERED_PAIR):
    """Detached first-order sampling diagnostic; never an acceptance correction.

    Only additive shell translations are measured. Native translations, rotation
    updates, finite-step curvature and internal arithmetic remain unallocated.
    """
    if h not in _fd_steps(difference_pair):
        raise ValueError('registered fixed difference step required')
    minus = _copy_array(minus_q,(SHELL_ROWS,))
    plus = _copy_array(plus_q,(SHELL_ROWS,))
    requested = _copy_array(tangent,(MIXED_ROWS,))
    scales = _copy_array(rows,(MIXED_ROWS,))
    load = _copy_array(direction,(MIXED_ROWS,))
    difference = _copy_array(original_difference,(MIXED_ROWS,))
    indices = np.array(free,copy=True)
    if (type(n) is not int or not 0 < n < MIXED_ROWS or indices.ndim != 1
            or indices.dtype.kind not in 'iu' or np.any(indices < 0)
            or np.any(indices >= MIXED_ROWS) or len(np.unique(indices)) != len(indices)):
        raise ValueError('bounded mechanical row selection required')
    if getattr(jacobian,'shape',None) != (MIXED_ROWS,MIXED_ROWS):
        raise ValueError('bounded baseline operator required')
    operator = jacobian.copy()
    values = operator.data if hasattr(operator,'tocsr') else operator
    if not np.isfinite(values).all():
        raise ValueError('finite baseline operator required')
    translation = np.arange(SHELL_ROWS)[np.arange(SHELL_ROWS)%6 < 3]
    represented = (plus[translation]-minus[translation])/(2*h)
    delta = np.zeros(MIXED_ROWS)
    delta[translation] = represented-requested[translation]
    contribution = _copy_array(operator@delta,(MIXED_ROWS,))
    remainder = difference-contribution
    mechanical = indices[indices<n]
    return dict(diagnostic_only=True,is_exact_error_bound=False,
        operator_defect_excluded=False,changes_registered_acceptance=False,
        original_failures_superseded=False,
        scope='additive shell translations only; first-order baseline-J contribution; not all input rounding or finite-step curvature',
        shell_translation_indices=translation.copy(),
        requested_translation_direction=requested[translation].copy(),
        represented_translation_direction=represented.copy(),
        translation_direction_sampling_difference=delta.copy(),
        first_order_baseline_J_contribution=contribution.copy(),
        first_order_represented_translation_remainder=remainder.copy(),
        original_witness=fold.directional_error_witness(difference,scales,mechanical,n,load),
        contribution_witness=fold.directional_error_witness(contribution,scales,mechanical,n,load),
        first_order_remainder_witness=fold.directional_error_witness(remainder,scales,mechanical,n,load))


def diagnose_saved_g13(folder,*,cancellation_token=None,on_evaluation=None,
                      numerical_candidate=BASELINE_NUMERICS,
                      difference_pair=REGISTERED_PAIR):
    """Historical g13 facade; original defaults and numerical path are preserved."""
    return diagnose_saved_endpoint(folder,generation=13,cancellation_token=cancellation_token,
        on_evaluation=on_evaluation,numerical_candidate=numerical_candidate,
        difference_pair=difference_pair)


def diagnose_saved_endpoint(folder,*,generation,cancellation_token=None,on_evaluation=None,
                            numerical_candidate=BASELINE_NUMERICS,
                            difference_pair=REGISTERED_PAIR):
    """Exactly one bound restore, one border, baseline+four actual assemblies."""
    if type(generation) is not int or generation not in (13,14,15):
        raise ValueError('exact frozen saved generation 13/14/15 required')
    if numerical_candidate not in (BASELINE_NUMERICS,LOG_ONLY_CANDIDATE):
        raise ValueError('explicit known private Q4 numerical identity required')
    steps = _fd_steps(difference_pair)
    if difference_pair == INTERMEDIATE_PAIR and numerical_candidate != LOG_ONLY_CANDIDATE:
        raise ValueError('prospective intermediate pair requires explicit Log-only candidate')
    raw,packet = fold.load_saved(folder,generation)
    cancellation_safe_point(cancellation_token,'before fresh saved-endpoint diagnostic owner')
    owner = ss.create_analysis(definition=case.definition())
    before = None
    observer_failure = [None]
    def deliver_evaluation(payload):
        try:
            on_evaluation(payload)
        except BaseException as error:
            observer_failure[0] = error
            raise
    result = dict(status='INCONCLUSIVE_LOCAL_Q4_DIAGNOSTIC',generation=generation,evaluations=[],checks=[],
        diagnostic_completion='PARTIAL',
        scientific_acceptance=False,simple_fold_proved=False,capacity_proved=False,
        yield_branch_margin='UNESTABLISHED',G4D='OPEN',same_law_external_and_refinement='OPEN',F03='OPEN',
        registered_relative_criterion=fold.ELASTIC_FD_LIMIT)
    result.update(numerical_candidate=numerical_candidate,baseline_numerics=BASELINE_NUMERICS,
        difference_pair=difference_pair,difference_steps=steps,
        numerical_provenance=dict(principal_log='original' if numerical_candidate == BASELINE_NUMERICS else
            '_ge_beam3_g3c_so3_numerics.so3_log value/first/second; private Q4 function context',
            Exp='original _ge_beam3_mixed_ad.so3_exp',spatial_chart='original _ge_beam3_pose_joint._exp_terms',
            candidate_is_public_or_owner_dispatch=False),
        checkpoint_preservation_proves_operator_compatibility=False,
        mechanical_operator_compatibility_proved=False)
    try:
        restored = owner.restore(raw)
        before = deepcopy(case._plain(owner.recover()))
        if restored.status != 'accepted_prefix' or before != packet['observation']['state'] or owner.checkpoint() != raw:
            raise ValueError(f'saved g{generation} restore differs from complete bound public state/bytes')
        driver = owner._finite_driver
        if (owner.preflight()['identity'] != case.VERIFIED_FORCE_PREFIX_MODEL or
            driver._external is not None or driver._programme is not None or driver._adaptive_state is not None or
            owner._definition.shell.j2_curve is not None or owner._definition.shell_reference_policy != 'flat' or
            owner._footprint_increment_chart or owner._shell_count != SHELL_ROWS or
            owner._definition.q4_pullback_policy != q4.POLICY):
            raise fold.AssessmentRefusal(f'unsupported_local_g{generation}_diagnostic_domain')
        if not case.material_activity(owner.recover())['elastic_fold_interpretation_eligible']:
            raise fold.AssessmentRefusal('active_saved_material_history')
        state = driver._increment_origin(driver._state)
        n = owner._K.shape[0]; size = n+len(state.multipliers)
        if size != MIXED_ROWS or n != 462 or len(state.multipliers) != 90 or len(owner._beam_ids) != 18:
            raise fold.AssessmentRefusal('exact_saved_mixed_row_count_required')
        origins = driver._shell_origins,driver._beam_origins
        full = driver._load(case.VERIFIED_FORCE_PREFIX_ACTIONS)
        applied = np.array(owner.recover().applied_dead_load,copy=True)
        amplitude = owner.recover().amplitude
        if (not np.array_equal(applied,.5*full+amplitude*full) or
            packet['observation']['proportional_factor'] != .5+amplitude):
            raise ValueError(f'actual g{generation} load differs from frozen proportional law')
        fold.require_director_chart(state.q[:SHELL_ROWS])
        metric = case.metric_for(tuple(owner.preflight()['arc_metric_node_order']),case.Specimen())
        fold.require_metric_owner(owner,metric)
        energy = float(np.max((abs(owner._K)@owner._coordinate_scale)*owner._coordinate_scale))
        if not np.isfinite(energy) or energy <= 0.:
            raise fold.AssessmentRefusal('unresolved_reference_work_scale')
        columns = np.r_[owner._coordinate_scale,energy/owner._constraint_scale]; rows = columns/energy
        free = np.r_[owner._free,np.arange(n,size)]
        direction = np.r_[full,np.zeros(size-n)]
        nodal = SHELL_ROWS+6*len(owner._beam_ids)
        physical = np.array([i for i in range(nodal) if i%6 < 3])
        with Q4LocalCapture(owner,numerical_candidate=numerical_candidate) as capture:
            def evaluate(label,trial):
                assembled,records = capture.assemble(trial,*origins,applied,cancellation_token)
                if not np.isfinite(assembled[0]).all() or any(v != 0. for row in fold._material_observables(assembled[5]) for v in row):
                    raise fold.AssessmentRefusal('nonfinite_or_active_trial_material')
                captured = dict(label=label,generation=generation,status='COMPLETED_EVALUATION',
                    diagnostic_completion='PARTIAL',accepted_checkpoint_sha256=fold.SAVED[generation][1],
                    numerical_candidate=numerical_candidate,baseline_numerics=BASELINE_NUMERICS,
                    mechanical_operator_compatibility_proved=False,
                    accepted_public_state=deepcopy(before),assembled_residual=np.array(assembled[0],copy=True),
                    Q4=records,actual_mixed_state=dict(q=trial.q.copy(),frames=trial.frames.copy(),
                        cells=trial.cells.copy(),resultants=trial.resultants.copy(),multipliers=trial.multipliers.copy(),
                        shell_anchors=None if trial.shell_anchors is None else trial.shell_anchors.copy()))
                if deepcopy(case._plain(owner.recover())) != before or owner.checkpoint() != raw:
                    raise RuntimeError('local diagnostic changed accepted public state/checkpoint')
                captured.update(complete_public_recovery_unchanged=True,checkpoint_bytes_unchanged=True)
                _publish_evaluation(result,captured,None if on_evaluation is None else deliver_evaluation)
                return assembled,records
            base,base_records = evaluate('baseline',state)
            if not np.isfinite([base[2],base[3]]).all() or base[2] > fold.EQUILIBRIUM_LIMIT or base[3] > fold.COMPATIBILITY_LIMIT:
                raise fold.AssessmentRefusal('baseline_original_equilibrium_or_compatibility')
            tangent = fold.supported_branch_tangent(base[1],direction,free,
                tip_dof=6*owner._shell_index[18]+2,physical_translation_dofs=physical,
                length=metric.length,column_scales=columns,row_scales=rows,constraint_start=n)
            actual_tangent = tangent.pop('tangent')
            poses,norm = fold._poses_and_directions(owner,state,actual_tangent,metric)
            baseline_jacobian = base[1].copy()  # own before signed evaluations invalidate scratch
            expected = baseline_jacobian@actual_tangent
            result.update(actual_mixed_tangent=_sparse_owned(base[1]),signed_branch_tangent=tangent,
                actual_mixed_direction=actual_tangent.copy(),full_unit_factor_load=direction.copy(),
                actual_applied_load=applied.copy(),reference_work_nm=energy,row_scales=rows.copy(),
                column_scales=columns.copy(),free_rows=free.copy(),constraint_start=n,
                physical_tangent_norm=norm,accepted_public_state=deepcopy(before))
            derivatives = []
            for h in steps:
                signed = []
                for sign in (-1,1):
                    moved = driver._advance(state,actual_tangent,sign*h)
                    fold.require_director_chart(moved.q[:SHELL_ROWS])
                    moved_poses = np.concatenate((fold.physical_frames(moved.q[:SHELL_ROWS],moved.shell_anchors),moved.frames))
                    metric.evaluate((moved.q[:nodal]-state.q[:nodal]).reshape(-1,6)[:,:3],moved_poses,poses)
                    signed.append((*evaluate(f'h={h:.0e},sign={sign:+d}',moved),
                                   np.array(moved.q[:SHELL_ROWS],copy=True)))
                observed = (signed[1][0][0]-signed[0][0][0])/(2*h)
                witness = q4_difference_witness(base_records,signed[0][1],signed[1][1],actual_tangent,h,rows,free,n,direction,
                    difference_pair=difference_pair)
                witness.update(step=h,assembled_original_derivative=observed,
                    assembled_original_expected_derivative=expected.copy(),
                    assembled_original_difference=observed-expected,
                    assembled_minus_Q4_reconstruction_remainder=(observed-expected)-witness['Q4_only_original_difference'],
                    reconstruction_remainder_provenance='assembled FD-minus-Jt difference less separately reconstructed Q4 FD-minus-elementwise-Jt difference; includes scatter/reconstruction and combined-J arithmetic; no non-Q4 attribution',
                    assembled_witness=fold.directional_error_witness(observed-expected,rows,free[free<n],n,direction))
                witness['shell_translation_sampling'] = shell_translation_sampling_witness(
                    signed[0][2],signed[1][2],actual_tangent,h,baseline_jacobian,rows,free,n,direction,
                    observed-expected,difference_pair=difference_pair)
                result['checks'].append(witness); derivatives.append(observed)
            result['plateau'] = dict(original_difference=derivatives[0]-derivatives[1],
                witness=fold.directional_error_witness(derivatives[0]-derivatives[1],rows,free[free<n],n,direction))
        result.update(complete_public_recovery_unchanged=True,checkpoint_bytes_unchanged=True,
            diagnostic_completion='COMPLETED_BOUNDED_DIAGNOSTIC',
            scratch_cache_lifecycle='ordinary evaluator; not restored or replaced',
            yield_branch_margin='UNESTABLISHED',scientific_acceptance=False,simple_fold_proved=False,
            capacity_proved=False,registered_relative_criterion=fold.ELASTIC_FD_LIMIT,
            complete_Jacobian_or_forward_slope_accuracy_proved=False,
            G4D='OPEN',same_law_external_and_refinement='OPEN',F03='OPEN')
    except (fold.AssessmentRefusal,fold.ChordDomainError,fold.BL5TrialDomainError,fold._RotationStepTooLarge) as error:
        if observer_failure[0] is error:
            raise  # Even a typed diagnostic exception from the writer is a caller failure.
        result.update(reason=str(error),error_type=type(error).__name__)
    finally:
        try:
            if before is not None and (deepcopy(case._plain(owner.recover())) != before or owner.checkpoint() != raw):
                raise RuntimeError('local diagnostic changed complete accepted public recovery/bytes')
            if before is not None:
                result.update(complete_public_recovery_unchanged=True,checkpoint_bytes_unchanged=True)
        finally:
            owner.close()
    repeated_raw,repeated_packet = fold.load_saved(folder,generation)
    if repeated_raw != raw or repeated_packet != packet:
        raise ValueError('bound saved-endpoint inputs changed during local diagnostic')
    return deepcopy(case._plain(result))


if __name__ == '__main__':
    raise SystemExit('SOURCE_ONLY_UNRUN: fresh local-diagnostic numerical authority is required')
