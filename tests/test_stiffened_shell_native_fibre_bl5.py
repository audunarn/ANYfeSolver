"""Public elastic native BL5 workflow and bounded conservative controls."""
from dataclasses import replace
from hashlib import sha256
import numpy as np
import pickle
import pytest
from scipy.linalg import eig
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver import _stiffened_shell_ge_point_buckling as buckling
from examples.stiffened_shell_native_fibre_bl5 import definition, workflow
from test_stiffened_shell_ge_line_buckling import line_oracle, direct_line_ad
from test_stiffened_shell_ge_mixed_buckling import dense_mixed_oracle
from anyfesolver._ge_beam3_pose_joint import _exp_terms
from anyfesolver._ge_beam3_retained_generalized_state import Mechanical
from anyfesolver._stiffened_shell_ge_finite_experimental import _position_pair

ACTIONS = (ss.NodalAction('beam',105,(-1000.,0.,100.,0.,0.,0.)),)


def native_line_oracle(owner,packet):
    if not owner._definition.shell.s3_elements:
        h,km=line_oracle(owner,packet);return h,km,h-km
    h,km,g=dense_mixed_oracle(owner,packet)
    snap=packet.snapshot;driver=owner._finite_driver
    u=snap.q[owner._shell_count:owner._shell_count+driver._beam_nodal].reshape(-1,6)[:,:3]
    positions,low=_position_pair(driver._reference_positions,u)
    mechanical=Mechanical(positions,low,snap.frames,snap.cells,snap.resultants)
    for sample,(rows,basis) in zip(owner._line_samples,owner._line_terms,strict=True):
        op,slots,weight,_,_=sample;rows=rows+6*len(owner._connections)
        anchors=None if snap.shell_anchors is None else snap.shell_anchors[slots[:12:6]//6]
        local=direct_line_ad(op,snap.q[slots[:12]].reshape(2,6),anchors,mechanical,basis@snap.multipliers[rows])
        cross=local.gap_jacobian.T@basis
        material=np.block([[np.zeros((21,21)),cross],[cross.T,np.zeros((len(rows),len(rows)))]])
        geometric=np.zeros_like(material);geometric[:21,:21]=local.hessian
        extended=np.r_[slots,packet.primal_count+rows]
        km[np.ix_(extended,extended)] += weight*material
        g[np.ix_(extended,extended)] += weight*geometric
        h[np.ix_(extended,extended)] += weight*(material+geometric)
    return h,km,g


@pytest.fixture(scope='module', params=(('biaxial','fibre','p0'),
    ('biaxial','mixed','continuous_p1_coarse'),('axial','fibre','p0')))
def accepted(request):
    model = definition(*request.param)
    owner = ss.create_analysis(definition=model)
    result = owner.solve(ACTIONS, (.5,1.), max_iterations=24, max_backtracks=12)
    assert result.status == 'completed'
    yield owner
    owner.close()


def test_public_recovery_fresh_restart_continue_and_spectrum(accepted):
    owner = accepted; raw = owner.checkpoint(); result = owner.recover()
    assert owner.capabilities['analysis'][-1] == ('experimental_elastic_native_fibre_mixed_shell_bl5'
        if owner._bl5_native_mixed_shell else 'experimental_elastic_native_fibre_bl5')
    assert owner.native_line_buckling_capabilities['definition_admitted']
    assert not owner.line_buckling_capabilities['definition_admitted']
    assert result.beam_fibre_material and result.beam_fibre_energy and result.beam_shell_line_work
    assert all(not record.strain.flags.writeable for record in result.beam_fibre_material)
    fresh = ss.create_analysis(definition=owner._definition)
    try:
        replay = fresh.restore(raw)
        assert fresh.checkpoint() == raw and replay.beam_shell_line_work and replay.beam_fibre_material
        continued = fresh.solve(ACTIONS, (1.2,), max_iterations=24, max_backtracks=12)
        assert continued.status == 'completed' and continued.free_equilibrium_ratio <= 1e-8
        assert continued.constraint_max <= 1e-9
        before = fresh.checkpoint()
        spectrum = fresh.buckling(ss.GENativeLineBucklingRequest(continued.generation,count=2))
        assert type(spectrum) is ss.GENativeLineBucklingResult and spectrum.status == 'completed', spectrum.failure_reason
        assert spectrum.policy == buckling._native_line_policy(fresh)
        assert spectrum.resultant_modes and spectrum.cell_rotation_modes and spectrum.line_coefficient_modes
        assert max(spectrum.equation_residuals) <= 1e-8 and max(spectrum.constraint_residuals) <= 1e-8
        assert fresh.checkpoint() == before
    finally: fresh.close()
    assert owner.checkpoint() == raw


def test_full42_and_direct_bl5_scalar_curvature_oracle(accepted):
    owner = accepted; raw = owner.checkpoint()
    packet, guard = buckling.capture(owner,owner._generation,lines=True,mixed=True,native_lines=True)
    h, km, g = native_line_oracle(owner,packet); free = list(packet.free)
    assert packet.primal_count == owner._shell_count+6*len(owner._beam_ids)+24*len(owner._beam_parts)
    assert packet.line_stations and any(np.linalg.norm(station.basis@packet.snapshot.multipliers[
        np.array(station.rows)+6*len(owner._connections)]) > 0 for station in packet.line_stations)
    np.testing.assert_allclose(packet.hessian.toarray(),h,rtol=1e-11,atol=2e-6)
    np.testing.assert_allclose(packet.material.toarray(),km,rtol=1e-11,atol=2e-6)
    driver=owner._finite_driver; state=driver._state; n=packet.primal_count
    direction=np.sin(np.arange(packet.hessian.shape[0])+1.)
    direction[:n] *= owner._coordinate_scale; direction[n:] *= owner._constraint_scale
    direction[owner._fixed]=0.
    caches={name:getattr(driver,name) for name in ('_line_trial_recovery','_point_trial_recovery',
        '_distributing_trial_recovery','_generalized_trial_recovery')}
    def covector(amount):
        trial=driver._advance(state,direction,amount)
        residual=driver._assemble(trial,driver._load_factor,pickle.loads(packet.snapshot.shell_origins),
            pickle.loads(packet.snapshot.beam_origins),applied_load=packet.snapshot.applied_dead_load)[0].copy()
        for node,row in enumerate(trial.q[:owner._shell_count].reshape(-1,6)):
            s=slice(6*node+3,6*node+6);residual[s]=_exp_terms(row[3:])[1].T@residual[s]
        starts=[owner._shell_count+6*node+3 for node in range(len(owner._beam_ids))]
        retained=owner._shell_count+6*len(owner._beam_ids)
        starts += [retained+24*i+3*c for i in range(len(owner._beam_parts)) for c in range(2)]
        for start in starts:
            s=slice(start,start+3);residual[s]=_exp_terms(amount*direction[s])[1].T@residual[s]
        return residual
    try:
        finite=[(covector(step)-covector(-step))/(2*step) for step in (3e-6,1e-6,3e-7)]
    finally:
        for name,value in caches.items():setattr(driver,name,value)
    exact=packet.hessian@direction;row_scale=np.r_[owner._coordinate_scale,owner._constraint_scale]
    errors=[np.max(abs((value-exact)*row_scale))/max(1.,np.max(abs(exact*row_scale))) for value in finite]
    assert max(errors) <= 1e-6,errors
    _,_,scale = buckling._scaled_pencil(owner,packet); congruence = np.outer(scale,scale)
    values = eig(km[np.ix_(free,free)]*congruence,-g[np.ix_(free,free)]*congruence,right=False)
    expected = [v.real for v in values if np.isfinite(v) and v.real > 0 and abs(v.imag) <= 1e-8*abs(v.real)]
    spectrum = owner.buckling(ss.GENativeLineBucklingRequest(owner._generation,count=2))
    assert spectrum.status == 'completed',spectrum.failure_reason
    assert all(min(abs(f-v)/max(abs(f),abs(v)) for v in expected) <= 1e-7 for f in spectrum.prestress_multipliers)
    guard(); assert owner.checkpoint() == raw


def test_public_refusals_atomic_corruption_cancel_mutation(accepted):
    owner = accepted; raw = owner.checkpoint()
    for kind in (ss.GEPointBucklingRequest,ss.GELineBucklingRequest,ss.GEMixedBucklingRequest):
        with pytest.raises(ValueError): owner.buckling(kind(owner._generation))
    with pytest.raises(ValueError): owner.buckling(ss.GENativeLineBucklingRequest(owner._generation+1))
    token = CancellationToken(); token.cancel('native line spectrum')
    cancelled = owner.buckling(ss.GENativeLineBucklingRequest(owner._generation),cancellation_token=token)
    assert cancelled.status == 'cancelled' and not cancelled.full_coordinates
    with pytest.raises(SolveCancelled): owner.solve(ACTIONS,(1.2,),cancellation_token=token)
    fresh = ss.create_analysis(definition=owner._definition)
    try:
        with pytest.raises(ValueError): fresh.restore(raw[:-1]+bytes((raw[-1]^1,)))
        assert fresh._finite_driver is None and fresh._generation == 0
    finally: fresh.close()
    packet,guard = buckling.capture(owner,owner._generation,lines=True,mixed=True,native_lines=True)
    saved = owner._line_terms; rows,basis = saved[0]; changed = basis.copy(); changed[0,0] += .01
    owner._line_terms = ((rows,changed),)+saved[1:]
    try:
        with pytest.raises(ValueError): guard()
    finally: owner._line_terms = saved
    guard()
    original = packet.material; changed = original.copy(); changed.data[0] += 1.
    object.__setattr__(packet,'material',changed)
    try:
        with pytest.raises(ValueError,match='immutable accepted buckling pencil'): guard()
    finally: object.__setattr__(packet,'material',original)
    guard(); assert owner.checkpoint() == raw


def test_true_yield_refusal_and_ownership_error_not_domain_or_retry(monkeypatch):
    owner = ss.create_analysis(definition=definition(yield_stress=100.))
    try:
        prefix = owner.solve(ACTIONS,(1e-6,),max_iterations=24,max_backtracks=12)
        assert prefix.status == 'completed'; raw = owner.checkpoint()
        with pytest.raises(ss.BL5NativeElasticDomainError) as raised:
            owner.solve(ACTIONS,(1.,),max_iterations=24,max_backtracks=12)
        assert raised.value.element_id and 'plastic history' in raised.value.cause
        assert owner.checkpoint() == raw
        calls = []
        def invariant(*args):
            calls.append(1); raise ValueError('physical-fibre station cell identity mismatch')
        monkeypatch.setattr(buckling,'_elastic_fibre',invariant)
        with pytest.raises(ValueError) as invariant_error:
            owner.solve(ACTIONS,(2e-6,),max_iterations=24,max_backtracks=12)
        assert type(invariant_error.value) is ValueError and len(calls) == 1
        assert owner.checkpoint() == raw
    finally: owner.close()


def test_explicit_model_policy_and_incompatible_snapshot(accepted):
    model = accepted._definition
    with pytest.raises(ValueError): ss.create_analysis(definition=replace(model,bl5_material_policy='elastic'))
    with pytest.raises(ValueError): ss.create_analysis(definition=replace(model,lines=()))
    other = ss.create_analysis(definition=replace(model,shell=replace(model.shell,young_pa=model.shell.young_pa*1.01)))
    try:
        assert other.preflight()['identity'] != accepted.preflight()['identity']
        with pytest.raises(ValueError): other.restore(accepted.checkpoint())
        assert other._finite_driver is None and other._generation == 0
    finally: other.close()


def test_headless_public_example():
    result = workflow(count=1)
    assert result['status'] == 'completed',result['failure_reason']
    assert result['fresh_restore_bytes_equal'] and result['continued_bytes_equal']
