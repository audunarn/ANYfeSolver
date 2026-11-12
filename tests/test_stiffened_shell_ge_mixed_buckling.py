"""Supervised matching-point S3/physical-fibre conservative buckling checks."""
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
import pickle
import json
import numpy as np
import pytest
from scipy.linalg import eig
from anyfesolver import stiffened_shell as public
from anyfesolver import _stiffened_shell_ge_point_buckling as buckling
from anyfesolver._stiffened_shell_ge_finite_experimental import _position_pair
from anyfesolver._ge_beam3_variational_shell import deformation, response as shell_response
from anyfesolver._ge_beam3_pose_joint import _exp_terms
from anyfesolver._ge_beam3_mixed_ad import Jet2, constant_matrix, matmul, matvec, transpose, so3_exp, so3_log
from anyfesolver._ge_beam3_retained_fibre import RetainedFibreOperator
from anyfesolver.b3_ge import FlowCurve
from anyfesolver._ge_beam3_fibre_section import FibreHistory
from anyfesolver.control import CancellationToken
from test_stiffened_shell_ge_point_buckling import _MockInverse
from test_stiffened_shell_ge_experimental import _panel
from test_stiffened_shell_ge_physical_fibre_linear import _definition


def definition(shell='s3', beams='mixed'):
    model = _definition(mixed=beams=='mixed', triangles=shell=='s3')
    if beams == 'generalized': model = replace(model, beams=_panel().beams)
    if shell == 'mixed':
        model = replace(model, shell=replace(model.shell, q4_elements=((1,2,5,4),),
            s3_elements=((2,3,6),(2,6,5))))
    return model if shell == 's3' else replace(model,q4_pullback_policy=public.Q4_ALGORITHMIC_PULLBACK_V1)


@pytest.fixture(scope='module', params=(('q4','fibre'),('s3','generalized'),('s3','fibre'),('mixed','mixed')))
def accepted_owner(request):
    owner=public.create_analysis(definition=definition(*request.param))
    result=owner.solve((public.NodalAction('beam',105,(-1000.,0.,100.,0.,0.,0.)),),
        (.5,1.),max_iterations=24,max_backtracks=12)
    assert result.status=='completed'
    yield owner
    owner.close()


def test_full_retained_split_and_actual_s3_spatial_parity(accepted_owner):
    owner=accepted_owner; before=owner.checkpoint()
    packet,guard=buckling.capture(owner,owner._generation,mixed=True)
    h,km,g=dense_mixed_oracle(owner,packet)
    assert np.linalg.norm(packet.snapshot.multipliers)>0
    for actual,expected in ((packet.hessian,h),(packet.material,km),(packet.geometric,g)):
        np.testing.assert_allclose(actual.toarray(),expected,rtol=1e-11,atol=2e-6)
    free=list(packet.free)
    np.testing.assert_allclose((packet.material+packet.geometric)[free,:][:,free].toarray(),
        h[np.ix_(free,free)],rtol=1e-11,atol=2e-6)
    assert packet.primal_count==owner._shell_count+6*len(owner._beam_ids)+24*len(owner._beam_parts)
    from anyfesolver._stiffened_shell_s3_conservative_buckling import conservative_components
    for (_,element,slots,_),origin in zip(owner._shell_parts,pickle.loads(packet.snapshot.shell_origins),strict=True):
        if len(slots)!=18: continue
        anchors=None if packet.snapshot.shell_anchors is None else packet.snapshot.shell_anchors[slots[::6]//6]
        u=packet.snapshot.q[slots]
        g,hlocal,_,_,_=conservative_components(owner._shell_model,element,u,origin,3,
            rotation_anchors=anchors,stable_translation=owner._finite_driver._stable_s3)
        r,j,_=shell_response(owner._shell_model,element,u,origin,3,
            rotation_anchors=anchors,stable_translation=owner._finite_driver._stable_s3)
        p=np.eye(18); extra=np.zeros((18,18))
        for node,row in enumerate(u.reshape(-1,6)):
            _,a,da=_exp_terms(row[3:]); s=slice(6*node+3,6*node+6)
            p[s,s]=a; extra[s,s]=np.einsum('ijk,i->jk',da,r[s])
        np.testing.assert_allclose(g,p.T@r,rtol=1e-11,atol=1e-9)
        np.testing.assert_allclose(hlocal,p.T@j+extra,rtol=1e-11,atol=2e-6)
    guard(); assert owner.checkpoint()==before


def test_descriptor_factors_recovery_restart_and_chart_invariance(accepted_owner):
    source=accepted_owner; before=source.checkpoint()
    owner=public.create_analysis(definition=source._definition)
    try:
        owner.restore(before)
        packet,_=buckling.capture(owner,owner._generation,mixed=True)
        h,km,g=dense_mixed_oracle(owner,packet); free=list(packet.free)
        _,_,scale=buckling._scaled_pencil(owner,packet); congruence=np.outer(scale,scale)
        values=eig(km[np.ix_(free,free)]*congruence,-g[np.ix_(free,free)]*congruence,right=False)
        finite=[v.real for v in values if np.isfinite(v) and v.real>0 and abs(v.imag)<=1e-8*abs(v.real)]
        old=owner.buckling(public.GEMixedBucklingRequest(owner._generation,count=2))
        assert type(old) is public.GEMixedBucklingResult and old.status=='completed',old.failure_reason
        assert owner.checkpoint()==before and old.policy==buckling.MIXED_POLICY
        for factor in old.prestress_multipliers:
            assert min(abs(factor-v)/max(1.,abs(v)) for v in finite)<1e-7
        owner.rebase_rotations(expected_generation=owner._generation)
        saved=owner.checkpoint(); new=owner.buckling(public.GEMixedBucklingRequest(owner._generation,count=2))
        assert new.status=='completed',new.failure_reason
        assert owner.checkpoint()==saved
        for result in (old,new):
            assert max(result.equation_residuals)<=1e-8 and max(result.constraint_residuals)<=1e-8
            assert result.resultant_modes and result.multiplier_modes
        for a,b in zip(old.prestress_multipliers,new.prestress_multipliers,strict=True):
            assert abs(a-b)/max(abs(a),abs(b),np.finfo(float).tiny)<=1e-9
        test_actual_update_covector_plateau(owner)
        replay=public.create_analysis(definition=owner._definition)
        try:
            replay.restore(saved); restored,_=buckling.capture(replay,replay._generation,mixed=True)
            current,_=buckling.capture(owner,owner._generation,mixed=True)
            np.testing.assert_array_equal(restored.hessian.toarray(),current.hessian.toarray())
            assert replay.checkpoint()==saved
        finally: replay.close()
    finally: owner.close()
    assert source.checkpoint()==before


def test_distinct_admission_and_stale_guard(accepted_owner):
    owner=accepted_owner
    assert owner.mixed_buckling_capabilities['definition_admitted']
    assert owner.mixed_buckling_capabilities['material_capture_validation_pending']
    assert not owner.point_buckling_capabilities['definition_admitted']
    for cls in (public.GEPointBucklingRequest,public.GELineBucklingRequest):
        with pytest.raises(ValueError): owner.buckling(cls(owner._generation))
    with pytest.raises(ValueError,match='matching generation'):
        buckling.capture(owner,owner._generation+1,mixed=True)


def test_consumed_shell_policy_and_correction_mutation_guard(accepted_owner):
    owner = accepted_owner; driver = owner._finite_driver
    before = owner.checkpoint()
    _, guard = buckling.capture(owner, owner._generation, mixed=True)
    stable = driver._stable_s3
    driver._stable_s3 = not stable
    try:
        with pytest.raises(ValueError, match='accepted state changed'): guard()
    finally:
        driver._stable_s3 = stable
    guard()
    corrections = driver._q4_corrections
    driver._q4_corrections = dict(corrections)
    driver._q4_corrections[-1] = 'mutation witness'
    try:
        with pytest.raises(ValueError, match='accepted state changed'): guard()
    finally:
        driver._q4_corrections = corrections
    guard()
    assert owner.checkpoint() == before


@pytest.mark.parametrize('corruption', ('q', 'resultants', 'nan_equilibrium', 'nan_constraint'))
def test_current_accepted_equilibrium_rejected_before_factorization(accepted_owner, monkeypatch, corruption):
    owner = accepted_owner; driver = owner._finite_driver
    saved_state = driver._state; saved_accepted = driver._accepted
    original = owner.checkpoint()
    monkeypatch.setattr(buckling, 'factorize_cached', lambda *args: pytest.fail('corrupt accepted state reached factorization'))
    try:
        if corruption == 'q':
            q = saved_state.q.copy(); q[6] += 1e-3
            driver._state = replace(saved_state, q=q)
        elif corruption == 'resultants':
            resultants = saved_state.resultants.copy(); resultants[0,0] += 1000.
            driver._state = replace(saved_state, resultants=resultants)
        else:
            field = 'free_equilibrium_ratio' if corruption == 'nan_equilibrium' else 'constraint_max'
            driver._accepted = replace(saved_accepted, **{field: float('nan')})
        corrupt_checkpoint = owner.checkpoint()
        with pytest.raises(ValueError, match='equilibrium/constraint witness'):
            owner.buckling(public.GEMixedBucklingRequest(owner._generation))
        assert owner.checkpoint() == corrupt_checkpoint
        assert driver._state is not None and driver._accepted is not None
    finally:
        driver._state = saved_state; driver._accepted = saved_accepted
    assert owner.checkpoint() == original


@pytest.mark.parametrize('shell,beams', (('q4','fibre'), ('s3','generalized'),
                                      ('s3','fibre'), ('mixed','mixed')))
def test_public_example_definition_matches_exercised_model(shell, beams):
    from examples.stiffened_shell_ge_mixed_buckling import definition as example_definition
    candidate = example_definition(shell, beams); expected = definition(shell, beams)
    assert candidate.shell == expected.shell
    assert candidate.points == expected.points and candidate.fixed == expected.fixed
    assert candidate.q4_pullback_policy == expected.q4_pullback_policy
    assert tuple(beam.sha256 for beam in candidate.beams) == tuple(beam.sha256 for beam in expected.beams)


def test_fibre_admission_every_station_identity_history_branch_and_high_low(accepted_owner):
    owner=accepted_owner
    core=next((element.operator for element,_,_ in owner._beam_parts if type(element.operator) is RetainedFibreOperator),None)
    if core is None: return
    origin=core.cell.virgin()
    stations=[dict(fibres=[dict(fibre_id=f.fibre_id,stress=[0.,0.],branch='ELASTIC')
                          for f in core.section.fibres]) for _ in origin.stations]
    def response(history=origin):
        return SimpleNamespace(history=history,material=json.dumps(dict(cell_identity=core.cell.identity,
            derivative_kind='CLASSICAL_SMOOTH_BRANCH',stations=stations)))
    buckling._elastic_fibre(core,response(),origin)
    with pytest.raises(ValueError,match='must not advance'):
        buckling._elastic_fibre(core,response(replace(origin,cell_identity=origin.cell_identity+'changed')),origin)
    last=stations[-1]['fibres'][-1]; fibre=core.section.fibres[-1]
    last['fibre_id']='wrong'
    with pytest.raises(ValueError,match='inventory/order'): buckling._elastic_fibre(core,response(),origin)
    last['fibre_id']=fibre.fibre_id; last['branch']='PLASTIC_POSITIVE'
    with pytest.raises(ValueError,match='elastic-interior'): buckling._elastic_fibre(core,response(),origin)
    last['branch']='ELASTIC'; radius=fibre.curve.flow_stress[0]; eps=np.finfo(float).eps
    last['stress']=[radius,-128*eps*radius]
    buckling._elastic_fibre(core,response(),origin)
    for pair in ([radius,-32*eps*radius],[radius,0.],[radius,eps*radius]):
        last['stress']=pair
        with pytest.raises(ValueError,match='yield boundary'): buckling._elastic_fibre(core,response(),origin)
    # Opposing fibre stresses must never cancel into an accepted mean stress.
    stations[-1]['fibres'][0]['stress']=[-radius,0.]
    last['stress']=[radius,0.]
    with pytest.raises(ValueError,match='yield boundary'): buckling._elastic_fibre(core,response(),origin)
    for station in stations:
        for recovered in station['fibres']: recovered['stress']=[0.,0.]
    tiny_fibres=tuple(replace(f,curve=FlowCurve.linear(1e-320,0.)) for f in core.section.fibres)
    tiny_core=SimpleNamespace(cell=core.cell,section=SimpleNamespace(identity=core.section.identity,fibres=tiny_fibres))
    buckling._elastic_fibre(tiny_core,response(),origin)
    last['stress']=[1e-320,0.]
    with pytest.raises(ValueError,match='yield boundary'): buckling._elastic_fibre(tiny_core,response(),origin)
    minimum_core=SimpleNamespace(cell=core.cell,section=SimpleNamespace(identity=core.section.identity,
        fibres=tuple(replace(f,curve=FlowCurve.linear(float(np.nextafter(0.,1.)),0.)) for f in core.section.fibres)))
    for station in stations:
        for recovered in station['fibres']: recovered['stress']=[0.,0.]
    with pytest.raises(ValueError,match='yield boundary'): buckling._elastic_fibre(minimum_core,response(),origin)


def test_actual_fixed_history_fibre_cell_elastic_unload(accepted_owner):
    core=next((element.operator for element,_,_ in accepted_owner._beam_parts
               if type(element.operator) is RetainedFibreOperator),None)
    if core is None: return
    # Controlled constitutive predecessor only: no accepted owner history or
    # material parameter changes. Actual cell equilibrium unloads at zero p.
    virgin=core.cell.virgin()
    predecessor=replace(virgin,stations=tuple(FibreHistory(core.section.identity,
        tuple((1e-4,0.,2e-4,0.) for _ in core.section.fibres)) for _ in virgin.stations))
    payload=core.cell.response([0.]*18,predecessor)
    assert buckling.canonical(payload['history'])==buckling.canonical(predecessor)
    assert all(f.branch=='ELASTIC' for s in payload['stations'] for f in s['fibres'])
    actual=SimpleNamespace(history=payload['history'],material=buckling.canonical(payload))
    buckling._elastic_fibre(core,actual,predecessor)


def test_mixed_cancel_backend_material_and_mutation_boundaries(accepted_owner,monkeypatch):
    owner=accepted_owner; before=owner.checkpoint()
    packet,guard=buckling.capture(owner,owner._generation,mixed=True)
    saved=owner._coordinate_scale; owner._coordinate_scale=saved.copy(); owner._coordinate_scale[0]*=2
    try:
        with pytest.raises(ValueError,match='model/support provenance'): guard()
    finally: owner._coordinate_scale=saved
    guard()
    token=CancellationToken(); handle=_MockInverse()
    monkeypatch.setattr(buckling,'factorize_cached',lambda *args:handle)
    def interrupted(operator,**kwargs):
        operator.matvec(np.ones(operator.shape[0])); token.cancel('mixed iteration')
        operator.matvec(np.ones(operator.shape[0]))
    monkeypatch.setattr(buckling,'eigs',interrupted)
    result=owner.buckling(public.GEMixedBucklingRequest(owner._generation),cancellation_token=token)
    assert result.status=='cancelled' and not result.full_coordinates
    def broken(rhs): raise RuntimeError('Sparse solve produced NaN/Inf values')
    handle.solve=broken
    monkeypatch.setattr(buckling,'eigs',lambda operator,**kwargs:operator.matvec(np.ones(operator.shape[0])))
    result=owner.buckling(public.GEMixedBucklingRequest(owner._generation))
    assert result.status=='failed' and not result.full_coordinates
    if owner._has_physical_fibre:
        def reject(*args): raise ValueError('injected physical fibre active plastic branch')
        monkeypatch.setattr(buckling,'_elastic_fibre',reject)
        monkeypatch.setattr(buckling,'factorize_cached',lambda *args:pytest.fail('active material reached factorization'))
        with pytest.raises(ValueError,match='active plastic'):
            owner.buckling(public.GEMixedBucklingRequest(owner._generation))
    assert owner.checkpoint()==before


def dense_mixed_oracle(owner, packet):
    """Independent dense scatter of actual local scalar-potential derivatives.

    Point AD uses the actual additive shell chart and zero-Exp beam chart,
    avoiding the production spatial-row point conversion entirely.
    """
    snap = packet.snapshot; driver = owner._finite_driver
    size = packet.material.shape[0]
    assert size <= 256, 'dense oracle is bounded and never a production solve'
    h = np.zeros((size, size))
    material = np.zeros_like(h)
    geometric = np.zeros_like(h)
    for (_eid, element, slots, _), origin in zip(owner._shell_parts, pickle.loads(snap.shell_origins), strict=True):
        reference = np.array([owner._shell_model.mesh.nodes[node].coords() for node in element.node_ids])
        anchors = None if snap.shell_anchors is None else snap.shell_anchors[slots[::6]//6]
        local, d, dd = deformation(reference, snap.q[slots], stable_translation=driver._stable_s3 if len(slots)==18 else True, rotation_anchors=anchors)
        material_object = owner._shell_model.get_material(element.material_name)
        if len(slots) == 18:
            f, k, _ = element.compute_nonlinear_response(owner._shell_model.mesh, material_object, local, origin, 3, True)
            local_material = k
            local_geometric = np.zeros_like(k)
        else:
            f, k, _, parts = element.compute_nonlinear_response(owner._shell_model.mesh, material_object,
                local, origin, 3, True, _return_tangent_components=True)
            local_material = parts['material']
            local_geometric = parts['geometric']
        h[np.ix_(slots,slots)] += d.T@k@d+np.einsum('i,ijk->jk',f,dd)
        material[np.ix_(slots,slots)] += d.T@local_material@d
        # Scatter prestress independently; subtracting two large material-scale
        # matrices loses binary64 digits in small geometric entries.
        geometric[np.ix_(slots,slots)] += d.T@local_geometric@d+np.einsum('i,ijk->jk',f,dd)
    u = snap.q[owner._shell_count:owner._shell_count+driver._beam_nodal].reshape(-1, 6)[:, :3]
    positions, low = _position_pair(driver._reference_positions, u)
    for index, ((element, slots, _), origin) in enumerate(zip(owner._beam_parts, pickle.loads(snap.beam_origins), strict=True)):
        nodes = np.array([owner._beam_index[node] for node in element.node_ids])
        response = element.operator.evaluate(positions[nodes], low[nodes], snap.frames[nodes],
            snap.cells[index], snap.resultants[index], origin=origin)
        slots = slots+owner._shell_count
        h[np.ix_(slots, slots)] += response.hessian+response.hessian_low
        mixed = response.hessian+response.hessian_low
        local_material = mixed.copy(); local_material[:24, :24] = 0.
        material[np.ix_(slots, slots)] += local_material
        local_geometric = np.zeros_like(mixed); local_geometric[:24, :24] = mixed[:24, :24]
        geometric[np.ix_(slots, slots)] += local_geometric
    for index, (joint, slots) in enumerate(zip(owner._connections, owner._joint_slots, strict=True)):
        sn = slots[0]//6; bn = (slots[6]-owner._shell_count)//6
        sx, sl = _position_pair(owner._shell_xyz[sn], snap.q[slots[:3]])
        distance = (positions[bn]-sx)+(low[bn]-sl)
        v = [Jet2.variable(0., i, 18) for i in range(18)]
        theta = snap.q[slots[3:6]]
        anchor = np.eye(3) if snap.shell_anchors is None else snap.shell_anchors[sn]
        rs = matmul(so3_exp([v[3+j]+float(theta[j]) for j in range(3)]), constant_matrix(anchor, 18))
        rb = matmul(so3_exp(v[9:12]), constant_matrix(snap.frames[bn], 18))
        separation = [v[6+j]-v[j]+float(distance[j]) for j in range(3)]
        ct = [value-float(offset) for value, offset in zip(matvec(transpose(rs), separation), joint._offset)]
        cr = so3_log(matmul(constant_matrix(joint._relative.T, 18), matmul(transpose(rs), rb)))
        potential = sum((value*(v[12+j]+float(snap.multipliers[6*index+j])) for j, value in enumerate(ct+cr)), Jet2.constant(0., 18))
        slots = np.r_[slots, np.arange(packet.primal_count+6*index, packet.primal_count+6*index+6)]
        h[np.ix_(slots, slots)] += potential.hessian
        local_material = potential.hessian.copy(); local_material[:12, :12] = 0.
        material[np.ix_(slots, slots)] += local_material
        local_geometric = np.zeros((18,18)); local_geometric[:12, :12] = potential.hessian[:12, :12]
        geometric[np.ix_(slots, slots)] += local_geometric
    return h, material, geometric


def test_actual_update_covector_plateau(accepted_owner):
    owner = accepted_owner; driver = owner._finite_driver
    packet, _ = buckling.capture(owner, owner._generation, mixed=True)
    state = driver._state; n = packet.primal_count
    direction = np.sin(np.arange(packet.material.shape[0])+1.)
    direction[:n] *= owner._coordinate_scale
    direction[n:] *= owner._constraint_scale
    direction[owner._fixed] = 0.
    # Three fixed steps form the registered derivative plateau; no adaptation
    # or history commit changes the accepted parent or constitutive update.
    steps = (3e-6, 1e-6, 3e-7)
    cache_names = ('_line_trial_recovery','_point_trial_recovery','_distributing_trial_recovery','_generalized_trial_recovery')
    caches = {name: getattr(driver, name) for name in cache_names}
    before = owner.checkpoint()
    def covector(amount):
        trial = driver._advance(state, direction, amount)
        residual = driver._assemble(trial, driver._load_factor,
            pickle.loads(packet.snapshot.shell_origins), pickle.loads(packet.snapshot.beam_origins),
            applied_load=packet.snapshot.applied_dead_load)[0]
        result = residual.copy()
        for node, row in enumerate(trial.q[:owner._shell_count].reshape(-1, 6)):
            s = slice(6*node+3, 6*node+6)
            result[s] = _exp_terms(row[3:])[1].T@residual[s]
        starts = [owner._shell_count+6*node+3 for node in range(len(owner._beam_ids))]
        retained = owner._shell_count+6*len(owner._beam_ids)
        starts += [retained+24*i+3*c for i in range(len(owner._beam_parts)) for c in range(2)]
        for start in starts:
            s = slice(start, start+3)
            result[s] = _exp_terms(amount*direction[s])[1].T@residual[s]
        return result
    try:
        finite_differences = [(covector(step)-covector(-step))/(2*step) for step in steps]
    finally:
        for name, value in caches.items():
            setattr(driver, name, value)
    exact = packet.hessian@direction
    row_scale = np.r_[owner._coordinate_scale, owner._constraint_scale]
    errors = [np.max(abs((finite_difference-exact)*row_scale))/max(1., np.max(abs(exact*row_scale)))
              for finite_difference in finite_differences]
    assert max(errors) <= 1e-6, dict(steps=steps, errors=errors)
    assert owner.checkpoint() == before
