"""Successor graph tests; mechanics only under the shared reviewed runner."""
import json
from hashlib import sha256
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ge_beam3_g3c_physical_history_owner as history

SCIENTIFIC_RECORDS=[]
ASSIGNMENT=None
MO_TESTS={
 'MO01':'test_physical_inventory_and_obligation_map',
 'MO02':'test_physical_family_work','MO03':'test_physical_family_work',
 'MO04':'test_physical_family_work','MO05':'test_physical_directional',
 'MO06':'test_physical_independent_joint_work_transport','MO07':'test_physical_history_assignment',
 'MO08':'DEFERRED_FULL_OPERATOR_HISTORY_TRANSPORT_PARTITION_ADDENDUM','MO09':'test_physical_atomicity',
 'MO10':'test_physical_atomicity','MO11':'test_physical_observation_and_cache_guards',
 'MO12':'test_physical_observation_and_cache_guards','MO13':'test_physical_preflight_guards',
 'MO14':'test_physical_history_assignment','MO15':'test_physical_mutation_assignment',
 'MO16':'test_physical_recovery_witness_guards','MO17':'SHARED_RUNNER_PROCESS_INVENTORY',
 'MO18':'SHARED_RUNNER_TWO_FULL_CYCLE_AND_INDEPENDENT_REVIEW'}


def selected():
    if type(ASSIGNMENT)is not dict or set(ASSIGNMENT)!={'graph','variant'}:
        raise ValueError('reviewed physical owner assignment required')
    if ASSIGNMENT['graph']not in history.GRAPHS or ASSIGNMENT['variant']not in history.VARIANTS:
        raise ValueError('unregistered physical owner assignment')
    return ASSIGNMENT['graph'],ASSIGNMENT['variant']


def record(name,**values):
    SCIENTIFIC_RECORDS.append(dict(test=name,assignment=ASSIGNMENT,**values,production_qualified=False,full_g3c_qualified=False))


def close(actual,expected,tolerance=1e-11):
    import numpy as np
    a=np.asarray(actual,dtype=float);b=np.asarray(expected,dtype=float)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    error=float(np.linalg.norm(a-b)/max(1.,np.linalg.norm(a),np.linalg.norm(b)))
    assert error<=tolerance
    return error


def test_physical_inventory_and_obligation_map():
    matrix=history.history_matrix();inherited,_=history.packet.authorities()
    assert matrix==inherited['history_matrix']
    assert len(matrix)==375 and sum(r['accepted_stages'] for r in matrix)==3075
    assert sum(r['accepted_stages']+1 for r in matrix)==3450
    assert len(history.work_inventory('smoke'))==5
    rehearsal=history.work_inventory('rehearsal')
    assert sum(r['kind']=='history' for r in rehearsal)==10 and sum(r['kind']=='prefix' for r in rehearsal)==80
    assert sum(r['stages'] for r in rehearsal if r['kind']=='history')==70
    assert set(MO_TESTS)=={f'MO{i:02}' for i in range(1,19)}
    _,source=history.packet.authorities();counts=((3,8,2,24),(3,9,2,24),(3,10,2,24),(3,9,2,24),(6,17,8,54))
    for name,expected in zip(history.GRAPHS,counts):
        for variant in history.VARIANTS:
            definition,expanded,_=source.expand_inert(name,variant)
            graph=expanded['graph']
            assert (len(graph['elements']),len(graph['nodes']),len(graph['joints']),6*(len(graph['fixed_nodes'])+len(graph['joints'])))==expected
            assert definition['schema']==source.DEFINITION_SCHEMA and definition['operator_graph_id']==source.OPERATOR_GRAPH
    record('inventory',histories=375,events=3075,prefixes=3450,obligations=MO_TESTS)


def test_physical_inert_schema_guards():
    import pytest
    p=history.packet
    for raw in (b'{"x":0,"x":1}\n',b'{"x":NaN}\n',b'{ "x":0}\n',b'{"x":Infinity}\n'):
        with pytest.raises(ValueError):p.strict(raw)
    for policy in ('GE_BEAM3_G3C_MIXED_ELASTIC_OWNER_V1','UNREGISTERED'):
        raw=p.canonical(dict(schema='GE_BEAM3_G3C_GRAPH_RESTART_V1',policy=policy))
        with pytest.raises(ValueError,match='physical successor checkpoint required'):
            p.preflight(raw,sha256(raw).hexdigest(),expected_runtime_sha256='0'*64)
    record('inert_schema',rejections=6)


def native_checks(previous,current):
    import numpy as np
    old={r['element_id']:r['payload'] for r in previous['native_rows']}
    for row in current['native_rows']:
        payload=row['payload'];response=payload['response']
        assert payload['previous_state_sha256']==old[row['element_id']]['state_sha256']
        assert payload['epoch']==current['epoch']
        A=np.asarray(response['full']['jacobian']);r=np.asarray(response['full']['residual'])
        assert A.shape==(42,42) and r.shape==(42,) and response['internal_error']<=1e-11
        solved=np.linalg.solve(A[18:,18:],np.column_stack((r[18:],A[18:,:18])))
        close(response['lift'],-solved[:,1:]);close(response['correction'],-solved[:,0])
        close(response['tangent'],A[:18,:18]-A[:18,18:]@solved[:,1:])
        close(response['residual'],r[:18]-A[:18,18:]@solved[:,0])


def test_physical_family_work():
    import numpy as np
    from anyfesolver._ge_beam3_g3c_physical_owner import MixedGraphOwner,recovery_witness
    from anyfesolver._ge_beam3_g3c_stable.operator import schur
    graph,variant=selected();owner=MixedGraphOwner(graph,variant)
    before=json.loads(owner.snapshot_bytes())['state']
    for command in history.commands('NONE',.01)[:2]:
        owner.solve(command);current=json.loads(owner.snapshot_bytes())['state'];native_checks(before,current);before=current
    snapshot=owner.snapshot_bytes()
    trial=owner.trial(np.asarray(current['total_u']),np.asarray(current['multipliers']),history.commands('NONE',.01)[2])
    candidate=json.loads(trial['candidate'])
    for row in candidate['adapter_rows']:
        actual=trial['diagnostics'][row['element_id']]
        assert recovery_witness(row['family'],actual)==row['recovery_witness_sha256']
        assert row['source_material_committed']is False and row['physical_recovery_complete']is False
        if row['family']=='Q4':
            close(actual.internal_block64@actual.internal_inverse64,np.eye(64))
            close(actual.internal_inverse64@actual.internal_block64,np.eye(64))
            close(actual.schur_chart_hessian,actual.physical_chart_hessian)
            close(actual.chart_force,actual.physical_chart_force+sum((c.chart_force for c in actual.numerical_channels),np.zeros(24)))
            for station in actual.stations:close(station.resultant,station.constitutive@station.strain)
    # MO03 retains the registered G1 nonzero native line/couple diagnostic
    # without changing the graph histories' frozen spatial dead nodal load.
    # Evaluate every actual native operator at this accepted pose and verify
    # the full42/internal24 response and its Schur reduction directly.
    model,elements,states=owner._native_model(current);ids=[n for n,x in json.loads(owner._expanded)['graph']['nodes']]
    total=np.asarray(current['total_u']);qa=np.asarray(current['rotations']);native_count=0
    for element in elements:
        mapping=element.get_dof_mapping(model.mesh);indices=[ids.index(n)for n in element.node_ids]
        x,low=element._coordinates(total[mapping]);state=states[element.element_id];response=state['response']
        frames=qa[indices]@element.operator.reference.nodal_triads
        zero=element.operator.evaluate(x,low,frames,response['rotations'],response['resultants'])
        loaded=element.operator.evaluate(x,low,frames,response['rotations'],response['resultants'],
            line=(.01,-.02,.005),couple=(.001,.002,-.001))
        assert np.linalg.norm(loaded['residual'][18:]-zero['residual'][18:])>1e-12
        assert abs(loaded['potential']-zero['potential'])>1e-12 and loaded['conservative']is False
        reduced=schur(loaded['residual'],loaded['jacobian'])
        solved=np.linalg.solve(loaded['jacobian'][18:,18:],
            np.c_[loaded['residual'][18:],loaded['jacobian'][18:,:18]])
        close(reduced[0],loaded['residual'][:18]-loaded['jacobian'][:18,18:]@solved[:,0])
        close(reduced[1],loaded['jacobian'][:18,:18]-loaded['jacobian'][:18,18:]@solved[:,1:])
        native_count+=1
    assert owner.snapshot_bytes()==snapshot and trial['state_committed']is False
    record('family_work',accepted_state_sha256=sha256(snapshot).hexdigest(),adapter_count=len(candidate['adapter_rows']),
           native_load_witnesses=native_count,native_internal_residual_nonzero=True,native_load_work_nonzero=True)


def directional(owner,command):
    import numpy as np
    state=json.loads(owner.snapshot_bytes())['state'];n=owner.size;m=len(state['multipliers'])
    u=np.asarray(state['total_u'])+np.linspace(-.002,.003,n);mu=np.resize([.3,-.2,.4,-.1,.25,-.15],m)
    z=np.r_[u,mu];v=np.sin(np.arange(n+m)+1.);v/=np.linalg.norm(v)
    before=owner.snapshot_bytes();trial=owner.trial(u,mu,command)
    close(trial['tangent'],trial['tangent'].T)
    errors=[]
    for h in (1e-4,1e-5,1e-6):
        a=z+h*v;b=z-h*v
        plus=owner.trial(a[:n],a[n:],command)['residual'];minus=owner.trial(b[:n],b[n:],command)['residual']
        errors.append(close((plus-minus)/(2*h),trial['tangent']@v,1e-7))
    assert owner.snapshot_bytes()==before
    return errors


def test_physical_directional():
    from anyfesolver._ge_beam3_g3c_physical_owner import MixedGraphOwner
    owner=MixedGraphOwner(*selected())
    record('directional',errors=directional(owner,history.commands('NONE',.01)[0]))


def test_physical_atomicity():
    import pytest
    from anyfesolver._ge_beam3_g3c_physical_owner import MixedGraphOwner
    _,source=history.packet.authorities();graph,variant=selected()
    expanded=source.expand_inert(graph,variant)[1]
    last=next(e['id'] for e in reversed(expanded['graph']['elements']) if e['family']!='NATIVE')
    targets=('family:'+str(last),'prepare','native_committed','before_publish')
    for target in targets:
        owner=MixedGraphOwner(*selected());before=owner.snapshot_bytes();seen=[]
        def failure(stage):
            if stage==target:seen.append(stage);raise RuntimeError('intentional late failure')
        with pytest.raises(RuntimeError,match='intentional late failure'):owner.solve(history.commands('NONE',.01)[0],hook=failure)
        assert seen and owner.snapshot_bytes()==before and owner._active is None and not owner._owned_lock.locked()
    record('atomicity',stages=list(targets))


def test_physical_token_and_definition_guards():
    import pytest
    from anyfesolver._ge_beam3_g3c_physical_owner import MixedGraphOwner
    for kind in ('nonce','definition','initial','reentry'):
        owner=MixedGraphOwner(*selected());before=owner._published
        def mutate(stage):
            if stage!='pose':return
            if kind=='nonce':object.__setattr__(owner,'_active',object())
            elif kind=='definition':object.__setattr__(owner,'_definition',owner._definition+b' ')
            elif kind=='initial':object.__setattr__(owner,'_initial','0'*64)
            else:owner.solve(history.commands('NONE',.01)[0])
        with pytest.raises((ValueError,RuntimeError)):owner.solve(history.commands('NONE',.01)[0],hook=mutate)
        assert owner._published is before and owner._active is None and not owner._owned_lock.locked()
        if kind=='initial':
            with pytest.raises(ValueError):owner.snapshot_bytes()
    record('tokens',rejections=4)


def independent_constraints(owner,total,qa,mu,W=None,translation=None):
    """Independent scalar differentiation of frozen support/joint work."""
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'docs/reference_cases'))
    import ge_beam3_q4_affine_numerical_chart as oracle
    S=oracle.Scalar;const=oracle.constants
    g=json.loads(owner._expanded)['graph'];ids=[n for n,x in g['nodes']];X=np.array([x for n,x in g['nodes']])
    n=owner.size;out=np.zeros(n+len(mu));H=np.zeros((len(out),len(out)))
    W=np.eye(3)if W is None else W;t=np.zeros(3)if translation is None else translation
    previous=np.asarray(json.loads(owner._published.state)['total_u'])
    for k,node in enumerate(g['fixed_nodes']):
        i=ids.index(node);z=[S.coordinate(total[6*i+j],j)for j in range(6)]
        Q=oracle.exponential([z[j+3]-previous[6*i+j+3]for j in range(3)])@const(qa[i])
        c=[z[j]+X[i,j]-(W@X[i]+t)[j]for j in range(3)]+list(oracle.logarithm(const(W.T)@Q))
        idx=list(range(6*i,6*i+6));mm=list(range(n+6*k,n+6*k+6))
        J=np.array([v.g[:6]for v in c]);h=sum(mu[6*k+j]*v.h[:6,:6]for j,v in enumerate(c))
        out[idx]+=J.T@mu[6*k:6*k+6];out[mm]=[v.v for v in c]
        H[np.ix_(idx,idx)]+=h;H[np.ix_(idx,mm)]+=J.T;H[np.ix_(mm,idx)]+=J
    for j,row in enumerate(g['joints']):
        ij=[ids.index(row['master']),ids.index(row['slave'])];k=len(g['fixed_nodes'])+j
        idx=[6*i+l for i in ij for l in range(6)];z=[S.coordinate(total[i],a)for a,i in enumerate(idx)]
        frames=np.array([row['master_frame'],row['slave_frame']]);Q=[]
        for a,i in enumerate(ij):
            Q.append(oracle.exponential([z[6*a+l+3]-previous[6*i+l+3]for l in range(3)])@const(qa[i]@frames[a]))
        distance=np.array([S(X[ij[1],l]-X[ij[0],l])+z[6+l]-z[l]for l in range(3)],dtype=object)
        offset=frames[0].T@(X[ij[1]]-X[ij[0]])
        c=list(Q[0].T@distance-offset)+list(oracle.logarithm(const((frames[0].T@frames[1]).T)@Q[0].T@Q[1]))
        J=np.array([v.g[:12]for v in c]);h=sum(mu[6*k+a]*v.h[:12,:12]for a,v in enumerate(c))
        mm=list(range(n+6*k,n+6*k+6));out[idx]+=J.T@mu[6*k:6*k+6];out[mm]=[v.v for v in c]
        H[np.ix_(idx,idx)]+=h;H[np.ix_(idx,mm)]+=J.T;H[np.ix_(mm,idx)]+=J
    return out,H


def test_physical_independent_joint_work_transport():
    import numpy as np
    from anyfesolver._ge_beam3_g3c_physical_owner import MixedGraphOwner
    graph,variant=selected();owner=MixedGraphOwner(graph,variant);n=owner.size
    state=json.loads(owner._published.state);qa=np.asarray(state['rotations']);u=np.linspace(-.012,.009,n);mu=np.resize([.3,-.2,.1,.25,-.15,.4],len(state['multipliers']))
    expected=independent_constraints(owner,u,qa,mu);actual=owner._constraints(u,qa,mu,None)
    close(actual[0],expected[0]);close(actual[1],expected[1])
    # The complete trial consists of the nonconstraint physical operator plus
    # this independently differentiated support/joint scalar potential.
    cmd=history.commands('NONE',.01)[0]
    physical=owner.trial(u,np.zeros_like(mu),cmd);full=owner.trial(u,mu,cmd)
    close(full['residual']-physical['residual'],expected[0]-independent_constraints(owner,u,qa,np.zeros_like(mu))[0])
    close(full['tangent']-physical['tangent'],expected[1]-independent_constraints(owner,u,qa,np.zeros_like(mu))[1])
    errors=[]
    for motion in history.MOTIONS[1:]:
        moved=MixedGraphOwner(graph,variant,motion);command=dict(kind='PREPARE_COMMON_MOTION',step=4)
        # Independent Rodrigues value, never owner._targets or joint pullback.
        programs=json.loads(moved._programs);v=np.asarray(programs['common_rotation_vectors'][int(motion[-1])]);a=np.linalg.norm(v)
        K=np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])
        W=(np.eye(3) if a==0. else
           np.eye(3)+np.sin(a)/a*K+(1-np.cos(a))/a**2*(K@K));t=np.array([2.,-3.,1.])
        if variant=='PROPER_GLOBAL_TRANSFORM':
            U=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]);W=U@W@U.T;t=U@t+np.array([2.,-3.,1.])-W@np.array([2.,-3.,1.])
        X=np.array([x for i,x in json.loads(moved._expanded)['graph']['nodes']]);q=u.reshape(-1,6).copy()
        q[:,:3]=(W@(X+q[:,:3]).T).T+t-X;q[:,3:]=(W@q[:,3:].T).T
        anchors=np.tile(W,(len(X),1,1));got=moved._constraints(q.ravel(),anchors,mu,command)
        ref=independent_constraints(moved,q.ravel(),anchors,mu,W,t)
        errors.extend([close(got[0],ref[0]),close(got[1],ref[1])])
        # Material joint multipliers are unchanged under common proper motion.
        first=n+6*len(json.loads(moved._expanded)['graph']['fixed_nodes'])
        close(got[0][first:],actual[0][first:])
    record('independent_joint_work_transport',common_motions=4,errors=errors)


def test_physical_observation_and_cache_guards():
    import numpy as np
    import pytest
    import threading
    from unittest.mock import patch
    from anyfesolver import _ge_beam3_g3c_physical_owner as mechanics
    Owner=mechanics.MixedGraphOwner;graph,variant=selected();owner=Owner(graph,variant)
    state=json.loads(owner._published.state);u=np.zeros(owner.size);mu=np.zeros(len(state['multipliers']));command=history.commands('NONE',.01)[0]
    baseline=owner.trial(u,mu,dict(command));before=owner._published
    def mutate_callers(stage):
        if stage=='pose':u[:]=17.;mu[:]=19.;command['force_scale']=10.
    actual=owner.trial(u,mu,command,hook=mutate_callers)
    close(actual['residual'],baseline['residual']);close(actual['tangent'],baseline['tangent']);assert owner._published is before
    probes=[]
    families=[e['id']for e in json.loads(owner._expanded)['graph']['elements']if e['family']!='NATIVE']
    stages=tuple(dict.fromkeys(('pose','family:'+str(families[0]),'family:'+str(families[-1]),'prepare','before_publish')))
    for stage in stages:
        for kind in ('foreign_nonce','cleared_nonce','replace_lock','release_lock','dispatch'):
            obj=Owner(graph,variant);original=obj._published;lock=obj._owned_lock
            def attack(observed):
                if observed!=stage:return
                if kind=='foreign_nonce':object.__setattr__(obj,'_active',object())
                elif kind=='cleared_nonce':object.__setattr__(obj,'_active',None)
                elif kind=='replace_lock':object.__setattr__(obj,'_lock',threading.Lock())
                elif kind=='release_lock':lock.release()
                else:object.__setattr__(obj,'_dispatch',())
            with pytest.raises((ValueError,RuntimeError)):obj.solve(history.commands('NONE',.01)[0],hook=attack)
            assert obj._published is original and obj._active is None
            probes.append(stage+':'+kind)
    obj=Owner(graph,variant);original=obj._published
    with patch.object(mechanics.MixedGraphOwner,'_targets',lambda self,cmd:None):
        with pytest.raises(ValueError):obj.solve(history.commands('NONE',.01)[0])
    assert obj._published is original
    # A coherent replacement of all public definition bytes and their seal is
    # still foreign to the constructor-captured graph identity.
    _,source=history.packet.authorities();other=next(name for name in history.GRAPHS if name!=graph)
    definition,expanded,programs=source.expand_inert(other,variant)
    obj=Owner(graph,variant);original=obj._published
    for name,value in (('_definition',mechanics.canonical(definition)),('_expanded',mechanics.canonical(expanded)),
                       ('_programs',mechanics.canonical(programs))):object.__setattr__(obj,name,value)
    object.__setattr__(obj,'_seal',mechanics.sha((obj._definition.hex(),obj._expanded.hex(),obj._programs.hex())))
    with pytest.raises(ValueError):obj.solve(history.commands('NONE',.01)[0])
    assert obj._published is original;probes.append('coherent_definition_swap')
    # A true overlapping capture is rejected by the owned lock while the
    # first nonpublishing observation remains deterministic.
    obj=Owner(graph,variant);original=obj._published;entered=threading.Event();release=threading.Event();failures=[]
    state=json.loads(original.state);u=np.zeros(obj.size);mu=np.zeros(len(state['multipliers']))
    def hold(stage):
        if stage=='pose':entered.set();release.wait(5)
    def first():
        try:obj.trial(u,mu,history.commands('NONE',.01)[0],hook=hold)
        except BaseException as exc:failures.append(exc)
    thread=threading.Thread(target=first);thread.start();assert entered.wait(5)
    with pytest.raises(RuntimeError,match='owner already in use'):
        obj.trial(u,mu,history.commands('NONE',.01)[0])
    release.set();thread.join(5);assert not thread.is_alive() and not failures and obj._published is original
    probes.append('concurrent_capture')
    if graph in ('J_Q4_PAIR','J_MULTIFAMILY_LOOP'):
        for kind in ('tuple','entry','descriptor','registry','operator'):
            obj=Owner(graph,variant);original=obj._published;cache=obj._q4_cache;local=cache[0][1]
            if kind=='tuple':object.__setattr__(obj,'_q4_cache',tuple(list(cache)))
            elif kind=='entry':object.__setattr__(obj,'_q4_cache',((cache[0][0],object()),)+cache[1:])
            elif kind=='descriptor':object.__setattr__(local,'_body',local._body+b' ')
            elif kind=='registry':object.__setattr__(obj,'_expanded',obj._expanded+b' ')
            else:object.__setattr__(local,'_prepared_seal','0'*64)
            with pytest.raises((ValueError,RuntimeError)):obj.solve(history.commands('NONE',.01)[0])
            assert obj._published is original
            probes.append('Q4:'+kind)
    record('observation_cache',caller_copy=True,rejections=probes)


def test_physical_recovery_witness_guards():
    import numpy as np
    import pytest
    from dataclasses import replace
    from anyfesolver import _ge_beam3_g3c_physical_owner as mechanics
    owner=mechanics.MixedGraphOwner(*selected());state=json.loads(owner._published.state)
    trial=owner.trial(np.linspace(-.001,.002,owner.size),np.zeros(len(state['multipliers'])),history.commands('NONE',.01)[0])
    candidate=json.loads(trial['candidate']);count=0
    for row in candidate['adapter_rows']:
        result=trial['diagnostics'][row['element_id']]
        assert mechanics.recovery_witness(row['family'],result)==row['recovery_witness_sha256']
        if row['family']=='Q4':
            station=result.stations[0];bad=np.array(station.resultant);bad[0]+=1.
            mutated=replace(result,stations=(replace(station,resultant=bad),)+result.stations[1:])
            with pytest.raises(ValueError):mechanics.recovery_witness('Q4',mutated)
            # Numerical-channel work must never be reported as physical work.
            extra=sum((c.chart_force for c in result.numerical_channels),np.zeros(24))
            if not np.any(extra):raise AssertionError('registered deformed leakage witness is zero')
            with pytest.raises(ValueError):mechanics.recovery_witness('Q4',replace(result,physical_chart_force=result.physical_chart_force+extra))
            # The uncondensed direct block contains only the geometric term.
            # Omitting the independently bound station Schur term must fail.
            with pytest.raises(ValueError):mechanics.recovery_witness('Q4',replace(result,schur_chart_hessian=result.direct_chart_hessian))
            coherent=np.eye(24)
            with pytest.raises(ValueError):mechanics.recovery_witness('Q4',replace(result,
                direct_chart_hessian=result.direct_chart_hessian+coherent,
                schur_chart_hessian=result.schur_chart_hessian+coherent))
            count+=4
    record('recovery_witness',mutations=count)
