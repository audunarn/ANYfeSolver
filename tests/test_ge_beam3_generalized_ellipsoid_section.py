"""Six-component coupled plastic section and owned local transactions."""
from dataclasses import replace
from hashlib import sha256
import numpy as np
import pytest
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection,GeneralizedStation,GeneralizedHistory
from anyfesolver._ge_beam3_fibre_section import canonical
from docs.reference_cases.ge_beam3_generalized_ellipsoid_oracle import response as independent
from test_ge_beam3_schur_line_program import save


def section(coupled=True):
    c=np.diag([4.,6.,8.,2.,3.,5.]);m=np.diag([1.,.25,.5,2.,.75,1.5])
    if coupled:
        a=np.eye(6);a[0,3]=.2;a[1,4]=-.3;a[2,5]=.25;a[0,1]=.1
        b=np.eye(6);b[0,5]=.2;b[1,3]=-.15;b[2,4]=.1
        c=a.T@c@a;m=b.T@m@b
    return EllipsoidalGeneralizedSection(c,m,.25,.6)


def values(r,name):return getattr(r,name)+getattr(r,name+'_low')
def scalar(pair):return sum(pair)


def oracle(s,e,origin):
    return independent(dict(elastic=s.elastic,metric=s.metric,strain=e,plastic=origin.plastic,accumulated=origin.accumulated,
        yield_force=s.yield_force,hardening=s.hardening))


@pytest.mark.parametrize('axis',range(6))
def test_each_component_yields(axis,tmp_path):
    s=section(False);e=np.eye(6)[axis]*.4;r=s.response(e);o=oracle(s,e,s.virgin())
    assert r.branch=='PLASTIC' and scalar(r.plastic_increment)>0
    assert abs(sum(r.history.plastic[axis]))>0
    np.testing.assert_allclose(values(r,'resultants'),o['stress'],atol=1e-11,rtol=0.)
    np.testing.assert_allclose(values(r,'tangent'),o['tangent'],atol=1e-11,rtol=0.)
    save(tmp_path/'axis.json',dict(axis=axis,response=r,oracle=o))


def test_nonproportional_history_and_oracle(tmp_path):
    s=section();origin=s.virgin();rows=[];last=0.
    for e in ([.3,-.2,.25,.4,-.3,.2],[.24,-.16,.20,.32,-.24,.16],[-.4,.3,-.2,-.3,.25,-.4],[.2,.4,.1,-.3,-.2,.5]):
        e=np.array(e);r=s.response(e,origin=origin);o=oracle(s,e,origin)
        np.testing.assert_allclose(values(r,'resultants'),o['stress'],atol=1e-11,rtol=0.)
        np.testing.assert_allclose(values(r,'tangent'),o['tangent'],atol=1e-11,rtol=0.)
        np.testing.assert_allclose([sum(v) for v in r.history.plastic],o['plastic'],atol=1e-11,rtol=0.)
        assert abs(scalar(r.incremental_potential)-o['potential'])<1e-11 and scalar(r.dissipation)>=0
        p=scalar(r.history.accumulated);assert p>=last;last=p
        rows.append(dict(response=r,oracle=o));origin=r.history
    assert rows[1]['response'].branch=='ELASTIC'
    assert all(sum(v)!=0 for v in origin.plastic)
    save(tmp_path/'history.json',rows)


@pytest.mark.parametrize('control',['strain','resultant'])
def test_potential_gradient_and_hessian(control,tmp_path):
    s=section();x=np.array([.3,-.2,.25,.4,-.3,.2])*(3 if control=='resultant' else 1)
    r=s.response(x,control=control);v=np.sin(np.arange(6)+.4);eps=2e-6
    a=s.response(x+eps*v,control=control);b=s.response(x-eps*v,control=control)
    if control=='strain':
        gradient=values(r,'resultants');hessian=values(r,'tangent')
        numerical=(values(a,'resultants')-values(b,'resultants'))/(2*eps)
        energy=(scalar(a.incremental_potential)-scalar(b.incremental_potential))/(2*eps)
    else:
        gradient=values(r,'strain');hessian=values(r,'compliance')
        numerical=(values(a,'strain')-values(b,'strain'))/(2*eps)
        energy=(scalar(a.dual_potential)-scalar(b.dual_potential))/(2*eps)
    errors=dict(gradient=abs(energy-gradient@v)/max(1.,abs(gradient@v)),hessian=float(np.linalg.norm(numerical-hessian@v)/max(1.,np.linalg.norm(hessian@v))))
    assert max(errors.values())<=1e-7
    np.testing.assert_allclose(hessian,hessian.T,atol=1e-11,rtol=0.);assert np.linalg.eigvalsh(hessian).min()>0
    fenchel=values(r,'resultants')@values(r,'strain')-scalar(r.dual_potential)-scalar(r.incremental_potential)
    assert abs(fenchel)<=1e-11
    save(tmp_path/'derivatives.json',dict(control=control,response=r,errors=errors,fenchel_error=float(fenchel)))


def test_material_coordinate_covariance(tmp_path):
    s=section();e=np.array([.3,-.2,.25,.4,-.3,.2]);r=s.response(e)
    # Explicit work-coordinate scaling/permutation; no inferred orientation.
    a=np.diag([2.,.5,1.,.25,4.,2.])[:,[0,2,1,3,5,4]];inv=np.linalg.inv(a)
    transformed=EllipsoidalGeneralizedSection(inv.T@s.elastic@inv,a@s.metric@a.T,.25,.6)
    rr=transformed.response(a@e)
    error=float(np.linalg.norm(values(rr,'resultants')-inv.T@values(r,'resultants')))
    assert error<=1e-11
    np.testing.assert_allclose(values(rr,'tangent'),inv.T@values(r,'tangent')@inv,atol=1e-11,rtol=0.)
    assert abs(scalar(rr.incremental_potential)-scalar(r.incremental_potential))<=1e-11
    save(tmp_path/'covariance.json',dict(base=r,transformed=rr,error=error))


def test_owned_trial_commit_discard_replay(tmp_path):
    s=section();a=GeneralizedStation(s);b=GeneralizedStation(s);e=np.array([.3,-.2,.25,.4,-.3,.2]);before=canonical(a.history)
    p=a.trial(e)
    with pytest.raises(ValueError):b.commit(p)
    assert canonical(a.history)==before
    stale=p;p=a.trial(e)
    with pytest.raises(ValueError):a.commit(stale)
    accepted=a.commit(p);assert a.epoch==1 and canonical(a.replay())==canonical(p.response)
    with pytest.raises(ValueError):a.commit(p)
    before=canonical(a.history);p=a.trial(.1*e);a.discard(p);assert canonical(a.history)==before
    pending=a.trial(e)
    with pytest.raises(ValueError):a.trial([float('nan')]*6)
    with pytest.raises(ValueError):a.commit(pending)
    p=a.trial(-e);object.__setattr__(p.response,'production_qualified',True)
    with pytest.raises(ValueError,match='mutated'):a.commit(p)
    assert canonical(a.history)==before and a.epoch==1
    with pytest.raises(AttributeError):a.history=s.virgin()
    with pytest.raises(AttributeError):a.epoch=10
    save(tmp_path/'transaction.json',dict(history=accepted,replay=a.replay(),foreign_stale_repeat_mutation_rejected=True))


@pytest.mark.parametrize('bad',['nonsymmetric','indefinite','zero-hardening','boolean','foreign-history','negative-p','cache','parameters','control'])
def test_fail_closed(bad,tmp_path):
    s=section()
    if bad=='nonsymmetric':
        c=s.elastic.copy();c[0,1]+=.01
        with pytest.raises(ValueError):EllipsoidalGeneralizedSection(c,s.metric,.25,.6)
    elif bad=='indefinite':
        with pytest.raises(ValueError):EllipsoidalGeneralizedSection(s.elastic,-s.metric,.25,.6)
    elif bad=='zero-hardening':
        with pytest.raises(ValueError):EllipsoidalGeneralizedSection(s.elastic,s.metric,.25,0.)
    elif bad=='boolean':
        with pytest.raises(ValueError):s.response([True]*6)
    elif bad=='foreign-history':
        with pytest.raises(ValueError):s.response(np.zeros(6),origin=replace(s.virgin(),section_identity='0'*64))
    elif bad=='negative-p':
        with pytest.raises(ValueError):s.response(np.zeros(6),origin=replace(s.virgin(),accumulated=(-1.,0.)))
    elif bad=='cache':
        object.__setattr__(s,'_s',tuple(tuple(row) for row in np.eye(6)))
        with pytest.raises(ValueError,match='compiled'):s.response(np.zeros(6))
    elif bad=='parameters':
        object.__setattr__(s,'hardening',.7)
        with pytest.raises(ValueError,match='authority'):s.response(np.zeros(6))
    else:
        with pytest.raises(ValueError):s.response(np.zeros(6),control='unknown')
    save(tmp_path/'rejected.json',dict(case=bad,rejected=True))


def test_yield_boundary_and_serialization(tmp_path):
    s=EllipsoidalGeneralizedSection(np.eye(6),np.eye(6),.25,.5);x=np.array([.25,0.,0.,0.,0.,0.])
    r=s.response(x,control='resultant');assert r.branch=='YIELD_BOUNDARY' and r.derivative_kind=='SEMISMOOTH_ELASTIC_SELECTION'
    np.testing.assert_array_equal(values(r,'tangent'),np.eye(6))
    a=s.response(x*.5);b=s.response(x*.5);assert canonical(a)==canonical(b)
    assert not a.resultants.flags.writeable and not a.tangent.flags.writeable
    save(tmp_path/'boundary.json',dict(boundary=r,canonical_sha256=sha256(canonical(a)).hexdigest(),production_qualified=False))


def test_internal_deadline_representation_and_history(monkeypatch,tmp_path):
    import anyfesolver._ge_beam3_generalized_ellipsoid_section as module
    s=section()
    with pytest.raises(ValueError,match='range'):s.response(np.array([1e308]*6),control='resultant')
    bad=replace(s.virgin(),plastic=((1.,1.),)+s.virgin().plastic[1:])
    with pytest.raises(ValueError,match='normalized'):s.response(np.zeros(6),origin=bad)
    clock=[]
    def advance():clock.append(True);return 16.*len(clock)
    monkeypatch.setattr(module,'monotonic',advance)
    with pytest.raises(RuntimeError,match='deadline'):s.response(np.zeros(6))
    save(tmp_path/'limits.json',dict(deadline_rejected=True,unrepresentable_response_rejected=True,unnormalized_history_rejected=True))


def test_oracle_import_boundary(tmp_path):
    import ast
    from pathlib import Path
    source=Path(__file__).resolve().parents[1]/'docs/reference_cases/ge_beam3_generalized_ellipsoid_oracle.py'
    raw=source.read_bytes();tree=ast.parse(raw)
    imports={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
    assert imports=={'decimal','time'} and not any(isinstance(n,ast.Import) for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('eval','exec','__import__') for n in ast.walk(tree))
    save(tmp_path/'oracle-boundary.json',dict(imports=sorted(imports),source_sha256=sha256(raw).hexdigest(),no_producer_imports=True))
