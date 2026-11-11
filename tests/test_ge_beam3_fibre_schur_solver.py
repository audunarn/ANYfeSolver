"""Native state matrix parity; no historical nonlinear path or spectrum rerun."""
from hashlib import sha256
from pathlib import Path
import numpy as np
import pytest
from anyfesolver._ge_beam3_fibre_schur_solver import ResultantSchurSolver
from anyfesolver import _ge_beam3_fibre_line_program as control
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_fibre_line_program import model,line
from test_ge_beam3_line_fibre_modes import case,ROOT,read


@pytest.fixture(autouse=True)
def forbid_nonlinear_rerun(monkeypatch):
    def reject(*args,**kw): raise AssertionError('native history rerun forbidden')
    monkeypatch.setattr(control,'solve_force_program',reject)


def linearization(curved=True):
    p=ForceProgram((.7,),((3,.01,-.02,.03),))
    context=control.Context(model(curved=curved),p,line_forces=line())
    direction=np.zeros(context.layout.count)
    direction[context.layout.free]=.02*np.sin(np.arange(len(context.layout.free)))
    trial=context.layout.advance(context.initial.mechanical,direction)
    r,h,_,_=context.assemble(trial,.7,context.initial.histories)
    return context,context.tangent(r,h,.7),r


def compare(context,jacobian,rhs,tmp_path,name):
    capture=canonical(context.initial)
    factor=ResultantSchurSolver(context.layout,jacobian)
    answer=factor.solve(rhs)
    free=context.layout.free
    expected=np.zeros_like(rhs)
    expected[free]=np.linalg.solve(jacobian[np.ix_(free,free)],rhs[free])
    error=float(np.linalg.norm(answer.increment-expected)/max(np.finfo(float).tiny,np.linalg.norm(expected)))
    assert error<=1e-11
    assert answer.backward_error<=1e-11 and not answer.production_qualified
    assert canonical(context.initial)==capture
    assert not np.any(answer.increment[context.layout.fixed])
    record=dict(diagnostics=factor.diagnostics(),relative_full_solve_difference=error,
                backward_error=answer.backward_error,increment=answer.increment,
                matrix_sha256=sha256(canonical(jacobian)).hexdigest(),production_qualified=False)
    with (tmp_path/name).open('xb') as stream: stream.write(canonical(record))
    return factor,answer


@pytest.mark.parametrize('curved',[False,True])
def test_nonstationary_spatial_newton_and_factor_reuse(curved,tmp_path):
    c,a,r=linearization(curved)
    assert np.linalg.norm(a-a.T)>1e-4  # Do not mislabel the spatial Jacobian SPD.
    f,answer=compare(c,a,-r,tmp_path,'newton.json')
    repeated=f.solve(-r)
    np.testing.assert_array_equal(repeated.increment,answer.increment)
    assert f.diagnostics()['factorizations']==1 and f.diagnostics()['solves']==2
    assert f.diagnostics()['eliminated_force_coordinates']==18
    assert f.diagnostics()['retained_physical_cell_rotations']==6


@pytest.mark.parametrize('macros',[1,2,4])
def test_preserved_curved_multielement_states_and_multiple_rhs(macros,tmp_path):
    made,p,forces,raw,_=case(macros)
    c=control.Context(made,p,line_forces=forces)
    accepted,records=c.restore(raw,expected_sha256=sha256(raw).hexdigest())
    r,h,_,_=c.assemble(accepted.mechanical,1.,accepted.origins)
    a=c.tangent(r,h,1.)
    b=np.column_stack([np.cos(np.arange(c.layout.count)+offset) for offset in (0.,.4,.9)])
    f,_=compare(c,a,b,tmp_path,'saved-matrix.json')
    assert f.diagnostics()['eliminated_force_coordinates']==18*macros
    assert c.checkpoint(records)==raw


def test_accepted_plastic_origin_trial_and_recovered_force_increments(tmp_path):
    raw=read(ROOT/'ge-beam3-native-line-20260907-065c21d/cycle-a/pytest/test_physical_plasticity_resta0/paused.json',
             25593,'679c09a857c12cdfefd9ea96d25cce590b7a37ae2e0f6633afd8a7c5716f4736')
    p=ForceProgram((.25,.5,1.,.5,0.,-.5,0.),(),max_iterations=24)
    c=control.Context(model(plastic=True),p,line_forces=line((.7,.02,.03)))
    accepted,records=c.restore(raw,expected_sha256=sha256(raw).hexdigest())
    assert any(row[2]>0 for cell in accepted.histories for station in cell.stations for row in station.rows)
    step=np.zeros(c.layout.count); step[c.layout.free]=.001*np.sin(np.arange(len(c.layout.free)))
    trial=c.layout.advance(accepted.mechanical,step)
    r,h,_,_=c.assemble(trial,.5,accepted.histories)
    compare(c,c.tangent(r,h,.5),-r,tmp_path,'plastic-trial.json')
    assert c.checkpoint(records)==raw


@pytest.mark.parametrize('mutation',['shape','boolean','nan','positive-material','asymmetric-material','coupling','layout'])
def test_malformed_linearization_rejected(mutation):
    c,a,r=linearization()
    if mutation=='shape': a=a[:-1,:-1]
    if mutation=='boolean': a=np.ones_like(a,dtype=bool)
    if mutation=='nan': a[0,0]=np.nan
    if mutation=='positive-material': a[24:42,24:42]*=-1
    if mutation=='asymmetric-material': a[24,25]+=.001
    if mutation=='coupling': a[24,6]+=.001
    if mutation=='layout': c.layout.compatibility=c.layout.compatibility[:-1]
    with pytest.raises((ValueError,np.linalg.LinAlgError)): ResultantSchurSolver(c.layout,a)


def test_capture_ownership_rhs_and_mutation_guards():
    c,a,r=linearization(); f=ResultantSchurSolver(c.layout,a)
    with pytest.raises(ValueError): f.matrix.setflags(write=True)
    with pytest.raises(AttributeError): f.geometry=()
    for b in (np.ones(len(a)-1),np.ones(len(a),dtype=bool),np.full(len(a),np.inf),np.ones((len(a),17))):
        with pytest.raises(ValueError): f.solve(b)
    # Mutating a caller-owned matrix cannot silently retarget a captured factor.
    before=f.solve(-r).increment; a[:]=0.
    np.testing.assert_array_equal(f.solve(-r).increment,before)
    c.layout.free=c.layout.free[:-1]
    with pytest.raises(ValueError,match='layout changed'): f.solve(-r)


@pytest.mark.parametrize('case_id',['zero-rhs','indefinite-geometric-block'])
def test_zero_rhs_and_nonpositive_geometric_directions_are_not_regularized(case_id,tmp_path):
    c,a,r=linearization()
    if case_id=='zero-rhs': r[:]=0.
    else:
        g=[i for i in c.layout.free if i not in c.layout.compatibility]
        a[np.ix_(g,g)]-=1000*np.eye(len(g))
    factor,result=compare(c,a,-r,tmp_path,'signed-linear-system.json')
    if case_id=='zero-rhs': assert not np.any(result.increment) and result.backward_error==0.
    else: assert np.min(np.linalg.eigvalsh((factor.reduced+factor.reduced.T)/2))<0.
    assert factor.diagnostics()['matrix_class']=='general'


def test_cross_element_resultant_coupling_is_not_silently_dropped():
    made,p,forces,_,_=case(2)
    c=control.Context(made,p,line_forces=forces)
    r,h,_,_=c.assemble(c.initial.mechanical,0.,c.initial.histories)
    a=c.tangent(r,h,0.); start=c.layout.nodal_count
    a[start+6,start+30]=a[start+30,start+6]=1e-20
    with pytest.raises(ValueError,match='element-local'): ResultantSchurSolver(c.layout,a)
