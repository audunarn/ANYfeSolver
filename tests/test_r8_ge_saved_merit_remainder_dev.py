"""Prepared pure-source controls; no model, physical evaluation, FD or sparse solve."""
import ast,copy
from pathlib import Path
import pytest
from scripts import r8_ge_saved_merit_remainder_dev as d
def test_source_exact_linear_and_signed_load():
    K=dict(shape=[2,2],row=[0,0,1,1],col=[0,1,0,1],data=[0.,2.,1.,3.])
    assert d.mv(K,[-.25,-1.375])==[-2.75,-4.375]
    parts=[dict(slots=[0,1],force=[1.,2.],tangent=[[0.,2.],[1.,3.]]),dict(slots=[0],force=[-1.],tangent=[[0.]])]
    assert d.scatter_force(parts,2)==[0.,2.] and d.scatter_action(parts,[-.25,-1.375])==[-2.75,-4.375]
    result=d.merit_identity([2.,-1.],[-2.,1.],[0.,0.],.125)
    assert result['derivative0']==-10. and result['remainder_squared']==0.
    assert result['phi_trial']==5.*.875**2 and result['decomposition_error']==0.
def test_source_nonlinear_remainder_identity():
    # R0=1, Jd=-1 and actual remainder .25 at t=.5: exact phi change=-.4375.
    result=d.merit_identity([1.],[-1.],[.25],.5)
    assert result['linear_term']==-1. and result['linear_model_quadratic']==.25
    assert result['remainder_cross']==.25 and result['remainder_squared']==.0625
    assert result['measured_change']==-.4375 and result['decomposition_error']==0.
    # Smooth scalar counterexample, not a material case or a proposed physical step:
    # R(s)=1-s+9000s^2 has exact negative derivative at0, but every permitted
    # halving0..12 lies beyond its decreasing interval (0,1/9000).
    value=lambda s:1-s+9000*s*s
    assert all(value(2.**(-k))>1. for k in range(13)) and value(1e-4)<1.
    with pytest.raises(ValueError,match='complete merit'):d.merit_identity([1.,2.],[1.],[0.],.5)
@pytest.mark.skipif(not d.INPUT.exists(), reason='local saved MERIT_INPUT.json evidence is unavailable')
def test_source_complete_roles_and_refusal():
    v=d.input_data();assert len(v['roles'])==10 and sorted(i for rows in v['roles'].values() for i in rows)==list(range(132))
    assert v['roles']['fixed_shell_support_rows']==[0,1,2,3,4,5,18,19,20,21,22,23]
    for change in (lambda x:x.__setitem__('amount',1/128),lambda x:x['roles']['POINT_constraints'].pop(),lambda x:x['trial_assembly'].__setitem__('predecessor_shell_id',-1),lambda x:x['trial_q4_maps']['maps']['unavailable'].append('spatial_tangent')):
        bad=copy.deepcopy(v);change(bad)
        with pytest.raises(ValueError):d.validate(bad)
def test_source_inventory_no_mechanics():
    source=Path(d.__file__).read_text();tree=ast.parse(source)
    functions={n.name for n in ast.parse(Path(__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    assert len(d.SOURCE_NAMES)==4 and set(d.SOURCE_NAMES)==functions
    forbidden={'spsolve','solve','inv','pinv','evaluate','correct','_assemble','_advance','rotation','restore_ge_checkpoint_binary','solve_ge_displacement'}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in ast.walk(tree))
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(part in ast.unparse(n) for part in ('numpy','scipy','anyfesolver')) for n in ast.walk(tree))
    assert 'endpoint_tangent_trapezoid_remainder' in source and 'one finite pair/trapezoid cannot prove' in source
