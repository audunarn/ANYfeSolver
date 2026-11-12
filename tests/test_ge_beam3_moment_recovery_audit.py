"""Independent work-map development checks; no new equilibrium solves."""
import ast
import inspect
import numpy as np
import pytest
from docs.reference_cases import ge_beam3_moment_recovery_audit as audit


def exp(v):
    v=np.asarray(v); a=np.linalg.norm(v); s=audit.skew(v)
    return np.eye(3)+np.sinc(a/np.pi)*s+.5*np.sinc(a/(2*np.pi))**2*s@s


@pytest.mark.parametrize('angle',[[0.,0.,0.],[1e-8,-2e-8,3e-8],[.3,.2,-.4],[1.9,.7,-.6]])
def test_endpoint_log_work_gradient(angle):
    base=exp([2.,.4,1.]); q=base@exp(angle); moment=np.array([.7,-.2,.5]); direction=np.array([-.6,.2,.4])
    ell,j=audit.rotation_log_jacobian(base.T@q)
    np.testing.assert_allclose(ell,angle,rtol=1e-11,atol=1e-11)
    analytic=direction@base@j.T@moment; h=1e-6
    def value(sign): return moment@audit.rotation_log_jacobian(base.T@exp(sign*h*direction)@q)[0]
    assert abs((value(1)-value(-1))/(2*h)-analytic)<1e-7


def test_projection_objectivity_and_exact_constant_frame():
    r=np.array([exp([.1*t,.2*t,0.]) for t in (-1.,-.3,.3,1.)]); u=exp([.4,-.1,.7])
    q=u@r; w=[.2,.3,.3,.2]
    found=audit.best_constant_rotation(r,q,w)
    np.testing.assert_allclose(found,u,atol=1e-11,rtol=1e-11)
    common=exp([2.,.5,-.8])
    np.testing.assert_allclose(audit.best_constant_rotation(r,common@q,w),common@found,atol=1e-11,rtol=1e-11)


@pytest.mark.parametrize('matrix',[np.zeros((3,3)),np.diag([-1.,1.,1.]),exp([3.,0.,0.])])
def test_improper_or_outside_chart_rotation_rejects(matrix):
    with pytest.raises(ValueError): audit.rotation_log_jacobian(matrix)


def test_audit_imports_no_mechanics_or_reference_solver():
    tree=ast.parse(inspect.getsource(audit))
    modules=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any('anyfesolver' in n or 'moment_reference' in n for n in modules)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('solve','solve_force_program') for n in ast.walk(tree))


def test_saved_discrete_work_and_best_cell_frame_bound(tmp_path):
    if not (audit.ROOT/'comparison.json').is_file(): pytest.skip('External development packets not installed; not qualification')
    result=audit.build()
    (tmp_path/'audit.json').write_bytes(audit.canonical(result))
    assert len(result['rows'])==9 and sum(len(r['cells']) for r in result['rows'])==42
    assert result['native_solves']==result['reference_solves']==0 and not result['production_qualified']
    for row in result['rows']:
        for cell in row['cells']:
            assert cell['native_frame_weighted_rms']+1e-12>=cell['best_constant_frame_weighted_rms']


@pytest.mark.parametrize('mutation',['frame','force','curvature','moment','nodal_director','cell_rotation','residual'])
def test_independent_work_maps_reject_mutated_saved_fields(mutation):
    if not (audit.ROOT/'comparison.json').is_file(): pytest.skip('External development packets not installed; not qualification')
    detail=audit.parse((audit.ROOT/'fields-diagnostic.json').read_bytes())
    packet=audit.parse((audit.ROOT/'checkpoint-1.json').read_bytes())
    rec=detail['recoveries'][0]; reference=detail['references'][1]['value']
    station=rec['fields'][0]['stations'][0]; mechanical=packet['records'][0]['mechanical']
    if mutation=='frame': station['current_frame']=(exp([.1,0.,0.])@station['current_frame']).tolist()
    if mutation=='force': station['resultants'][0]+=.1
    if mutation=='curvature': station['strain'][3]+=.1
    if mutation=='moment': station['resultants'][3]+=.1
    if mutation=='nodal_director': mechanical['nodal_frames'][0]=(exp([.1,0.,0.])@mechanical['nodal_frames'][0]).tolist()
    if mutation=='cell_rotation': mechanical['cell_rotations'][0][0]=(exp([.1,0.,0.])@mechanical['cell_rotations'][0][0]).tolist()
    if mutation=='residual': packet['records'][0]['residual'][0]+=.1
    with pytest.raises(ValueError): audit.inspect_record(1,packet,rec,reference)
