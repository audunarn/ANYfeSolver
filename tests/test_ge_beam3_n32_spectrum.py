import ast
from copy import deepcopy
from pathlib import Path
from decimal import Decimal,localcontext
import numpy as np
import pytest
from docs.reference_cases import ge_beam3_spatial_physical_pencil as old
from docs.reference_cases import ge_beam3_n32_physical_pencil as new
from docs.reference_cases import ge_beam3_n32_spectral_compare as field
from docs.reference_cases import ge_beam3_n32_spectrum_worker as worker
from docs.reference_cases.ge_beam3_retained_prestress_protocol import canonical

def test_kernel_ast_has_only_three_capacity_constants_changed():
    a=ast.parse(Path(old.__file__).read_text());b=ast.parse(Path(new.__file__).read_text())
    a.body=a.body[1:];b.body=b.body[1:]
    counts={8192:0,512:0}
    class Capacity(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int and node.value in counts:
                counts[node.value]+=1;node.value={8192:12288,512:640}[node.value]
            return node
    Capacity().visit(a)
    assert counts=={8192:1,512:2} and ast.dump(a)==ast.dump(b)

def small_packet():
    return dict(left=np.eye(3).tolist(),right=np.eye(3).tolist(),geometric=[[-2.,0.,1.],[0.,8.,2.],[1.,2.,4.]],
        kinetic=[[1.,0.,0.],[0.,2.,0.]],free_dofs=[0,1,2],algebraic_dofs=[2])

@pytest.mark.parametrize('digits',(80,100))
def test_inherited_signed_kernel_byte_identical(digits):
    p=small_packet();before=canonical(p)
    expected=old.spectrum(p,digits,2);actual=new.spectrum(p,digits,2)
    assert canonical(expected)==canonical(actual) and canonical(p)==before
    assert actual['eigenvalues'][0]<0 and max(actual['original_residuals'])<=1e-11

def test_capacity_and_old_guard_unchanged():
    wide=[[0.]]*9216
    assert len(new.matrix(wide,1))==9216
    with pytest.raises(ValueError):old.matrix(wide,1)
    with pytest.raises(ValueError):new.matrix([[0.]]*12289,1)
    p=small_packet();p['geometric']=[[0.]]*641
    with pytest.raises(ValueError):new.assemble(p,80,lambda:None)

@pytest.mark.parametrize('kind',('mass','free','nan','shape','bool','singular'))
def test_bad_factors_fail_closed(kind):
    p=small_packet()
    if kind=='mass':p['kinetic'][0][2]=1.
    elif kind=='free':p['free_dofs']=[0,1,1]
    elif kind=='nan':p['geometric'][0][0]=float('nan')
    elif kind=='shape':p['left'][0].pop()
    elif kind=='bool':p['right'][0][0]=True
    else:p['geometric'][2][2]=-1.
    with pytest.raises((ValueError,np.linalg.LinAlgError)):new.spectrum(p,80,2)

def cross_matrix(v):
    x,y,z=v;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

@pytest.mark.parametrize('transformed',(False,True))
def test_physical_velocity_all_64_halves_and_six_rigid_fields(transformed):
    q=np.array([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]]) if transformed else np.eye(3)
    shift=np.array([.2,-.3,.7]) if transformed else np.zeros(3)
    modes=np.zeros((582,6));rotations=np.tile(q,(32,2,1,1))
    for i,x in enumerate(np.linspace(-1,1,65)):
        point=q@np.array([x,.1*(1-x*x),0.])+shift
        modes[6*i:6*i+3,:3]=np.eye(3);modes[6*i:6*i+3,3:]=-cross_matrix(point)
        modes[6*i+3:6*i+6]=999. # Nodal algebraic trace must not enter physical velocity.
    for half in range(64):modes[390+3*half:393+3*half,3:]=np.eye(3)
    for half in range(64):
        for t in (0.,.173,1.):
            x=-1+(half+t)/32;point=q@np.array([x,.1*(1-x*x),0.])+shift
            expected=np.block([[np.eye(3),-cross_matrix(point)],[np.zeros((3,3)),np.eye(3)]])
            np.testing.assert_allclose(field.velocity(modes,rotations,half,t),expected,rtol=0,atol=1e-14)
    for half,t in ((-1,.5),(64,.5),(True,.5),(0,float('nan')),(0,1.01)):
        with pytest.raises(ValueError):field.velocity(modes,rotations,half,t)
    with pytest.raises(ValueError):field.velocity(modes[:-1],rotations,0,.5)

@pytest.mark.parametrize('kind',('zero','sign','rate','mac','shape','nan'))
def test_matching_gate_mutations(kind):
    values=np.array([-4.,1.,9.,16.,25.,36.]);other=values.copy();mac=np.eye(6)
    j,errors,matched,passed=field.match(values,other,mac)
    assert passed and j.tolist()==list(range(6)) and not errors.any()
    if kind=='zero':other[0]=0.
    elif kind=='sign':other[0]=4.
    elif kind=='rate':other[2]=9./1.021**2
    elif kind=='mac':mac[2,2]=.949
    elif kind=='shape':mac=mac[:-1]
    else:mac[0,0]=float('nan')
    if kind in ('zero','shape','nan'):
        with pytest.raises(ValueError):field.match(values,other,mac)
    else:assert field.match(values,other,mac)[3] is False

def test_one_to_one_matching_and_strict_thresholds():
    values=np.array([-4.,1.,9.,16.,25.,36.]);permutation=[0,3,1,4,5,2]
    reference=values[permutation];mac=np.eye(6)[:,permutation]
    assert field.match(values,reference,mac)[3]
    reference=values.copy();reference[1]=1.0404
    assert field.match(reference,values,np.eye(6))[3] is False
    reference[1]=1.0199**2
    assert field.match(reference,values,np.eye(6)*.95)[3]

@pytest.mark.parametrize('sign',('plus','minus'))
def test_actual_factor_authority(sign,tmp_path):
    packet,mechanical=worker.inputs(sign)
    assert len(packet['kinetic'])==9216 and len(mechanical['cell_rotations'])==32
    assert packet['completed_targets']==0 and 98 in packet['free_dofs']
    (tmp_path/'manifest.json').write_bytes(b'{}\n')
    with pytest.raises(ValueError,match='manifest'):worker.inputs(sign,tmp_path)

def test_worker_has_no_production_import_or_owner_solve():
    tree=ast.parse(Path(worker.__file__).read_text())
    assert not any(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('anyfesolver') for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('Context','solve','solve_bvp','prepare') for n in ast.walk(tree))

def test_original_deadline_and_trace_cancellation(monkeypatch):
    times=iter((0.,121.));monkeypatch.setattr(new,'monotonic',lambda:next(times))
    with pytest.raises(ValueError,match='120-second'):new.spectrum(small_packet(),80,2)
    def cancelled():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):new.trace_solve([[Decimal(1)]],[[Decimal(1)]],cancelled)

@pytest.mark.parametrize('kind',('digits','dimension','mode','root','residual','pivot','trace','scope'))
def test_result_schema_mutations(kind):
    # Schema-only fixture, deliberately not presented as a solved eigenproblem.
    r=dict(digits=100,physical_dimension=381,algebraic_dimension=189,
        full_modes=np.zeros((6,582)).tolist(),eigenvalues=[-4.,1.,9.,16.,25.,36.],
        original_residuals=[0.]*6,mass_orthogonality=0.,original_ritz_error=0.,trace_pivots=['1']*189,
        trace_residual='0',schur_skew='0',negative_modes_retained=True,original_factors_used=True,
        physical_current_rest_mass=True,production_qualified=False,interval_certified=False)
    worker.validate_result(r,100)
    if kind=='digits':r['digits']=100.
    elif kind=='dimension':r['physical_dimension']=285
    elif kind=='mode':r['full_modes'][0].pop()
    elif kind=='root':r['eigenvalues'][1]=float('nan')
    elif kind=='residual':r['original_residuals'][0]=2e-11
    elif kind=='pivot':r['trace_pivots'][-1]='-1'
    elif kind=='trace':r['trace_residual']='1e-59'
    else:r['production_qualified']=True
    with pytest.raises(ValueError):worker.validate_result(r,100)
