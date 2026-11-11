"""Current generalized factor adapter, not public beam qualification."""
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from anyfesolver import _ge_beam3_native_generalized_factor_modal as factor
from anyfesolver._native_relative_factor_chain_modes import solve_relative_factor_chain_modes
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_p5.algebra import rotation
from test_ge_beam3_schur_line_program import save
from test_ge_beam3_generalized_slenderness_diagnostic import make as slender
from test_ge_beam3_native_generalized_modal import make
import test_ge_beam3_native_generalized_modal as inherited
import test_ge_beam3_native_spectral_prestress as prestress

BASE=Path('C:/Users/AudunArnesenNyhus/AppData/Local/ANYrelease/ge-beam3-energy-slenderness-2d8b046-20260908')

@pytest.mark.parametrize('case',('signed-wide','repeated','truncated'))
def test_relative_signed_kernel(case,tmp_path):
    if case=='signed-wide':
        left=np.diag([1.,2.,1e12]);geometric=np.diag([-3.,-1.,0.]);expected=np.array([-2.,3.,1e24]);count=3
    else:
        left=np.diag([1.,1.,2.]);geometric=np.zeros((3,3));expected=np.array([1.,1.,4.]);count=1 if case=='truncated' else 3
    if case=='truncated':
        with pytest.raises(ValueError,match='truncate'):
            solve_relative_factor_chain_modes(left,np.eye(3),geometric,np.eye(3),(0,1,2),(),bounds=(-10.,2e24),num_modes=count)
        save(tmp_path/'kernel.json',dict(case=case,rejected=True));return
    result=solve_relative_factor_chain_modes(left,np.eye(3),geometric,np.eye(3),(0,1,2),(),bounds=(-10.,2e24),num_modes=count)
    assert np.max(abs(result.eigenvalues-expected)/np.maximum(1.,abs(expected)))<=1e-11
    assert result.original_ritz_residual<=1e-11 and not result.production_qualified
    save(tmp_path/'kernel.json',dict(case=case,result=result))

@pytest.mark.parametrize('value',(0.,float('nan'),True,1e-8))
def test_relative_control_rejection(value,tmp_path):
    with pytest.raises(ValueError):
        solve_relative_factor_chain_modes(np.eye(2),np.eye(2),np.zeros((2,2)),np.eye(2),(0,1),(),bounds=(-1.,2.),num_modes=2,relative_width=value)
    save(tmp_path/'rejection.json',dict(rejected=True,kind='invalid-relative-width'))

@pytest.mark.parametrize('kind',('signed-zero','skew','indefinite'))
def test_compliance_factor_guards(kind,tmp_path):
    s=np.diag(np.geomspace(1e-12,1.,18));s[0,1]=s[1,0]=-0.
    if kind=='skew':s[0,1]=.1
    if kind=='indefinite':s[0,0]=-1.
    if kind=='signed-zero':
        before=s.tobytes();a,error=factor.compliance_factor(s);b,other=factor.compliance_factor(s)
        assert s.tobytes()==before and a.tobytes()==b.tobytes() and error==other and error<=1e-11
        save(tmp_path/'compliance.json',dict(kind=kind,factor=a,error=error));return
    with pytest.raises((ValueError,np.linalg.LinAlgError)):factor.compliance_factor(s)
    save(tmp_path/'compliance.json',dict(kind=kind,rejected=True))

def _facade():
    def prepare(*args,**kwargs):
        packet,guard=factor.prepare(*args,**kwargs);return packet.base,guard
    def solve(*args,**kwargs):
        packet,result=factor.solve_modes(*args,bounds=(-100.,1e6),**kwargs);return packet.base,result
    return SimpleNamespace(prepare=prepare,solve_modes=solve)

@pytest.mark.parametrize('mutation',('inertia','nodal_moment','state','map','equilibrium','couple','support','caller_change'))
def test_inherited_input_guards(mutation,tmp_path,monkeypatch):
    monkeypatch.setattr(inherited,'modal',_facade())
    inherited.test_input_fail_closed(mutation,tmp_path)

@pytest.mark.parametrize('kind',('yield-boundary','plastic'))
def test_current_material_boundary(kind,tmp_path,monkeypatch):
    monkeypatch.setattr(prestress,'modal',_facade())
    prestress.test_actual_material_boundary_is_not_a_frequency(kind,tmp_path)

def test_live_trial_ownership(tmp_path,monkeypatch):
    monkeypatch.setattr(prestress,'modal',_facade())
    prestress.test_deformed_kinetic_work_and_live_trial_ownership(tmp_path)

@pytest.mark.parametrize('curved',(False,True))
def test_current_reference_factor_reduction(curved,tmp_path):
    model,states,inertias=make(curved,clamped=True);before=canonical(states)
    packet,result=factor.solve_modes(model,states,np.zeros(18),inertias,np.zeros(18),bounds=(-100.,1e6),num_modes=12)
    assert canonical(states)==before and np.min(result.eigenvalues)>0
    assert np.linalg.norm(result.full_modes.T@packet.base.mass@result.full_modes-np.eye(12))<=1e-11
    save(tmp_path/'reference.json',dict(packet=packet,result=result))

def test_tension_compression_signed_geometry(tmp_path):
    records=[]
    for strain in (-.001,0.,.001):
        model,states,inertias,total,force,store,error=prestress.axial_state(strain,1)
        before=canonical(states);generation=store.generation
        packet,result=factor.solve_modes(model,states,total,inertias,force,bounds=(-100.,1e6),num_modes=6)
        assert canonical(states)==before and store.generation==generation
        assert np.any(packet.geometric) if strain else not np.any(packet.geometric)
        records.append(dict(strain=strain,eigenvalues=result.eigenvalues,packet=packet,result=result))
    assert records[0]['eigenvalues'][0]<records[1]['eigenvalues'][0]<records[2]['eigenvalues'][0]
    save(tmp_path/'prestress.json',dict(records=records))

@pytest.mark.parametrize('rho',(100.,10000.,1000000.))
def test_full_slender_spectrum(rho,tmp_path):
    raw=(BASE/'archive-manifest.json').read_bytes()
    assert len(raw)==14269 and sha256(raw).hexdigest().upper()=='D865F9E38096E14C0384947F6722BF6719E4757B961EE6C32FC6E99AB6F0C1D3'
    manifest=json.loads(raw);records=[]
    for curved in (False,True):
        for name,q in (('E',np.eye(3)),('GENERAL',rotation([.4,-.3,.2]))):
            label=('curved' if curved else 'straight')+'-'+name
            refpath=next((BASE/'runs'/str(rho)/'pytest').rglob(label+'-decimal-100.json'))
            refraw=refpath.read_bytes();bound=manifest[refpath.relative_to(BASE).as_posix()]
            assert [len(refraw),sha256(refraw).hexdigest().upper()]==bound
            target=np.array(json.loads(refraw)['eigenvalues'],dtype=float)
            model,states,inertias=slender(rho,curved,q);before=canonical(states)
            print(dict(stage='factor-spectrum',rho=rho,case=label),flush=True)
            packet,result=factor.solve_modes(model,states,np.zeros(18),inertias,np.zeros(18),bounds=(-100.,1e28),num_modes=12)
            errors=abs(result.eigenvalues-target)/np.maximum(1.,abs(target))
            row=dict(case=label,errors=errors,packet=packet,result=result,reference_roots=target)
            save(tmp_path/(label+'.json'),row)
            assert canonical(states)==before and np.max(errors)<=1e-11 and np.min(result.eigenvalues)>0
            records.append(dict(case=label,maximum_error=float(np.max(errors))))
    save(tmp_path/'assessment.json',dict(rho=rho,records=records,production_qualified=False,independent_review='PENDING'))
