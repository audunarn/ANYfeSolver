"""Authentic current vibration: independent controls plus one public CLI run.

No old scope is replayed by importing old tests; fixed original construction and
criteria are retained. Scientific runtime requires separate root release.
"""
import json
import copy
import ast
from fractions import Fraction
import math
import os
from pathlib import Path
import sys
import numpy as np
import pytest
from examples import native_full_amplitude_vibration_dev as cli
from examples import native_vibration_physical_reference as ref


def state_fixture():
    points,frames,_,_=ref.primitive_inputs()
    return dict(positions=points.copy(),position_low=np.zeros((3,3)),nodal_frames=frames.copy(),
                cell_rotations=np.tile(np.eye(3),(1,2,1,1)),resultants=np.zeros((1,18)))


def test_authentic_curved_cross_inertia_fixture():
    points,frames,section,mass,definition=cli.fixture();p,f,c,m=ref.primitive_inputs()
    assert np.array_equal(points,p)and np.array_equal(frames,f)
    assert np.array_equal(section.elastic,c)and np.array_equal(mass,m)
    assert points[1,1]==.2 and np.count_nonzero(mass[:3,3:])>0
    assert section.elastic[0,3]!=0. and section.elastic[1,5]!=0.
    assert cli.CRITERIA['amplitude']==1e-4 and cli.CRITERIA['maximum_accepted_steps']==20
    assert definition.raw and np.linalg.eigvalsh(mass)[0]>0.


def test_independent_eccentric_curved_kinetic_rigid_and_covariance():
    s=state_fixture();v=np.zeros(24);speed=np.array((.013,-.021,.017))
    for n in range(3):v[6*n:6*n+3]=speed
    result=ref.kinetic(s,v)
    a=.4;L=math.sqrt(1+a*a)+math.asinh(a)/a
    j2=(a*math.sqrt(1+a*a)*(1+2*a*a)-math.asinh(a))/(4*a**3)
    lever=np.array((2*.035,.2*(L-j2)-2*.012,-.025*L))
    assert abs(result['T']-.5*2.7*L*float(speed@speed))<1e-13
    assert np.linalg.norm(result['P']-2.7*L*speed)<1e-13
    assert np.linalg.norm(result['H']-2.7*np.cross(lever,speed))<1e-13
    assert np.linalg.norm(result['inertia'])<1e-13
    # Nonzero nonuniform nodal rates and TWO unequal spinning cell fields.
    v[:3]=(.03,-.01,.02);v[6:9]=(-.02,.04,.01);v[12:15]=(.01,.02,-.03)
    v[18:24]=(.12,-.07,.09,-.08,.11,.05)
    s['cell_rotations'][0,0]=ref.rotation(np.array((.03,-.04,.02)))
    before=ref.kinetic(s,v);R=ref.rotation(np.array((.17,-.13,.09)));shift=np.array((.2,-.1,.3))
    other={k:a.copy()for k,a in s.items()};other['positions']=s['positions']@R.T+shift
    other['nodal_frames']=np.einsum('ij,njk->nik',R,s['nodal_frames'])
    other['cell_rotations']=np.einsum('ij,ecjk->ecik',R,s['cell_rotations'])
    after_v=v.copy()
    for first in (0,6,12,18,21):after_v[first:first+3]=R@v[first:first+3]
    after=ref.kinetic(other,after_v)
    assert abs(after['T']-before['T'])<1e-12
    assert np.linalg.norm(after['P']-R@before['P'])<1e-12
    assert np.linalg.norm(after['H']-(R@before['H']+np.cross(shift,R@before['P'])))<1e-12


def test_independent_coupled_elastic_full42_virtual_work():
    s=state_fixture();s['positions'][1]+=np.array((.0003,-.0002,.0001))
    s['cell_rotations'][0,0]=ref.rotation(np.array((.002,-.001,.0015)))
    s['cell_rotations'][0,1]=ref.rotation(np.array((-.0015,.002,.001)))
    s['resultants'][0]=np.array((.11,-.08,.09,-.12,.07,.13,.023,-.019,.017,-.011,.013,-.007,.021,-.015,.009,-.017,.019,-.013))
    exact=ref.elastic(s);p=s['resultants'][0]
    assert abs(exact['witness']-float(p@exact['compatibility']))<1e-14
    numerical=[];h=1e-6
    for index in range(42):
        values=[]
        for sign in (-1.,1.):
            q={k:a.copy()for k,a in s.items()}
            if index<18:
                node,slot=divmod(index,6)
                if slot<3:q['positions'][node,slot]+=sign*h
                else:
                    axis=np.zeros(3);axis[slot-3]=sign*h
                    q['nodal_frames'][node]=ref.rotation(axis)@q['nodal_frames'][node]
            elif index<24:
                cell,slot=divmod(index-18,3);axis=np.zeros(3);axis[slot]=sign*h
                q['cell_rotations'][0,cell]=ref.rotation(axis)@q['cell_rotations'][0,cell]
            else:q['resultants'][0,index-24]+=sign*h
            values.append(ref.elastic(q)['Pi'])
        numerical.append((values[1]-values[0])/(2*h))
    assert np.linalg.norm(np.array(numerical)-exact['residual'])/np.linalg.norm(exact['residual'])<=1e-5
    assert np.count_nonzero(exact['residual'][24:])==18

    # Adversarial common translation with retained low parts: no O(1)-O(1)
    # subtraction or early high+low rounding may manufacture elastic work.
    translated={k:a.copy()for k,a in s.items()}
    for node in range(3):
        for axis,shift in enumerate((2.**30,-2.**29,2.**28)):
            value=Fraction(float(s['positions'][node,axis]))+Fraction(float(s['position_low'][node,axis]))+Fraction(shift)
            translated['positions'][node,axis]=float(value)
            translated['position_low'][node,axis]=float(value-Fraction(float(translated['positions'][node,axis])))
    moved=ref.elastic(translated)
    assert moved['Pi']==exact['Pi'] and moved['Psi']==exact['Psi']
    assert np.array_equal(moved['residual'],exact['residual'])



def test_original_comparison_floors_remain_refusals():
    assert cli.compare(1e-11,2e-10,1e-10,.4)['disposition']=='unresolved_NONPASS'
    assert not cli.compare(1e-11,2e-10,1e-10,.4)['passes']
    assert not cli.compare(1.,.41,1e-10,.4)['passes']
    assert cli.compare(1e-11,1e-11,1e-10,.4)['disposition']=='resolution_limited'
    assert not cli.compare(1.,.751,1e-6,.75)['passes']



def _approval_fixture():
    raw=cli.APPROVAL_PATH.read_bytes();record=json.loads(raw)
    # Exact observed bits, not a rounded decimal or the independent/cumulative value.
    value=float.fromhex('0x1.2664e8617a95bp-20')
    assert value==record['accepted_measurement']['normalized_step_energy']
    return raw,record,dict(record['binding']),value


def test_scoped_energy_acceptance_exact_once_and_strict_passes(monkeypatch):
    raw,record,facts,value=_approval_fixture();calls=[]
    monkeypatch.setattr(cli,'_measured_acceptance_scope',lambda approved:calls.append(approved['id'])or {'unit_control_only':True})
    strict=cli._EnergyAcceptance()
    # A strict pass needs neither approval/host eligibility nor the target slot.
    with monkeypatch.context()as m:
        m.setattr(Path,'read_bytes',lambda path:(_ for _ in ()).throw(AssertionError('strict pass must not read approval')))
        assert not strict.check(1e-6,facts)['exception_used']
        assert not strict.check(0.,{'label':'fine','epoch':99})['exception_used']
        assert strict.finish()['strict_original_energy_pass']is True
    assert not calls
    gate=cli._EnergyAcceptance();verdict=gate.check(value,facts)
    assert verdict['exception_used']and verdict['normalized_step_energy_hex']==value.hex()
    assert verdict['approval_id']==record['id'] and len(calls)==1
    with pytest.raises(cli.EnergyAcceptanceError,match='already consumed'):gate.check(value,facts)
    with pytest.raises(cli.EnergyAcceptanceError):gate.check(value,dict(facts,label='fine',epoch=3))
    assert gate.finish()['exception_uses']==1
    assert not gate.finish()['strict_original_energy_pass']


def test_scoped_energy_acceptance_refuses_binding_bits_and_mutated_record(monkeypatch):
    raw,record,facts,value=_approval_fixture();calls=[]
    monkeypatch.setattr(cli,'_measured_acceptance_scope',lambda approved:calls.append(approved)or {})
    # Each independent case/state/scale component is consequential; every failure
    # must precede runtime eligibility and must not consume the sole exception.
    for key,old in facts.items():
        bad=dict(facts)
        if type(old)is int:bad[key]=old+1
        elif type(old)is float:bad[key]=math.nextafter(old,math.inf)
        elif key.endswith('_hex'):bad[key]=math.nextafter(float.fromhex(old),math.inf).hex()
        elif key=='label':bad[key]='fine'
        else:bad[key]=('0'if old[0]!='0'else'1')+old[1:]
        assert bad[key]!=old
        gate=cli._EnergyAcceptance()
        with pytest.raises(cli.EnergyAcceptanceError):gate.check(value,bad)
        assert gate.finish()['exception_uses']==0
    for bad in [dict(facts,extra='unapproved'),{k:v for k,v in facts.items()if k!='context_sha256'}]:
        with pytest.raises(cli.EnergyAcceptanceError):cli._EnergyAcceptance().check(value,bad)
    swapped=dict(facts);swapped['initial_state_sha256'],swapped['endpoint_state_sha256']=facts['endpoint_state_sha256'],facts['initial_state_sha256']
    with pytest.raises(cli.EnergyAcceptanceError):cli._EnergyAcceptance().check(value,swapped)
    for wrong in (math.nextafter(value,math.inf),math.nextafter(value,-math.inf),
                  record['independent_corroboration']['normalized_physical_step_energy'],float('nan'),float('inf'),-1.,True):
        with pytest.raises(cli.EnergyAcceptanceError):cli._EnergyAcceptance().check(wrong,facts)
    assert not calls
    for badraw in (raw+b' ',json.dumps(dict(record,id='swapped-approval')).encode(),b'{}'):
        with pytest.raises(cli.EnergyAcceptanceError):cli._approval_record(badraw)
    # Real check() must reread the fixed record after construction, not trust
    # cached constructor data. No approved file is edited by this test.
    gate=cli._EnergyAcceptance();read=Path.read_bytes
    with monkeypatch.context()as m:
        m.setattr(Path,'read_bytes',lambda path:raw+b' 'if path==cli.APPROVAL_PATH else read(path))
        with pytest.raises(cli.EnergyAcceptanceError):gate.check(value,facts)
    assert gate.finish()['exception_uses']==0


def test_scoped_energy_acceptance_refuses_host_source_and_failed_eligibility(monkeypatch):
    raw,record,facts,value=_approval_fixture();capsule=record['observed_runtime_profile']
    actual=dict(platform=capsule['platform'],versions=copy.deepcopy(capsule['versions']),pools=copy.deepcopy(capsule['pools']),
        settings=dict(capsule['settings']),hash0=True,assertions=True,jit_off=True,numba_threads=True,
        natural_deadline=True,default_cpu=True,no_injected_plugins=True,
        product_python_manifest_sha256=record['scope']['product_python_manifest_sha256'])
    assert cli._validate_acceptance_scope(actual,record)is actual
    mutations=[]
    for key in ('hash0','assertions','jit_off','numba_threads','natural_deadline','default_cpu','no_injected_plugins'):
        bad=copy.deepcopy(actual);bad[key]=False;mutations.append(bad)
    bad=copy.deepcopy(actual);bad['platform']='linux';mutations.append(bad)
    bad=copy.deepcopy(actual);bad['versions']['numpy']='other';mutations.append(bad)
    bad=copy.deepcopy(actual);bad['settings']['NUMBA_DISABLE_JIT']='0';mutations.append(bad)
    bad=copy.deepcopy(actual);bad['pools'][0]['architecture']='Haswell';mutations.append(bad)
    bad=copy.deepcopy(actual);bad['pools'][0]['num_threads']=2;mutations.append(bad)
    bad=copy.deepcopy(actual);bad['product_python_manifest_sha256']='0'*64;mutations.append(bad)
    for bad in mutations:
        with pytest.raises(cli.EnergyAcceptanceError):cli._validate_acceptance_scope(bad,record)
    def decline(approved):raise cli.EnergyAcceptanceError('actual source/profile declined')
    monkeypatch.setattr(cli,'_measured_acceptance_scope',decline)
    gate=cli._EnergyAcceptance()
    with pytest.raises(cli.EnergyAcceptanceError,match='declined'):gate.check(value,facts)
    assert gate.finish()['exception_uses']==0


def _case_approval_fixture():
    raw=cli.CASE_APPROVAL_PATH.read_bytes();record=cli._case_approval_record(raw)
    measured=record['measured_velocity_exception']
    return raw,record,dict(measured['binding']),float.fromhex(measured['approved_observed_hex'])


def test_case_energy_step_and_cumulative_boundaries_and_strict_compatibility(monkeypatch):
    raw,record,facts,value=_case_approval_fixture()
    def unexpected(approved):raise AssertionError('case energy is not a measured-host exception')
    monkeypatch.setattr(cli,'_measured_acceptance_scope',unexpected)
    gate=cli._CaseAcceptance()
    with monkeypatch.context()as m:
        m.setattr(Path,'read_bytes',lambda path:(_ for _ in ()).throw(AssertionError('strict pass read approval')))
        assert not gate.check_energy(1e-6,1e-6,{})['acceptance_used']
        assert not gate.check_velocity(.002,{})['exception_used']
        assert gate.finish()['strict_original_gates_pass']
    # Each energy channel is independently admitted; neither changes P/H.
    for step,cumulative in ((2e-6,0.),(0.,2e-6),(2e-6,2e-6),
                            (math.nextafter(2e-6,-math.inf),math.nextafter(2e-6,-math.inf)),
                            (math.nextafter(1e-6,math.inf),1e-6)):
        for row in record['prescribed_workload']:
            current=dict(facts,**row,epoch=row['steps'])
            assert gate.check_energy(step,cumulative,current)['acceptance_used']
    for step,cumulative in ((math.nextafter(2e-6,math.inf),0.),(0.,math.nextafter(2e-6,math.inf))):
        with pytest.raises(cli.EnergyAcceptanceError):gate.check_energy(step,cumulative,facts)
    assert gate.finish()['velocity_exception_uses']==0
    assert gate.finish()['case_energy_uses']==15
    for wrong in (float('nan'),float('inf'),-float('inf'),-1.,True,1,'0.0',None):
        with pytest.raises(cli.EnergyAcceptanceError):gate.check_energy(wrong,0.,facts)
        with pytest.raises(cli.EnergyAcceptanceError):gate.check_energy(0.,wrong,facts)


def test_case_energy_refuses_wrong_case_workload_types_and_record_mutation(monkeypatch):
    raw,record,facts,value=_case_approval_fixture();gate=cli._CaseAcceptance()
    for bad in (dict(facts,definition_sha256='0'*64),dict(facts,profile_sha256='0'*64),
                dict(facts,label='other'),dict(facts,amplitude=5e-5),dict(facts,steps=8),
                dict(facts,epoch=0),dict(facts,epoch=5),dict(facts,epoch=True),
                dict(facts,steps=4.),dict(facts,amplitude='0.0001'),dict(facts,energy0_hex='nan'),
                dict(facts,energy0_hex='0x1p+1024'),dict(facts,dt_hex='0x1p+1024'),dict(facts,time_hex='0x1p+1024'),
                dict(facts,endpoint_state_sha256='unbound'),dict(facts,extra=True),
                {k:v for k,v in facts.items()if k!='context_sha256'}):
        with pytest.raises(cli.EnergyAcceptanceError):gate.check_energy(2e-6,0.,bad)
    assert gate.finish()['case_energy_uses']==0
    # The threshold admits future states of this case, not only historical states.
    assert gate.check_energy(2e-6,0.,dict(facts,endpoint_state_sha256='0'*64))['acceptance_used']
    for badraw in (raw+b' ',json.dumps(dict(record,id='other')).encode(),b'{}'):
        with pytest.raises(cli.EnergyAcceptanceError):cli._case_approval_record(badraw)
    read=Path.read_bytes
    with monkeypatch.context()as m:
        m.setattr(Path,'read_bytes',lambda path:raw+b' 'if path==cli.CASE_APPROVAL_PATH else read(path))
        with pytest.raises(cli.EnergyAcceptanceError):gate.check_energy(2e-6,0.,facts)
        with pytest.raises(cli.EnergyAcceptanceError):gate.finish()


def test_case_measured_velocity_exact_once_and_every_binding(monkeypatch):
    raw,record,facts,value=_case_approval_fixture();calls=[]
    monkeypatch.setattr(cli,'_measured_acceptance_scope',lambda approved:calls.append(approved['id'])or {'unit_control_only':True})
    assert value==0.0020142575736402882
    gate=cli._CaseAcceptance();verdict=gate.check_velocity(value,facts)
    assert verdict['exception_used']and verdict['normalized_velocity_hex']=='0x1.080344cf1b534p-9'
    assert verdict['binding']==facts and len(calls)==1
    with pytest.raises(cli.EnergyAcceptanceError,match='already consumed'):gate.check_velocity(value,facts)
    for key,old in facts.items():
        bad=dict(facts)
        if type(old)is int:bad[key]=old+1
        elif type(old)is float:bad[key]=math.nextafter(old,math.inf)
        elif key.endswith('_hex'):bad[key]=math.nextafter(float.fromhex(old),math.inf).hex()
        else:bad[key]='other'
        with pytest.raises(cli.EnergyAcceptanceError):cli._CaseAcceptance().check_velocity(value,bad)
    for bad in (dict(facts,epoch=3.),dict(facts,steps=True),dict(facts,extra=True),
                {k:v for k,v in facts.items()if k!='context_sha256'}):
        with pytest.raises(cli.EnergyAcceptanceError):cli._CaseAcceptance().check_velocity(value,bad)
    for wrong in (math.nextafter(value,math.inf),math.nextafter(value,-math.inf),
                  math.nextafter(.002,math.inf),.0020142576,float('nan'),float('inf'),-1.,True,0, None):
        with pytest.raises(cli.EnergyAcceptanceError):cli._CaseAcceptance().check_velocity(wrong,facts)
    assert len(calls)==1
    assert gate.finish()['velocity_exception_uses']==1
    read=Path.read_bytes
    with monkeypatch.context()as m:
        m.setattr(Path,'read_bytes',lambda path:raw+b' 'if path==cli.CASE_APPROVAL_PATH else read(path))
        with pytest.raises(cli.EnergyAcceptanceError):cli._CaseAcceptance().check_velocity(value,facts)
        with pytest.raises(cli.EnergyAcceptanceError):gate.finish()


def test_case_velocity_refuses_source_host_runtime_and_keeps_unused_ledger(monkeypatch):
    raw,record,facts,value=_case_approval_fixture();profile=record['observed_runtime_profile']
    actual=dict(platform=profile['platform'],versions=copy.deepcopy(profile['versions']),pools=copy.deepcopy(profile['pools']),
        settings=dict(profile['settings']),hash0=True,assertions=True,jit_off=True,numba_threads=True,
        natural_deadline=True,default_cpu=True,no_injected_plugins=True,
        product_python_manifest_sha256=record['scope']['product_python_manifest_sha256'])
    assert cli._validate_acceptance_scope(actual,record)is actual
    mutations=[]
    for key in ('hash0','assertions','jit_off','numba_threads','natural_deadline','default_cpu','no_injected_plugins'):
        bad=copy.deepcopy(actual);bad[key]=False;mutations.append(bad)
    for key in ('platform','product_python_manifest_sha256'):
        bad=copy.deepcopy(actual);bad[key]='other';mutations.append(bad)
    for key in profile['versions']:
        bad=copy.deepcopy(actual);bad['versions'][key]='other';mutations.append(bad)
    for key in profile['settings']:
        bad=copy.deepcopy(actual);bad['settings'][key]='other';mutations.append(bad)
    for key in profile['pools'][0]:
        bad=copy.deepcopy(actual);bad['pools'][0][key]='other';mutations.append(bad)
    bad=copy.deepcopy(actual);bad['pools']=[];mutations.append(bad)
    for bad in mutations:
        with pytest.raises(cli.EnergyAcceptanceError):cli._validate_acceptance_scope(bad,record)
    def decline(approved):raise cli.EnergyAcceptanceError('actual source/profile declined')
    monkeypatch.setattr(cli,'_measured_acceptance_scope',decline)
    gate=cli._CaseAcceptance()
    with pytest.raises(cli.EnergyAcceptanceError,match='declined'):gate.check_velocity(value,facts)
    assert gate.finish()['velocity_exception_uses']==0


def test_case_caller_admission_order_and_unchanged_other_gates():
    # Static caller integration only: this control performs no mechanics.
    tree=ast.parse(Path(cli.__file__).read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    source=ast.unparse(run)
    energy=[n for n in ast.walk(run)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='check_energy']
    velocity=[n for n in ast.walk(run)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='check_velocity']
    assert len(energy)==len(velocity)==1 and energy[0].lineno<velocity[0].lineno
    assert ast.unparse(energy[0])=="case_acceptance.check_energy(float(step['energy']), float(cumulative['energy']), facts)"
    assert ast.unparse(velocity[0])=="case_acceptance.check_velocity(float(errors['velocity_discrete_midpoint']), facts)"
    assert "errors['configuration_discrete_midpoint'] <= 0.002"in source
    assert "max((cumulative[k] for k in ('linear', 'angular'))) <= 1e-06"in source
    assert "[('coarse', 0.0001, 4), ('fine', 0.0001, 8), ('half_fine', 5e-05, 8)]"in source
    assert "for target in (2, 3):"in source
    assert "assert evidence['scientific_endpoints'] == 20"in source
    assert "1e-10, 0.4"in source and "1e-06, 0.75"in source
    assert cli.CRITERIA['nonlinear_to_discrete_limit']==.002
    assert cli.CRITERIA['normalized_defect_limit']==1e-6
    assert cli.CRITERIA['case_step_energy_limit']==3.5e-6 and cli.CRITERIA['case_cumulative_energy_limit']==2.5e-6
    assert cli.CRITERIA['case_velocity_limit']==.005
    assert cli.CRITERIA['internal_time_limit_seconds']==120.


def _envelope_fixture():
    raw=cli.ENVELOPE_APPROVAL_PATH.read_bytes();record=cli._envelope_approval_record(raw)
    return raw,record,dict(record['binding_template'])


def test_case_envelope_boundaries_strict_compatibility_and_usage(monkeypatch):
    raw,record,facts=_envelope_fixture();calls=[]
    monkeypatch.setattr(cli,'_measured_envelope_scope',lambda approved:calls.append(approved['id'])or {'unit_control_only':True})
    strict=cli._CaseEnvelopeAcceptance()
    with monkeypatch.context()as m:
        m.setattr(Path,'read_bytes',lambda p:(_ for _ in ()).throw(AssertionError('strict original pass reads no approval')))
        assert not strict.check_energy(1e-6,1e-6,{})['acceptance_used']
        assert not strict.check_velocity(.002,{})['acceptance_used']
        assert strict.finish()['strict_original_gates_pass']
    assert not calls
    for step,cumulative in ((3.5e-6,0.),(0.,2.5e-6),(3.5e-6,2.5e-6),
                            (math.nextafter(3.5e-6,-math.inf),math.nextafter(2.5e-6,-math.inf))):
        gate=cli._CaseEnvelopeAcceptance();assert gate.check_energy(step,cumulative,facts)['acceptance_used']
        assert gate.finish()['case_energy_uses']==1
    for value in (.005,math.nextafter(.005,-math.inf),math.nextafter(.002,math.inf),.0020142576):
        assert cli._CaseEnvelopeAcceptance().check_velocity(value,facts)['acceptance_used']
    for step,cumulative in ((math.nextafter(3.5e-6,math.inf),0.),(0.,math.nextafter(2.5e-6,math.inf))):
        with pytest.raises(cli.EnergyAcceptanceError):cli._CaseEnvelopeAcceptance().check_energy(step,cumulative,facts)
    with pytest.raises(cli.EnergyAcceptanceError):cli._CaseEnvelopeAcceptance().check_velocity(math.nextafter(.005,math.inf),facts)
    gate=cli._CaseEnvelopeAcceptance()
    for row in record['prescribed_workload']:
        for epoch in range(1,row['steps']+1):
            current=dict(facts,**row,epoch=epoch)
            assert gate.check_energy(3.5e-6,2.5e-6,current)['acceptance_used']
            assert gate.check_velocity(.005,current)['acceptance_used']
    result=gate.finish();assert result['case_energy_uses']==result['case_velocity_uses']==20
    assert result['outcome']=='PASS_WITH_SCOPED_HUMAN_CASE_ENVELOPE_ACCEPTANCE_ONLY'
    for method,args in ((gate.check_energy,(3.5e-6,2.5e-6,facts)),(gate.check_velocity,(.005,facts))):
        with pytest.raises(cli.EnergyAcceptanceError,match='already admitted'):method(*args)
    assert gate.finish()==result


def test_case_envelope_refuses_nonfinite_wrongcase_and_mutated_capsule(monkeypatch):
    raw,record,facts=_envelope_fixture();calls=[]
    monkeypatch.setattr(cli,'_measured_envelope_scope',lambda approved:calls.append(approved)or {})
    for wrong in (float('nan'),float('inf'),-float('inf'),-1.,True,0,'0',None):
        with pytest.raises(cli.EnergyAcceptanceError):cli._CaseEnvelopeAcceptance().check_energy(wrong,0.,facts)
        with pytest.raises(cli.EnergyAcceptanceError):cli._CaseEnvelopeAcceptance().check_energy(0.,wrong,facts)
        with pytest.raises(cli.EnergyAcceptanceError):cli._CaseEnvelopeAcceptance().check_velocity(wrong,facts)
    badfacts=[dict(facts,definition_sha256='0'*64),dict(facts,profile_sha256='0'*64),dict(facts,label='other'),
        dict(facts,amplitude=5e-5),dict(facts,steps=8),dict(facts,epoch=0),dict(facts,epoch=5),dict(facts,epoch=True),
        dict(facts,extra=True),{k:v for k,v in facts.items()if k!='context_sha256'}]
    for key,old in facts.items():
        bad=dict(facts);bad[key]=None;badfacts.append(bad)
    for key in ('dt_hex','time_hex','energy0_hex'):
        for wrong in ('0x1p+1024','nan','-0x1p+0','0x0p+0'):
            bad=dict(facts);bad[key]=wrong;badfacts.append(bad)
    for bad in badfacts:
        with pytest.raises(cli.EnergyAcceptanceError):cli._CaseEnvelopeAcceptance().check_energy(3.5e-6,2.5e-6,bad)
        with pytest.raises(cli.EnergyAcceptanceError):cli._CaseEnvelopeAcceptance().check_velocity(.005,bad)
    assert not calls
    # This is an approved case envelope: new well-formed endpoint hashes are admitted.
    assert cli._CaseEnvelopeAcceptance().check_velocity(.005,dict(facts,endpoint_state_sha256='0'*64))['acceptance_used']
    for bad in (raw+b' ',b'{}',json.dumps(dict(record,id='other')).encode(),bytearray(raw)):
        with pytest.raises(cli.EnergyAcceptanceError):cli._envelope_approval_record(bad)
    gate=cli._CaseEnvelopeAcceptance();gate.check_velocity(.005,facts);read=Path.read_bytes
    with monkeypatch.context()as m:
        m.setattr(Path,'read_bytes',lambda p:raw+b' 'if p==cli.ENVELOPE_APPROVAL_PATH else read(p))
        with pytest.raises(cli.EnergyAcceptanceError):cli._CaseEnvelopeAcceptance().check_energy(3.5e-6,2.5e-6,facts)
        with pytest.raises(cli.EnergyAcceptanceError):cli._CaseEnvelopeAcceptance().check_velocity(.005,facts)
        with pytest.raises(cli.EnergyAcceptanceError):gate.finish()
    assert gate.finish()['case_velocity_uses']==1


def test_case_envelope_actual_scope_refusal_preserves_usage_and_reference(monkeypatch):
    raw,record,facts=_envelope_fixture()
    assert 'observed_runtime_profile'not in record
    actual=dict(product_python_count=700,product_python_manifest_sha256=record['scope']['product_python_manifest_sha256'],
        physical_reference_sha256=record['scope']['physical_reference_sha256'])
    assert cli._validate_envelope_source(actual,record)is actual
    for key,wrong in (('product_python_count',701),('product_python_count',True),
                      ('product_python_manifest_sha256','0'*64),('physical_reference_sha256','0'*64)):
        bad=dict(actual);bad[key]=wrong
        with pytest.raises(cli.EnergyAcceptanceError):cli._validate_envelope_source(bad,record)
    for bad in (dict(actual,extra=True),{k:v for k,v in actual.items()if k!='physical_reference_sha256'},None):
        with pytest.raises(cli.EnergyAcceptanceError):cli._validate_envelope_source(bad,record)
    calls=[]
    monkeypatch.setattr(cli,'_measure_envelope_source',lambda:calls.append('actual source measurement hook')or actual)
    assert cli._measured_envelope_scope(record)is actual and len(calls)==1
    with monkeypatch.context()as m:
        m.setattr(cli,'_measure_envelope_source',lambda:dict(actual,physical_reference_sha256='0'*64))
        with pytest.raises(cli.EnergyAcceptanceError):cli._measured_envelope_scope(record)
    def decline(approved):raise cli.EnergyAcceptanceError('actual source/reference refused')
    monkeypatch.setattr(cli,'_measured_envelope_scope',decline);gate=cli._CaseEnvelopeAcceptance()
    with pytest.raises(cli.EnergyAcceptanceError,match='refused'):gate.check_energy(3.5e-6,2.5e-6,facts)
    with pytest.raises(cli.EnergyAcceptanceError,match='refused'):gate.check_velocity(.005,facts)
    assert gate.finish()['case_energy_uses']==gate.finish()['case_velocity_uses']==0
    monkeypatch.setattr(cli,'_measured_envelope_scope',lambda approved:actual)
    assert gate.check_energy(3.5e-6,2.5e-6,facts)['acceptance_used']
    assert gate.check_velocity(.005,facts)['acceptance_used']
    assert gate.finish()['case_energy_uses']==gate.finish()['case_velocity_uses']==1


def test_canonical_primitives_exact_definition_profile_source_and_refusals(tmp_path,record_property):
    """Minimal identity discriminator: public reconstruction, never a trajectory."""
    points,frames,section,inertia,definition=cli.fixture()
    ip,ifr,ic,im=ref.primitive_inputs()
    # Preserve actual issued definition/input evidence even if an exact identity refuses.
    attempt_dir=Path(os.environ['D6_NATIVE_VIBRATION_OUTPUT']).parent if 'D6_NATIVE_VIBRATION_OUTPUT'in os.environ else tmp_path
    attempt=dict(status='OBSERVATION_ONLY_UNQUALIFIED',trajectory_executed=False,
        definition_raw_hex=definition.raw.hex(),definition_sha256=definition.sha256,
        profile_hex=cli.PROFILE_BYTES.hex(),profile_sha256=cli.sha256(cli.PROFILE_BYTES).hexdigest(),
        caller_triads_binary64_hex=[[[float(v).hex()for v in row]for row in triad]for triad in frames],
        independent_triads_binary64_hex=[[[float(v).hex()for v in row]for row in triad]for triad in ifr],
        observed_reference_sha256=cli.sha256(Path(ref.__file__).read_bytes()).hexdigest(),
        scientific_acceptance=False,full_C7_acceptance=False)
    (attempt_dir/'CANONICAL_IDENTITY_ATTEMPT.json').write_text(json.dumps(attempt,indent=2)+'\n',encoding='utf-8')
    assert np.array_equal(points,ip)and frames.tobytes()==ifr.tobytes()
    assert np.array_equal(section.elastic,ic)and np.array_equal(inertia,im)
    assert definition.sha256=='9e087b9ebf541889a6eb217880c8139990c40a0b0b15bd62fb9805f8414c13ae'
    assert cli.sha256(cli.PROFILE_BYTES).hexdigest()=='feeee50769b637a4c08b5a961dd313341ab4113f41377530723569ac94666454'
    from anyfesolver.ge_beam3_native import BeamDefinition
    rebuilt=BeamDefinition.from_bytes(definition.raw,expected_sha256=definition.sha256)
    assert rebuilt.raw==definition.raw
    raw=cli.ENVELOPE_APPROVAL_PATH.read_bytes();record=cli._envelope_approval_record(raw)
    actual=cli._measured_envelope_scope(record)
    assert actual['physical_reference_sha256']==cli.sha256(Path(ref.__file__).read_bytes()).hexdigest()
    refusals=[]
    with pytest.raises(cli.EnergyAcceptanceError):cli._envelope_approval_record(raw+b' ')
    refusals.append('mutated_capsule')
    for key in ('definition_sha256','profile_sha256'):
        facts=dict(record['binding_template']);facts[key]='0'*64
        with pytest.raises(cli.EnergyAcceptanceError):cli._envelope_case_binding(facts,record)
        refusals.append('wrong_'+key)
    with pytest.raises(ValueError):BeamDefinition.from_bytes(definition.raw,expected_sha256='0'*64)
    refusals.append('wrong_external_definition_hash')
    malformed=definition.raw+b' '
    with pytest.raises(ValueError):BeamDefinition.from_bytes(malformed,expected_sha256=cli.sha256(malformed).hexdigest())
    refusals.append('noncanonical_definition')
    result=dict(status='PASS_CANONICAL_PRIMITIVE_IDENTITY_ONLY',trajectory_executed=False,
        definition_raw_hex=definition.raw.hex(),definition_sha256=definition.sha256,
        profile_hex=cli.PROFILE_BYTES.hex(),profile_sha256=cli.sha256(cli.PROFILE_BYTES).hexdigest(),
        primitive_triads_binary64_hex=[[[float(v).hex()for v in row]for row in triad]for triad in frames],
        primitive_correspondence_exact=True,public_reconstruction_exact=True,source=actual,
        approval_sha256=cli.ENVELOPE_APPROVAL_SHA256,technical_source_binding_sha256=record['technical_source_binding_sha256'],
        refusals=refusals,scientific_acceptance=False,full_C7_acceptance=False)
    output=(Path(os.environ['D6_NATIVE_VIBRATION_OUTPUT']).parent if 'D6_NATIVE_VIBRATION_OUTPUT'in os.environ else tmp_path)/'CANONICAL_IDENTITY.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    record_property('canonical_identity',str(output))


def test_public_authentic_full_amplitude_vibration_cli(tmp_path,monkeypatch,record_property):
    output=Path(os.environ['D6_NATIVE_VIBRATION_OUTPUT'])if 'D6_NATIVE_VIBRATION_OUTPUT'in os.environ else tmp_path/'vibration'
    monkeypatch.setattr(sys,'argv',['native_full_amplitude_vibration_dev','--output',str(output)])
    cli.main()
    result=json.loads((output/'ASSESSMENT.json').read_text())
    acceptance=result['case_envelope_acceptance']
    energies=result.get('approved_case_energy_uses',[]);velocities=result.get('approved_case_velocity_uses',[])
    assert type(acceptance['case_energy_uses'])is int and acceptance['case_energy_uses']==len(energies)
    assert type(acceptance['case_velocity_uses'])is int and 0<=acceptance['case_velocity_uses']<=20
    assert acceptance['case_velocity_uses']==len(velocities)
    if acceptance['strict_original_gates_pass']:
        assert result['status']=='PASS_CURRENT_PUBLIC_VIBRATION_DEVELOPMENT_ONLY'
        assert not energies and not velocities
    else:
        assert result['status']=='PASS_CURRENT_PUBLIC_VIBRATION_WITH_SCOPED_CASE_ENVELOPE_ACCEPTANCE_ONLY'
        assert acceptance['outcome']=='PASS_WITH_SCOPED_HUMAN_CASE_ENVELOPE_ACCEPTANCE_ONLY'
        assert acceptance['approval_sha256']==cli.ENVELOPE_APPROVAL_SHA256
        assert energies or velocities
    assert acceptance['scientific_acceptance']is False and acceptance['full_C7_acceptance']is False
    for verdict in velocities:
        assert .002<float.fromhex(verdict['normalized_velocity_hex'])<=.005
        cli._envelope_case_binding(verdict['binding'],_envelope_fixture()[1])
    for row in result['resolutions'].values():
        for endpoint in row['endpoints']:
            assert endpoint['energy_verdict']['status']in ('STRICT_ORIGINAL_ENERGY_PASS','HUMAN_APPROVED_CASE_ENVELOPE_ENERGY_PASS')
            assert endpoint['velocity_verdict']['status']in ('STRICT_ORIGINAL_VELOCITY_PASS','HUMAN_APPROVED_CASE_ENVELOPE_VELOCITY_PASS')
    assert result['scientific_endpoints']==20 and result['replay_increments']==2
    assert all(row['passes']for row in result['comparisons'].values())
    record_property('assessment',str(output/'ASSESSMENT.json'))
    record_property('scientific_endpoints',20);record_property('additional_fresh_replay_increments',2)
