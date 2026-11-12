"""Frozen fifteen-node numerical recovery inventory; bounded runner only.

Every node is self-contained. Reference matrices are reused only inside a node;
no scientific result from a different child is an input to this module.
"""
import ast
from dataclasses import dataclass, fields, is_dataclass, replace, FrozenInstanceError
from hashlib import sha256
import json
import math
from itertools import permutations
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
import os
from pathlib import Path
import sys
from threading import Event, Thread
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'docs/reference_cases'))
from anyfesolver import _ge_beam3_g3c_affine_q4_recovery as candidate
from anyfesolver import _ge_beam3_g3c_affine_q4_registry as registry
import ge_beam3_q4_affine_numerical_checker as checker
from anyfesolver import _ge_beam3_g3c_affine_q4_chart as stable_chart
from anyfesolver import _ge_beam3_g3c_affine_q4_increment_chart as increment_chart
from anyfesolver import _ge_beam3_g3c_so3_numerics as stable_so3
from anyfesolver import _ge_beam3_variational_shell as original_fit
from anyfesolver import _ge_beam3_mixed_ad as original_ad
from anyfesolver import _ge_beam3_g3c_local_shell as original_shell
from anyfesolver import _ge_beam3_pose_joint as original_joint

SCIENTIFIC_RECORDS=[]
CONTRADICTIONS=[]
_FACADES={}
_RESULTS={}
_EXPECTED={}
_OBSERVATIONS={}
_CURRENT_OBSERVATION=None
SOURCES={
 'src/anyfesolver/e4_pl_element.py':'7fc46a18046e043512a5fb2c3f61ed0b95b834e4dde764f63c500a885e87ea38',
 'src/anyfesolver/elements.py':'f8c59792947a3c9b84416c61a4d9db98e42926d1bf21e74e736afbf6a9a88b37',
 'src/anyfesolver/_ge_beam3_g3c_local_shell.py':'69d9f27a96ed7e9a5959a42a081eeafbbe72021de6e4da1f91355c5fb3850f10',
 'src/anyfesolver/_ge_beam3_variational_shell.py':'b0db7a83633f4a835c06e940a6f0de36ab958f7e816c37a23c6ebc16259a2a60',
 'src/anyfesolver/_ge_beam3_mixed_ad.py':'b299ff765cd2eaae8b33a2ba1d05069f6fe8c39209e1eac75df356a2afb1fe36',
 'src/anyfesolver/_ge_beam3_pose_joint.py':'69cfb0a71761ab7cad0d62080d81511949e7c920f7f70162bcaba0497b402657',
 'src/anyfesolver/_ge_beam3_g3c_definition.py':'4b46b870fec010e3378df83c374d763f858d8f25820c14f9704dc6f0a656f0c2',
 'src/anyfesolver/_ge_beam3_g3c_owner.py':'18b9565d56192e23f192b1a3fbae3cb12ed7ede958c0e4b3e46279a799d1afc5',
 'docs/reference_cases/ge_beam3_g3c_fixtures_v1.json':'d5fecc80c3203ac7d191b4031d64bf35c56d2900066cb01fc1dfa4276d24c006'}


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('ascii')


def array_digest(value):
    a=np.asarray(value,dtype=np.float64)
    assert np.isfinite(a).all()
    return dict(shape=list(a.shape),sha256=sha256(a.astype('<f8',copy=False).tobytes()).hexdigest())


def norm(value):
    a=np.asarray(value,dtype=np.float64)
    assert np.isfinite(a).all()
    largest=float(np.max(np.abs(a))) if a.size else 0.
    return 0. if largest==0. else largest*float(np.linalg.norm(a/largest))


def close(actual,expected,*,scale=None,tolerance=1e-11):
    a=np.asarray(actual,dtype=np.float64);b=np.asarray(expected,dtype=np.float64)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    denominator=norm(b) if scale is None else float(scale)
    assert math.isfinite(denominator) and denominator>=0
    difference=norm(a-b)
    if denominator==0:
        assert difference==0
        return 0.
    ratio=difference/denominator
    assert ratio<=tolerance,(ratio,tolerance,array_digest(a),array_digest(b))
    return ratio


def length(construction):
    X=construction.coordinates
    return max(norm(X[(i+1)%4]-X[i]) for i in range(4))


def scale_vector(construction):
    return np.tile([length(construction)]*3+[1.]*3,4)


def facade(identity):
    if identity not in _FACADES:_FACADES[identity]=candidate.AffineQ4PhysicalRecovery(identity)
    return _FACADES[identity]


def context(identity,pose='MIXED',amplitude=1.):
    return registry.construction(identity),*registry.pose(identity,pose,amplitude)


def evaluate(c,q,accepted):
    key=(c.construction_id,np.asarray(q).tobytes(),np.asarray(accepted).tobytes())
    if key not in _RESULTS:
        print('BEAM CHECKPOINT recovery '+c.construction_id,flush=True)
        _RESULTS[key]=facade(c.construction_id).evaluate(q,accepted)
    return _RESULTS[key]


def independent(c,q,accepted):
    key=(c.construction_id,np.asarray(q).tobytes(),np.asarray(accepted).tobytes())
    if key not in _EXPECTED:
        _EXPECTED[key]=checker.evaluate(c.coordinates,q,accepted,normal=c.normal,
            material_direction=c.material_direction,director_polarity=c.director_polarity)
    return _EXPECTED[key]


def base_contexts():
    for identity in registry.base_ids():
        for pose in registry.POSES:yield identity,pose,*context(identity,pose)


def check_chart(c,trial,expected):
    scales=scale_vector(c);inverse=1/scales
    kin=trial.kinematics
    zero_pose=(np.array_equal(kin.current_positions,c.coordinates)
               and np.array_equal(kin.rotations,np.tile(np.eye(3),(4,1,1))))
    reference_scale=norm((c.coordinates-c.coordinates.mean(axis=0))/length(c))
    return dict(
        d=close(inverse*kin.deformation,inverse*expected['d'],
                scale=reference_scale if zero_pose else norm(inverse*expected['d'])),
        D=close(inverse[:,None]*kin.differential*scales[None,:],inverse[:,None]*expected['D']*scales[None,:]),
        D2=close(inverse[:,None,None]*kin.second*scales[None,:,None]*scales[None,None,:],
                 inverse[:,None,None]*expected['D2']*scales[None,:,None]*scales[None,None,:]),
        R=close(kin.frame,expected['R']),Q=close(kin.rotations,expected['Q']),x=close(kin.current_positions,expected['x']))


def physical_scales(c):
    q,qa=registry.pose(c.construction_id,'ZERO')
    zero=evaluate(c,q,qa);s=scale_vector(c)
    physical=norm(s[:,None]*zero.physical_chart_hessian*s[None,:])
    total=norm(s[:,None]*zero.chart_hessian*s[None,:])
    assert physical>0 and total>0
    return physical,total


def observe(table,row_id,c,q,qa):
    """Bind the exact inputs of an actual cached or fresh candidate evaluation."""
    global _CURRENT_OBSERVATION
    key=(c.construction_id,np.asarray(q).tobytes(),np.asarray(qa).tobytes())
    assert key in _RESULTS
    state=dict(construction_id=c.construction_id,coordinates=c.coordinates.tolist(),q=np.asarray(q).tolist(),
        accepted=np.asarray(qa).tolist(),normal=c.normal.tolist(),material_direction=c.material_direction.tolist(),
        director_polarity=c.director_polarity)
    locator=(table,row_id)
    assert locator not in _OBSERVATIONS
    value=dict(state=state,state_sha256=sha256(canonical(state)).hexdigest(),physical_checks={})
    _OBSERVATIONS[locator]=value;_CURRENT_OBSERVATION=locator


def save_physical_check(predicate,error,actual):
    assert _CURRENT_OBSERVATION is not None
    observation=_OBSERVATIONS[_CURRENT_OBSERVATION]
    assert predicate not in observation['physical_checks']
    result=dict(relative_error=error,passed=error<=1e-11,actual=array_digest(actual))
    observation['physical_checks'][predicate]=result
    return result


def physical_comparison(c,q,qa,predicate,actual,expected):
    """A verified physical mismatch is evidence, never a disguised pytest crash."""
    target=predicate.removeprefix('source_')
    scaled_actual=checker.physical_scaled(target,actual,c.coordinates)
    scaled_expected=checker.physical_scaled(target,expected,c.coordinates)
    error=checker.relative_error(scaled_actual,scaled_expected)
    table,row_id=_CURRENT_OBSERVATION
    observation=_OBSERVATIONS[(table,row_id)]
    assert observation['state']['construction_id']==c.construction_id
    if error>1e-11:
        directory=Path(os.environ['BEAM_QUALIFICATION_OUTPUT'])
        lease=json.loads((directory.parent/'lease.json').read_text())
        payload=dict(schema='Q4_AFFINE_NUMERICAL_CONTRADICTION_V1',predicate=predicate,
            coordinates=c.coordinates.tolist(),q=np.asarray(q).tolist(),accepted=np.asarray(qa).tolist(),
            normal=c.normal.tolist(),material_direction=c.material_direction.tolist(),director_polarity=c.director_polarity,
            actual=np.asarray(actual).tolist(),tolerance=1e-11,scale_mode='REFERENCE_EDGE_NONDIMENSIONAL_V1',
            source_identity=sha256(canonical(lease['inputs'])).hexdigest(),candidate_identity=lease['candidate']['commit'],
            fixture_identity=c.construction_id,node=os.environ['BEAM_QUALIFICATION_NODE'],table=table,row_id=row_id,
            state_sha256=observation['state_sha256'])
        # Independently reconstructed expected values and scales, not a caller's
        # fabricated expected operand. The runner verifies again before writing.
        verification=checker.verify_contradiction(payload)
        assert verification['accepted'] is True
        CONTRADICTIONS.append(payload)
    return save_physical_check(predicate,error,actual)


def check_physical(c,q,qa,trial,expected,*,table,row_id):
    observe(table,row_id,c,q,qa)
    s=scale_vector(c);zero,_=physical_scales(c)
    zero_pose=not np.any(q) and np.array_equal(qa,np.tile(np.eye(3),(4,1,1)))
    energy_scale=abs(float(expected['physical_energy']))
    if energy_scale==0 or zero_pose:energy_scale=zero
    force=s*expected['physical_force'];force_scale=zero if zero_pose else norm(force) or zero
    hessian=s[:,None]*expected['physical_hessian']*s[None,:]
    return dict(energy=(save_physical_check('physical_energy',close(trial.physical_energy,expected['physical_energy'],scale=energy_scale),trial.physical_energy) if zero_pose else
                       physical_comparison(c,q,qa,'physical_energy',trial.physical_energy,expected['physical_energy'])),
        force=(save_physical_check('physical_force',close(s*trial.physical_chart_force,force,scale=force_scale),trial.physical_chart_force) if zero_pose else
               physical_comparison(c,q,qa,'physical_force',trial.physical_chart_force,expected['physical_force'])),
        hessian=physical_comparison(c,q,qa,'physical_hessian',trial.physical_chart_hessian,expected['physical_hessian']),
        symmetry=close(s[:,None]*trial.physical_chart_hessian*s[None,:],(s[:,None]*trial.physical_chart_hessian*s[None,:]).T),
        spatial_force=close(s*trial.physical_spatial_force,s*expected['physical_spatial_force'],scale=force_scale),
        spatial_tangent=close(s[:,None]*trial.physical_spatial_row_chart_tangent*s[None,:],
                              s[:,None]*expected['physical_spatial_row_tangent']*s[None,:]))


def record(name,**tables):
    for table,rows in tables.items():
        assert rows and len({row['id'] for row in rows})==len(rows),table
        for row in rows:
            observation=_OBSERVATIONS.get((table,row['id']))
            if observation is not None:row.update(observation)
    SCIENTIFIC_RECORDS.append(dict(test=name,tables=tables,full_g3c_qualified=False,production_qualified=False))


def fingerprint_fixture():
    """Actual private encoding checks, including the bytes that blocked smoke."""
    @dataclass(frozen=True)
    class Packet:
        candidate: bytes
        children: tuple
    octets=b'\x00\xff\x80"\n\\'
    packet=Packet(octets,(b'accepted-state',np.array([0.,-0.])))
    values=[None,False,True,0,1,0.,-0.,1.,'','0',b'',b'0',octets,[1],(1,),
            {'kind':'bytes','value':octets.hex()},np.array([1.],dtype=np.float64),
            np.float64(1.),np.array([[1.]],dtype=np.float64),packet]
    digests=[candidate._fingerprint(value) for value in values]
    assert len(digests)==20 and len(set(digests))==20
    for value,digest in zip(values,digests):
        encoded=canonical(candidate._payload(value))
        assert sha256(encoded).hexdigest()==digest==candidate._fingerprint(value)
        assert canonical(json.loads(encoded))==encoded
    # A mapping that exactly resembles a byte tag must remain a mapping, not
    # collide with the bytes it resembles. This does not guess tag key names.
    assert candidate._fingerprint(octets)!=candidate._fingerprint(candidate._payload(octets))
    array=np.array([1.,2.]);snapshot=canonical(candidate._payload(array));prior=candidate._fingerprint(array)
    array[0]=3.
    assert snapshot!=canonical(candidate._payload(array)) and prior!=candidate._fingerprint(array)
    assert candidate._fingerprint(replace(packet,candidate=octets+b'\x00'))!=candidate._fingerprint(packet)
    assert candidate._fingerprint(replace(packet,children=(b'different-state',)))!=candidate._fingerprint(packet)
    invalid=(float('nan'),float('inf'),-float('inf'),np.array([float('nan')]),np.array([float('inf')]))
    for value in invalid:
        with pytest.raises((ValueError,TypeError)):candidate._fingerprint(value)
    return dict(id='typed_payload_encoding',distinct_fingerprints=20,nonfinite_rejections=5,
        evidence_sha256=sha256(canonical(digests)).hexdigest(),verified=True)


def chart_authority():
    assert candidate.CHART_NUMERICS_ID==increment_chart.NUMERICS_ID=='GE_BEAM3_Q4_AFFINE_INCREMENT_RESOLVED_CHART_NUMERICS_V1'
    assert candidate.deformation is increment_chart.deformation
    assert increment_chart.Jet2 is original_ad.Jet2
    assert increment_chart._exp_coefficients is stable_so3._exp_coefficients
    assert increment_chart._log_factor is stable_so3._log_factor
    assert increment_chart._polynomial is stable_so3._polynomial and increment_chart.LOG is stable_so3.LOG
    assert checker.chart.__name__=='ge_beam3_q4_affine_increment_chart'
    checker_ast=ast.parse(Path(checker.chart.__file__).read_text(encoding='utf-8'))
    for statement in ast.walk(checker_ast):
        if isinstance(statement,ast.Import):assert all(item.name in ('math','numpy','ge_beam3_q4_affine_numerical_chart') for item in statement.names)
        if isinstance(statement,ast.ImportFrom):assert statement.module=='decimal' and statement.level==0
    producer_ast=ast.parse(Path(increment_chart.__file__).read_text(encoding='utf-8'))
    for statement in ast.walk(producer_ast):
        if isinstance(statement,(ast.Import,ast.ImportFrom)):
            names=[item.name for item in statement.names] if isinstance(statement,ast.Import) else [statement.module or '']
            assert not any('checker' in name or 'reference_cases' in name for name in names)
    assert candidate._exp_terms is stable_chart._exp_terms
    assert stable_chart.rotation_jets is original_fit.rotation_jets
    assert stable_chart.Jet2 is original_ad.Jet2
    assert stable_chart.so3_exp is stable_so3.so3_exp and stable_chart.so3_log is stable_so3.so3_log
    assert stable_chart._exp_terms.__globals__['so3_exp'] is stable_so3.so3_exp
    assert candidate._spatial_connection.__globals__['_exp_terms'] is stable_chart._exp_terms
    assert registry._exp_terms is original_joint._exp_terms
    assert registry._exp_terms.__globals__['so3_exp'] is original_ad.so3_exp
    sources={
      'src/anyfesolver/_ge_beam3_g3c_affine_q4_chart.py':'2a26742ec7680a5e75934fb32ee8601d3877cd0befb250489f991f1b9d198a9f',
      'docs/reference_cases/ge_beam3_q4_affine_numerical_chart.py':'b7cc92f70d35e3b8f2ab10202cb2f3ad1116e92775425888e482293b323c23ac',
      'src/anyfesolver/_ge_beam3_g3c_so3_numerics.py':'588ace39570658f1a83c0110d3a019ebad770d6091b271ff4bbc40b97f96e12e',
      'docs/GE_BEAM3_G3C_SO3_NUMERICS_CONTRACT.md':'269bcd4e1fda151e7c22c396a70c43498e8450b222c263dd2655b9ca45998496',
      'docs/reference_cases/ge_beam3_g3c_so3_numerics_implementation_review_v1.json':'4debde89b44c990478487dda4852d7734b852ca01dfb2d3626e782c2328756f1',
      'docs/reference_cases/ge_beam3_g3c_so3_numerics_confirmation_receipt_v1.json':'aa7225cae347662bf92b3c02f8a1ff541d40e67f13db21900d0ad3269f88b91f',
      'docs/reference_cases/ge_beam3_g3c_so3_numerics_result_v1.json':'c8b938bbdbcce329aa83afc00037905d158142d7d2e417acee159bab95aba6fd',
      'src/anyfesolver/_ge_beam3_g3c_affine_q4_registry.py':'5f208b716d89ba778193d1261fb5975db4ece68c93fa2efce97e5c129c56acf1'}
    addendum='1eb29b8814e6555e7d16569e2c8d30c828a60458dea4a2801c10f4d58d2c2176'
    review='ca6de27e79186c9e0b37bf21e765322aa8e46dc323eb858db039951fec7efeab'
    increment_addendum='7e637ff69453715e5251614d76e08d2be7c7fd8ac4932c5fd7a4cc939023caf0'
    increment_review='04bda4789cc3529fbae2fd52f2881e8b8f7a6e48447cb6ec91c8275581ff6655'
    for path,digest in {**sources,'docs/GE_BEAM3_Q4_AFFINE_STABLE_CHART_ADDENDUM.md':addendum,
      'docs/reference_cases/ge_beam3_q4_affine_stable_chart_design_review_v1.json':review,
      'docs/GE_BEAM3_Q4_AFFINE_CANCELLATION_SAFE_CHART_ADDENDUM.md':increment_addendum,
      'docs/reference_cases/ge_beam3_q4_affine_increment_chart_design_review_v1.json':increment_review}.items():
        assert sha256((ROOT/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest
    return dict(id='stable_chart_bindings',chart_numerics_id=increment_chart.NUMERICS_ID,
        addendum_sha256=addendum,review_sha256=review,increment_addendum_sha256=increment_addendum,
        increment_review_sha256=increment_review,sources=sources,verified=True)


def station_join_fixture():
    natural=np.array(((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.)))/math.sqrt(3)
    for order in permutations(range(4)):
        source=natural[list(order)];join=candidate._station_bijection(source,natural)
        assert type(join)is tuple and sorted(join)==list(range(4))
        assert np.array_equal(source[list(join)],natural)
    duplicate=natural.copy();duplicate[1]=duplicate[0]
    missing=natural.copy();missing[0,0]=0.
    for source,target in ((duplicate,natural),(natural,duplicate),(missing,natural),(natural,missing)):
        with pytest.raises(ValueError):candidate._station_bijection(source,target)
    return dict(id='exact_coordinate_bijection',station_association_id=candidate.STATION_ASSOCIATION_ID,verified=True,rejections=4)


def test_affine_recovery_definition_and_source_identity():
    hashes={}
    for name,digest in SOURCES.items():
        raw=(ROOT/name).read_bytes().replace(b'\r\n',b'\n');assert sha256(raw).hexdigest()==digest
        hashes[name]=digest
    ids=registry.base_ids()+registry.graph_ids()
    assert len(registry.base_ids())==9 and len(registry.graph_ids())==10 and len(set(ids))==19
    rows=[]
    for identity in ids:
        c=registry.construction(identity);f=facade(identity)
        assert c.recipe_sha256==sha256(c.recipe).hexdigest()
        descriptor=f.descriptor()
        assert descriptor['chart_numerics_id']==increment_chart.NUMERICS_ID
        assert descriptor['station_association_id']=='GE_BEAM3_Q4_NATURAL_COORDINATE_BIJECTION_V1'
        assert not hasattr(f,'commit') and not hasattr(f,'restart')
        rows.append(dict(id=identity,recipe_sha256=c.recipe_sha256,descriptor_sha256=sha256(canonical(descriptor)).hexdigest()))
    lemma_raw=(ROOT/'docs/GE_BEAM3_Q4_AFFINE_RECOVERY_EXTENSION_LEMMA.md').read_bytes().replace(b'\r\n',b'\n')
    assert sha256(lemma_raw).hexdigest()=='156d33ae5a621b953b5d04050218618f6416bac1042f94d3d3fc5c305c9f8762'
    review_raw=(ROOT/'docs/reference_cases/ge_beam3_q4_affine_extension_lemma_review_v1.json').read_bytes()
    lemma=json.loads(review_raw)
    assert set(lemma)=={'decision','findings','reviewer','scope','subject_commit'}
    assert canonical(lemma)==review_raw and not lemma['findings'] and lemma['reviewer']['independent'] is True
    assert lemma['decision']=='ACCEPTED_GE_BEAM3_Q4_AFFINE_EXTENSION_LEMMA'
    assert lemma['scope']['lemma_sha256']==sha256(lemma_raw).hexdigest()
    record('definition_and_source_identity',definitions=rows,source_graph=[dict(id='source_graph',hashes=hashes)],
           extension_lemma=[dict(id='ideal_recipe_extension',lemma_sha256=sha256(lemma_raw).hexdigest(),review_sha256=sha256(review_raw).hexdigest())],
           fingerprint=[fingerprint_fixture()],chart_authority=[chart_authority()],station_join=[station_join_fixture()])


def test_affine_recovery_zero_and_station_constitutive():
    rows=[]
    for identity,pose,c,q,qa in base_contexts():
        trial=evaluate(c,q,qa);expected=independent(c,q,qa);metrics=check_chart(c,trial,expected)
        assert len(trial.stations)==4
        for i,station in enumerate(trial.stations):
            assert station.weight>0
            close(station.resultant,station.constitutive@station.strain,
                  scale=norm(station.resultant) or norm(station.constitutive))
            close(station.weight,expected['weights'][i]);close(station.natural,expected['natural'][i])
            close(station.reference_position,expected['reference_positions'][i]);close(station.current_position,expected['current_positions'][i])
        rows.append(dict(id=identity+'::'+pose,checks=metrics,energy=float(trial.physical_energy),stations=4))
    assert len(rows)==54;record('zero_and_station_constitutive',station=rows)


def test_affine_recovery_independent_material_fields():
    rows=[]
    for identity,pose,c,q,qa in base_contexts():
        trial=evaluate(c,q,qa);expected=independent(c,q,qa);checks=[]
        for i,s in enumerate(trial.stations):
            prepared=facade(c.construction_id)._prepared
            join=candidate._station_bijection(prepared.nonlinear_source_natural,prepared.natural)
            assert join==prepared.source_order and s.source_index==join[i]
            assert np.array_equal(s.source_natural,prepared.nonlinear_source_natural[join[i]])
            assert np.array_equal(s.source_natural,s.natural)
            assert np.array_equal(s.source_reference_position,prepared.nonlinear_source_positions[join[i]])
            close(s.source_reference_position,expected['reference_positions'][i])
            component_scale=np.array([1.,1.,1.,length(c),length(c),length(c),1.,1.])
            strain_scale=norm(component_scale[:,None]*expected['M'][i]*scale_vector(c)[None,:]) if pose=='ZERO' else norm(component_scale*expected['strains'][i])
            resultant_scale=norm((expected['C']@expected['M'][i])*scale_vector(c)[None,:]/component_scale[:,None]) if pose=='ZERO' else norm(expected['resultants'][i]/component_scale)
            checks.append(dict(M=close(s.mixed_map,expected['M'][i]),
                strain=close(component_scale*s.strain,component_scale*expected['strains'][i],scale=strain_scale),
                resultant=close(s.resultant/component_scale,expected['resultants'][i]/component_scale,scale=resultant_scale),
                frame=close(s.numbered_frame,expected['frame']),constitutive=close(s.constitutive,expected['C'])))
            close(component_scale*s.physical_strain,component_scale*expected['physical_strains'][i],scale=strain_scale)
            close(s.physical_resultant/component_scale,expected['physical_resultants'][i]/component_scale,scale=resultant_scale)
            for prefix in ('reference','current'):
                strain_tensors=np.array([expected[prefix+'_membrane_strain_tensor'][i],expected[prefix+'_curvature_tensor'][i]])
                resultant_tensors=np.array([expected[prefix+'_membrane_resultant_tensor'][i],expected[prefix+'_moment_tensor'][i]])
                scale=np.array([1.,length(c)])[:,None,None]
                close(getattr(s,prefix+'_strain_tensors')*scale,strain_tensors*scale,scale=strain_scale)
                close(getattr(s,prefix+'_resultant_tensors')/scale,resultant_tensors/scale,scale=resultant_scale)
                close(getattr(s,prefix+'_shear_strain'),expected[prefix+'_shear_strain'][i],scale=strain_scale)
                close(getattr(s,prefix+'_shear_resultant'),expected[prefix+'_shear_resultant'][i],scale=resultant_scale)
        # The source exposes q-local/internal coupling (24x35); the independent
        # checker exposes its transposed RHS in reference-global coordinates.
        # Preserve both APIs and explicitly transport only the comparison.
        H=trial.source_stationary_matrix;coupling=trial.source_coupling;solution=trial.source_solution
        assert H.shape==(35,35) and coupling.shape==(24,35) and solution.shape==(35,24)
        assert expected['Hsource'].shape==(35,35)
        assert expected['source_load'].shape==(35,24) and expected['source_solution'].shape==(35,24)
        T=np.kron(np.eye(8),trial.stations[0].numbered_frame)
        close(T.T@T,np.eye(24))
        rhs_global=coupling.T@T.T;solution_global=solution@T.T
        close(H,expected['Hsource']);close(rhs_global,expected['source_load'])
        close(solution_global,expected['source_solution'])
        # Test actual source equilibrium in both representations without solving
        # again. Row/column balancing avoids mixing stationary units in the norm.
        balance=np.sqrt(np.max(np.abs(H),axis=1))
        assert np.isfinite(balance).all() and np.all(balance>0)
        balanced=H/balance[:,None]/balance[None,:]
        close(balanced@(solution*balance[:,None]),coupling.T/balance[:,None])
        close(balanced@(solution_global*balance[:,None]),rhs_global/balance[:,None])
        close(-rhs_global.T@solution_global,expected['Kstationary'])
        rows.append(dict(id=identity+'::'+pose,stations=checks))
    assert len(rows)==54;record('independent_material_fields',independent=rows)


def test_affine_recovery_64_stationarity_and_schur():
    rows=[]
    for identity,pose,c,q,qa in base_contexts():
        trial=evaluate(c,q,qa);expected=independent(c,q,qa)
        block=trial.internal_block64;inverse=trial.internal_inverse64
        assert block.shape==(64,64) and inverse.shape==(64,64)
        close(block@inverse,np.eye(64));close(inverse@block,np.eye(64))
        close(trial.internal_residual64,np.zeros(64),scale=norm(block)*sum(norm(s.strain)+norm(s.resultant) for s in trial.stations) or norm(block))
        schur=trial.direct_chart_hessian-trial.internal_coupling64.T@inverse@trial.internal_coupling64
        close(trial.schur_chart_hessian,schur);close(schur,trial.physical_chart_hessian)
        for i,station in enumerate(trial.stations):
            close(station.internal_block,expected['station_internal'][i]);close(station.internal_inverse,expected['station_inverse'][i])
            close(station.chart_first,expected['station_J'][i]);close(station.chart_second,expected['station_second'][i])
        rows.append(dict(id=identity+'::'+pose,schur=array_digest(schur),full_internal_dimension=64))
    assert len(rows)==54;record('64_stationarity_and_schur',schur=rows)


def test_affine_recovery_actual_chart_work_hessian():
    rows=[]
    for identity,pose,c,q,qa in base_contexts():
        trial=evaluate(c,q,qa);expected=independent(c,q,qa)
        checks=check_physical(c,q,qa,trial,expected,table='work',row_id=identity+'::'+pose)
        zero_pose=not np.any(q) and np.array_equal(qa,np.tile(np.eye(3),(4,1,1)))
        if not zero_pose:
            checks['source_energy']=physical_comparison(c,q,qa,'source_physical_energy',trial.source_physical_energy,expected['physical_energy'])
            checks['source_force']=physical_comparison(c,q,qa,'source_physical_force',trial.source_physical_chart_force,expected['physical_force'])
            checks['source_hessian']=physical_comparison(c,q,qa,'source_physical_hessian',trial.source_physical_chart_hessian,expected['physical_hessian'])
            if any(not checks[key]['passed'] for key in ('source_energy','source_force','source_hessian')):
                rows.append(dict(id=identity+'::'+pose,checks=checks,chart_image_sentinels=False));continue
        D,S=trial.kinematics.differential,trial.kinematics.second
        numerical_force=sum((s.local_force for s in trial.numerical_channels),np.zeros(24))
        numerical_tangent=sum((s.local_tangent for s in trial.numerical_channels),np.zeros((24,24)))
        actual_force=trial.local_force-numerical_force;actual_tangent=trial.local_tangent-numerical_tangent
        close(trial.physical_chart_force,D.T@actual_force,scale=physical_scales(c)[0] if zero_pose else norm(trial.physical_chart_force))
        close(trial.physical_chart_hessian,D.T@actual_tangent@D+np.einsum('i,ijk->jk',actual_force,S))
        local_recovered=sum((s.weight*(s.mixed_map+s.nonlinear_first).T@s.resultant for s in trial.stations),np.zeros(24))
        local_hessian=sum((s.weight*((s.mixed_map+s.nonlinear_first).T@s.constitutive@(s.mixed_map+s.nonlinear_first)+
            np.einsum('a,aij->ij',s.resultant,s.nonlinear_second)) for s in trial.stations),np.zeros((24,24)))
        delta=actual_force-local_recovered;deltaK=actual_tangent-local_hessian
        close(D.T@delta,np.zeros(24),scale=physical_scales(c)[0] if zero_pose else norm(D.T@actual_force))
        close(D.T@deltaK@D+np.einsum('i,ijk->jk',delta,S),np.zeros((24,24)),scale=norm(trial.physical_chart_hessian))
        rows.append(dict(id=identity+'::'+pose,checks=checks,chart_image_sentinels=True))
    assert len(rows)==54;record('actual_chart_work_hessian',work=rows)


def test_affine_recovery_directional_derivatives_all_steps():
    rows=[]
    for identity in registry.base_ids():
        c,q,qa=context(identity);trial=evaluate(c,q,qa);s=scale_vector(c)
        direction=np.sin(np.arange(24)+1);direction/=norm(direction);direction*=s
        for h in (1e-4,1e-5,1e-6):
            plus=evaluate(c,q+h*direction,qa);minus=evaluate(c,q-h*direction,qa)
            work=(plus.physical_energy-minus.physical_energy)/(2*h)
            tangent=(plus.physical_chart_force-minus.physical_chart_force)/(2*h)
            a=close(work,float(trial.physical_chart_force@direction),tolerance=1e-7)
            b=close(s*tangent,s*(trial.physical_chart_hessian@direction),tolerance=1e-7)
            rows.append(dict(id=identity+'::h='+str(h),work=a,tangent=b))
    assert len(rows)==27;record('directional_derivatives_all_steps',directional=rows)


def test_affine_recovery_six_rigid_modes_and_common_motion():
    rigid=[];motion=[]
    for shape in registry.SHAPES:
        identity=shape+'::1';c,q,qa=context(identity,'ZERO');trial=evaluate(c,q,qa);s=scale_vector(c)
        X=c.coordinates-c.coordinates.mean(axis=0);R=np.zeros((24,6))
        for node in range(4):
            R[6*node:6*node+3,:3]=np.eye(3)
            for j in range(3):R[6*node:6*node+3,3+j]=np.cross(np.eye(3)[j],X[node])
            R[6*node+3:6*node+6,3:]=np.eye(3)
        K=s[:,None]*trial.chart_hessian*s[None,:];basis=R/s[:,None]
        close(K@basis,np.zeros((24,6)),scale=norm(K)*norm(basis))
        eigen=np.linalg.eigvalsh(K);threshold=1e-11*norm(K)
        assert np.sum(np.abs(eigen)<=threshold)==6 and np.sum(eigen>threshold)==18
        rigid.append(dict(id=identity,rigid_columns=6,total_positive_modes=18,eigenvalues=array_digest(eigen)))
        c,q,qa=context(identity);base=evaluate(c,q,qa)
        for index in range(4):
            moved,accepted,W=registry.common_motion(q,qa,c.coordinates,index);result=evaluate(c,moved,accepted)
            expected=independent(c,moved,accepted);check_chart(c,result,expected)
            checks=check_physical(c,moved,accepted,result,expected,table='common_motion',row_id=identity+'::motion='+str(index))
            close(result.physical_energy,base.physical_energy)
            for a,b in zip(result.stations,base.stations):close(a.strain,b.strain);close(a.resultant,b.resultant)
            transformed=(base.physical_spatial_force.reshape(4,6).reshape(4,2,3)@W.T).reshape(24)
            close(result.physical_spatial_force,transformed)
            motion.append(dict(id=identity+'::motion='+str(index),energy=float(result.physical_energy),checks=checks))
    assert len(rigid)==3 and len(motion)==12;record('six_rigid_modes_and_common_motion',rigid=rigid,common_motion=motion)


def test_affine_recovery_d4_and_director_transports():
    d4=[];director=[]
    for shape in registry.SHAPES:
        base_c,base_q,base_qa=context(shape+'::1');base=evaluate(base_c,base_q,base_qa)
        for k in range(8):
            identity=shape+'::1::D4:'+str(k);c,q,qa=context(identity);trial=evaluate(c,q,qa)
            expected=independent(c,q,qa);checks=check_physical(c,q,qa,trial,expected,table='d4',row_id=identity)
            close(trial.physical_energy,base.physical_energy)
            indices=np.array([6*i+j for i in c.permutation for j in range(6)])
            close(trial.physical_spatial_force,base.physical_spatial_force[indices])
            positions=np.array([s.reference_position for s in base.stations])
            matched=[]
            for station in trial.stations:
                index=int(np.argmin(np.linalg.norm(positions-station.reference_position,axis=1)));assert index not in matched;matched.append(index)
                close(station.reference_position,positions[index])
                close(station.reference_strain_tensors,base.stations[index].reference_strain_tensors)
                close(station.reference_resultant_tensors,base.stations[index].reference_resultant_tensors)
            d4.append(dict(id=identity,station_map=matched,checks=checks))
        for polarity in (-1,1):
            identity=shape+'::1::DIRECTOR:'+str(polarity);c,q,qa=context(identity);trial=evaluate(c,q,qa)
            expected=independent(c,q,qa);checks=check_physical(c,q,qa,trial,expected,table='director',row_id=identity)
            for station in trial.stations:
                sigma=polarity
                transform=np.diag([1,1,sigma,sigma,sigma,1,sigma,1])
                close(station.physical_strain,transform@station.strain)
                close(station.physical_resultant,transform@station.resultant)
                close(station.physical_strain@station.physical_resultant,station.strain@station.resultant)
            director.append(dict(id=identity,physical_polarity=polarity,checks=checks))
    assert len(d4)==24 and len(director)==6;record('d4_and_director_transports',d4=d4,director=director)


def test_affine_recovery_passive_and_same_pose_rebase():
    passive=[];rebases=[]
    for shape in registry.SHAPES:
        c,q,qa=context(shape+'::1');base=evaluate(c,q,qa)
        transformed,uq,accepted=context(shape+'::1::PASSIVE');trial=evaluate(transformed,uq,accepted)
        expected=independent(transformed,uq,accepted);check_chart(transformed,trial,expected)
        checks=check_physical(transformed,uq,accepted,trial,expected,table='passive',row_id=transformed.construction_id)
        W=transformed.passive_rotation
        close(trial.physical_energy,base.physical_energy)
        close(trial.physical_spatial_force,(base.physical_spatial_force.reshape(4,2,3)@W.T).reshape(24))
        for a,b in zip(trial.stations,base.stations):
            close(a.reference_strain_tensors,W@b.reference_strain_tensors@W.T)
            close(a.reference_resultant_tensors,W@b.reference_resultant_tensors@W.T)
        passive.append(dict(id=transformed.construction_id,energy=float(trial.physical_energy),checks=checks))
        rebased,accepted=registry.rebase(q,qa);trial=evaluate(c,rebased,accepted)
        expected=independent(c,rebased,accepted);check_chart(c,trial,expected)
        checks=check_physical(c,rebased,accepted,trial,expected,table='rebase',row_id=c.construction_id)
        close(trial.kinematics.deformation,base.kinematics.deformation);close(trial.physical_energy,base.physical_energy)
        close(trial.physical_spatial_force,base.physical_spatial_force)
        rebases.append(dict(id=c.construction_id,energy=float(trial.physical_energy),checks=checks))
    assert len(passive)==len(rebases)==3;record('passive_and_same_pose_rebase',passive=passive,rebase=rebases)


def test_affine_recovery_all_graph_q4_reference_variants():
    rows=[]
    source=json.loads((ROOT/'docs/reference_cases/ge_beam3_g3c_fixtures_v1.json').read_text())
    for identity in registry.graph_ids():
        graph_id,variant=identity.split('::')
        graph=next(g for g in source['graphs'] if g['id']==graph_id)
        element=next(e for e in graph['elements'] if e['family']=='Q4')
        nodes=dict(graph['nodes']);ids=list(element['nodes']);X=np.array([nodes[i] for i in ids],dtype=np.float64)
        direction=np.array([1.,0.,0.]);normal=np.array(source['definitions']['shell']['reference_normal'],dtype=np.float64)
        if variant=='RENUMBERED':ids=[10000+7*i for i in ids]
        if variant=='CONNECTIVITY_REVERSED':
            ids=[ids[i] for i in (0,3,2,1)];X=X[[0,3,2,1]]
        if variant=='PROPER_GLOBAL_TRANSFORM':
            W=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
            X=X@W.T+np.array([2.,-3.,1.]);normal=W@normal;direction=W@direction
        for pose in ('ZERO','MIXED'):
            c,q,qa=context(identity,pose);trial=evaluate(c,q,qa);expected=independent(c,q,qa)
            assert tuple(ids)==c.node_ids and np.array_equal(X,c.coordinates)
            assert np.array_equal(normal,c.normal) and np.array_equal(direction,c.material_direction)
            check_chart(c,trial,expected);checks=check_physical(c,q,qa,trial,expected,table='graph',row_id=identity+'::'+pose)
            assert c.element_id in (11,13,20055,20065)
            close(c.material_direction,c.passive_rotation@np.array([1.,0.,0.]))
            close(c.normal,c.passive_rotation@np.array([0.,0.,1.]))
            rows.append(dict(id=identity+'::'+pose,node_ids=list(c.node_ids),element_id=c.element_id,
                recipe_sha256=c.recipe_sha256,checks=checks))
    assert len(rows)==20;record('all_graph_q4_reference_variants',graph=rows)


def test_affine_recovery_tiny_physical_energy_and_numerical_separation():
    tiny=[];channels=[]
    for shape in registry.SHAPES:
        for amplitude in (1e-6,1e-3):
            c,q,qa=context(shape+'::1',amplitude=amplitude);trial=evaluate(c,q,qa);expected=independent(c,q,qa)
            assert trial.physical_energy>0 and expected['physical_energy']>0
            chart_checks=check_chart(c,trial,expected)
            checks=check_physical(c,q,qa,trial,expected,table='tiny',row_id=c.construction_id+'::amplitude='+str(amplitude))
            tiny.append(dict(id=c.construction_id+'::amplitude='+str(amplitude),checks=checks,chart=chart_checks,energy=float(trial.physical_energy)))
    for identity,pose,c,q,qa in base_contexts():
        trial=evaluate(c,q,qa);physical=sum(s.weight*.5*(s.strain@s.resultant) for s in trial.stations)
        close(trial.physical_energy,physical,scale=physical_scales(c)[0] if pose=='ZERO' else abs(physical))
        close(trial.energy,physical+sum(s.energy for s in trial.numerical_channels),scale=physical_scales(c)[1] if pose=='ZERO' else abs(trial.energy))
        close(trial.chart_force,trial.physical_chart_force+sum((s.chart_force for s in trial.numerical_channels),np.zeros(24)),
              scale=physical_scales(c)[1] if pose=='ZERO' else norm(trial.chart_force))
        close(trial.chart_hessian,trial.physical_chart_hessian+sum((s.chart_hessian for s in trial.numerical_channels),np.zeros((24,24))))
        assert len(trial.numerical_channels)==2
        channels.append(dict(id=identity+'::'+pose,physical=float(physical),numerical=[dict(name=s.name,energy=float(s.energy)) for s in trial.numerical_channels]))
    assert len(tiny)==6 and len(channels)==54;record('tiny_physical_energy_and_numerical_separation',tiny=tiny,channels=channels)


def test_affine_recovery_definition_observation_races(monkeypatch):
    c,q,qa=context('SQUARE::1');rows=[]
    original_descriptor=candidate.AffineQ4PhysicalRecovery.descriptor
    for route in ('descriptor','displacement_array','accepted_array','cancellation','material_descriptor','cache_array','cache_cancellation',
                  'preentry_E','preentry_coordinates','preentry_material_direction','preentry_policy','preentry_cached_definition',
                  'preentry_chart_missing','preentry_chart_old','preentry_chart_wrong','preentry_chart_cache',
                  'preentry_station_missing','preentry_station_old','preentry_station_wrong','preentry_station_cache'):
        f=candidate.AffineQ4PhysicalRecovery(c.construction_id);other=candidate.AffineQ4PhysicalRecovery('RECTANGLE::1')
        if route.startswith('cache_'):f.evaluate(q,qa)
        if route=='preentry_cached_definition':
            f.evaluate(q,qa)
            object.__setattr__(f,'_body',other._body);object.__setattr__(f,'_seal',other._seal)
        elif route in ('preentry_chart_cache','preentry_station_cache'):
            f.evaluate(q,qa)
            old=json.loads(f._body)
            if route=='preentry_chart_cache':old['chart_numerics_id']=stable_chart.NUMERICS_ID
            else:del old['station_association_id']
            prior=replace(f._prepared,definition_sha256=sha256(canonical(old)).hexdigest())
            object.__setattr__(f,'_prepared',prior);object.__setattr__(f,'_prepared_seal',candidate._fingerprint(prior))
        elif route.startswith('preentry_'):
            body=json.loads(f._body)
            key=route.removeprefix('preentry_')
            if key=='chart_missing':del body['chart_numerics_id']
            elif key=='chart_old':body['chart_numerics_id']=stable_chart.NUMERICS_ID
            elif key=='chart_wrong':body['chart_numerics_id']='FOREIGN_CHART_NUMERICS'
            elif key=='station_missing':del body['station_association_id']
            elif key=='station_old':body['station_association_id']='RAW_POSITIONAL_STATION_ZIP'
            elif key=='station_wrong':body['station_association_id']='FOREIGN_STATION_ASSOCIATION'
            elif key=='E':body[key]=200.
            elif key=='coordinates':body[key][0][0]+=.125
            elif key=='material_direction':body[key]=[0.,1.,0.]
            else:body[key]='GE_BEAM3_G3C_MATRIX_POSE_SHELL_PULLBACK_V1'
            raw=canonical(body);object.__setattr__(f,'_body',raw);object.__setattr__(f,'_seal',sha256(raw).hexdigest())
        entered=[]
        def swap():
            if route.startswith('cache_'):
                altered=replace(f._prepared,constitutive=np.array(f._prepared.constitutive)*2)
                object.__setattr__(f,'_prepared',altered);object.__setattr__(f,'_prepared_seal',candidate._fingerprint(altered))
            else:
                object.__setattr__(f,'_body',other._body);object.__setattr__(f,'_seal',other._seal)
        def forbidden(*args,**kwargs):entered.append(True);raise AssertionError('family entered after definition corruption')
        class Observed:
            def __init__(self,value):self.value=value
            def __array__(self,dtype=None,copy=None):swap();return np.array(self.value,dtype=dtype,copy=True)
        def observed_descriptor(self):
            value=original_descriptor(self)
            if self is f:
                if route=='material_descriptor':value['material_direction']=[0.,1.,0.]
                else:swap()
            return value
        with monkeypatch.context() as patch:
            patch.setattr(candidate,'family_objects',forbidden)
            if route in ('descriptor','material_descriptor'):patch.setattr(candidate.AffineQ4PhysicalRecovery,'descriptor',observed_descriptor)
            options={}
            if route in ('cancellation','cache_cancellation'):options['cancel_check']=lambda:(swap() or False)
            with pytest.raises(ValueError):
                f.evaluate(Observed(q) if route in ('displacement_array','cache_array') else q,
                           Observed(qa) if route=='accepted_array' else qa,**options)
        assert not entered
        rows.append(dict(id=route,rejected_before_family=True))
    record('definition_observation_races',races=rows)


def test_affine_recovery_immutable_detached_results_and_reentry(monkeypatch):
    c,q,qa=context('SQUARE::1');f=candidate.AffineQ4PhysicalRecovery(c.construction_id)
    mutable_q=np.array(q,copy=True);mutable_qa=np.array(qa,copy=True)
    first=f.evaluate(mutable_q,mutable_qa)
    assert type(first.candidate)is bytes and type(first.descriptor_bytes)is bytes
    full_fingerprint=candidate._fingerprint(first)
    assert full_fingerprint==sha256(canonical(candidate._payload(first))).hexdigest()
    assert candidate._fingerprint(replace(first,candidate=first.candidate+b'\xff'))!=full_fingerprint
    assert candidate._fingerprint(replace(first,descriptor_bytes=first.descriptor_bytes+b'\x00'))!=full_fingerprint
    def arrays(value):
        if isinstance(value,np.ndarray):yield value
        elif is_dataclass(value):
            for field in fields(value):yield from arrays(getattr(value,field.name))
        elif isinstance(value,tuple):
            for entry in value:yield from arrays(entry)
        else:assert not isinstance(value,(list,dict))
    snapshots=[array_digest(a) for a in arrays(first)]
    count=0
    for a in arrays(first):
        assert not a.flags.writeable
        with pytest.raises(ValueError):a.setflags(write=True)
        if a.size:
            with pytest.raises(ValueError):a.flat[0]=0
        count+=1
    assert count>30
    with pytest.raises((FrozenInstanceError,AttributeError)):first.physical_energy=0
    mutable_q[:]=0;mutable_qa[:]=0
    assert [array_digest(a) for a in arrays(first)]==snapshots
    assert candidate._fingerprint(first)==full_fingerprint
    # Intercept obsolete mechanical entrypoints during this existing genuine
    # repeat. Registry's captured original _exp_terms/so3_exp aliases remain
    # intact: source recipes intentionally retain their original arithmetic.
    entered=[];used={'exp':0,'log':0,'fit':0,'connection':0}
    def forbidden(*args,**kwargs):entered.append(True);raise AssertionError('obsolete mechanical chart entered')
    def counted(name,function):
        def call(*args,**kwargs):used[name]+=1;return function(*args,**kwargs)
        return call
    with monkeypatch.context() as patch:
        patch.setattr(original_shell,'deformation',forbidden)
        patch.setattr(original_shell,'so3_exp',forbidden);patch.setattr(original_shell,'so3_log',forbidden)
        patch.setattr(original_shell,'_exp_terms',forbidden)
        patch.setattr(original_ad,'so3_exp',forbidden);patch.setattr(original_ad,'so3_log',forbidden)
        patch.setattr(original_joint,'_exp_terms',forbidden)
        patch.setattr(stable_chart,'deformation',forbidden)
        patch.setattr(stable_chart,'rotation_jets',forbidden)
        patch.setattr(increment_chart,'_exp_delta',counted('exp',increment_chart._exp_delta))
        patch.setattr(increment_chart,'_log_delta',counted('log',increment_chart._log_delta))
        patch.setattr(increment_chart,'_rotation_jets_from_increments',counted('fit',increment_chart._rotation_jets_from_increments))
        patch.setattr(candidate,'_exp_terms',counted('connection',stable_chart._exp_terms))
        again=f.evaluate(q,qa)
    assert not entered and all(value>0 for value in used.values())
    assert [array_digest(a) for a in arrays(again)]==snapshots
    for a,b in zip(arrays(first),arrays(again)):assert not np.shares_memory(a,b) or not a.flags.writeable
    inner=[]
    def reentry():
        with pytest.raises(RuntimeError):f.evaluate(q,qa)
        inner.append(True);return False
    f.evaluate(q,qa,cancel_check=reentry);assert len(inner)==2
    changed=np.array(qa,copy=True);changed[0,0,0]+=1e-3
    with pytest.raises(ValueError):f.evaluate(q,changed)
    assert [array_digest(a) for a in arrays(first)]==snapshots
    entered=Event();release=Event();completed=[];errors=[]
    def pause():entered.set();assert release.wait(20);return False
    def concurrent():
        try:completed.append(f.evaluate(q,qa,cancel_check=pause))
        except BaseException as exc:errors.append(exc)
    thread=Thread(target=concurrent,daemon=True);thread.start()
    try:
        assert entered.wait(20)
        with pytest.raises(RuntimeError):f.evaluate(q,qa)
    finally:release.set();thread.join(30)
    assert not thread.is_alive() and not errors and len(completed)==1
    cached=f._prepared;badweights=np.array(cached.weights,copy=True)*2
    object.__setattr__(f,'_prepared',replace(cached,weights=badweights))
    with pytest.raises(ValueError):f.evaluate(q,qa)
    object.__setattr__(f,'_prepared',cached)
    evidence=sha256(canonical(snapshots)).hexdigest()
    record('immutable_detached_results_and_reentry',immutability=[dict(id='all_detached_arrays',array_count=count,verified=True,prior_arrays_sha256=evidence),
        dict(id='caller_arrays_preserved',verified=True,prior_arrays_sha256=evidence),
        dict(id='same_input_repeat',verified=True,prior_arrays_sha256=evidence),
        dict(id='reentry',rejections=len(inner),verified=True,prior_arrays_sha256=evidence),
        dict(id='changed_accepted_matrix',verified=True,prior_arrays_sha256=evidence),
        dict(id='concurrent_evaluation',verified=True,prior_arrays_sha256=evidence),
        dict(id='operator_cache_tamper',verified=True,prior_arrays_sha256=evidence),
        dict(id='nested_candidate_bytes',verified=True,prior_arrays_sha256=evidence,fingerprint_sha256=full_fingerprint),
        dict(id='old_chart_entrypoints_intercepted',verified=True,prior_arrays_sha256=evidence)])


def test_affine_recovery_unsupported_routes_and_cancellation(monkeypatch):
    c,q,qa=context('SQUARE::1');rows=[];entered=[]
    original_family=candidate.family_objects
    def counted(*args,**kwargs):entered.append(True);return original_family(*args,**kwargs)
    nonaffine=np.array(c.coordinates,copy=True);nonaffine[2,0]+=1e-4
    badnormal=np.array(c.normal,copy=True);badnormal[0]=1e-4
    badmaterial=np.array(c.material_direction,copy=True);badmaterial[1]=1e-4
    wrongnodes=tuple(reversed(c.node_ids))
    for name,options in (('nonaffine',{'coordinates':nonaffine}),('director',{'normal':badnormal}),
        ('material_direction',{'material_direction':badmaterial}),('node_ids',{'node_ids':wrongnodes}),
        ('generalized_section',{'generalized_section':np.eye(8)}),('history_section',{'plasticity':True}),
        ('offset',{'offset':.01}),('initial_fields',{'initial_strain':np.ones(8)}),
        ('foreign_policy',{'formulation_id':'GE_BEAM3_G3C_MATRIX_POSE_SHELL_PULLBACK_V1'})):
        with monkeypatch.context() as patch:
            patch.setattr(candidate,'family_objects',counted)
            with pytest.raises((ValueError,TypeError)):candidate.AffineQ4PhysicalRecovery(c.construction_id,**options)
        assert not entered;rows.append(dict(id=name,rejected_before_family=True))
    for name in ('GE_BEAM3_G3C_MATRIX_POSE_SHELL_PULLBACK_V1','OLD_RETAINED_35_VARIABLE_SYSTEM','FOREIGN'):
        with pytest.raises(ValueError):candidate.AffineQ4PhysicalRecovery(name)
        rows.append(dict(id=name,rejected_before_family=True))
    for label,u,accepted in (('nonfinite_q',np.full(24,np.nan),qa),('nonfinite_accepted',q,np.full((4,3,3),np.inf)),
                             ('wrong_q_shape',q[:18],qa),('wrong_rotation_shape',q,qa[:3])):
        with monkeypatch.context() as patch:
            patch.setattr(candidate,'family_objects',counted)
            with pytest.raises(ValueError):candidate.AffineQ4PhysicalRecovery(c.construction_id).evaluate(u,accepted)
        assert not entered;rows.append(dict(id=label,rejected_before_family=True))
    for phase in ('before_work','before_publication','invalid_callback'):
        f=candidate.AffineQ4PhysicalRecovery(c.construction_id);calls=[];entered.clear()
        def cancellation():
            calls.append(True)
            if phase=='invalid_callback':return 1
            return phase=='before_work' or len(calls)==2
        with monkeypatch.context() as patch:
            patch.setattr(candidate,'family_objects',counted)
            with pytest.raises(ValueError if phase=='invalid_callback' else candidate.AffineRecoveryCancelled):
                f.evaluate(q,qa,cancel_check=cancellation)
        assert bool(entered)==(phase=='before_publication')
        rows.append(dict(id=phase,callbacks=len(calls),family_entries=len(entered),published=False))
    record('unsupported_routes_and_cancellation',rejections=rows)


def increment_mutations(monkeypatch):
    c,q,qa=context('SQUARE::1',amplitude=1e-6)
    expected=checker.chart.evaluate(c.coordinates,q,qa)
    captured=[];verify=increment_chart._verify_eigen_derivatives
    def observe_derivatives(*args):
        result=verify(*args)
        captured.append((tuple(np.array(v,copy=True) if isinstance(v,np.ndarray) else v for v in args),result))
        return result
    with monkeypatch.context() as patch:
        patch.setattr(increment_chart,'_verify_eigen_derivatives',observe_derivatives)
        baseline=increment_chart.deformation(c.coordinates,q,qa)
    assert len(captured)==1
    witness,derivative_receipt=captured[0]
    assert derivative_receipt['derivative_coordinates']==24 and derivative_receipt['derivative_pairs']==576
    assert set(derivative_receipt['derivative_residuals'])=={'first_normalization','second_normalization','first_stationarity','second_stationarity','second_symmetry'}
    assert all(math.isfinite(v) and 0<=v<=1e-11 for v in derivative_receipt['derivative_residuals'].values())
    check_chart(c,type('Trial',(),{'kinematics':baseline})(),expected)
    rows=[];translations=q.reshape(4,6)[:,:3]
    covariance=increment_chart._decimal_covariance
    def missing_delta(reference,u):
        xc,uc,K,ki=covariance(reference,u)
        _,_,baseline_K,_=covariance(reference,np.zeros_like(u))
        return xc,uc,baseline_K,ki
    def naive(rotation,delta,reference,u):
        value=original_ad.matvec(original_ad.transpose(rotation),[a+b for a,b in zip(reference,u)])
        return [a-b for a,b in zip(value,reference)]
    rotation_helper=increment_chart._rotation_jets_from_increments
    def omit_delta_derivatives(*args):
        R,delta,xc,uc,receipt=rotation_helper(*args)
        delta=[[original_ad.Jet2.constant(v.value,24) for v in row] for row in delta]
        return R,delta,xc,uc,receipt
    for name,hook,mutation,quantity in (
        ('missing_delta_covariance','_decimal_covariance',missing_delta,'d'),
        ('naive_reference_subtraction','_translation_deformation',naive,'d'),
        ('omitted_delta_derivatives','_rotation_jets_from_increments',omit_delta_derivatives,'D')):
        with monkeypatch.context() as patch:
            patch.setattr(increment_chart,hook,mutation)
            bad=increment_chart.deformation(c.coordinates,q,qa)
        actual=bad.deformation if quantity=='d' else bad.differential
        scales=scale_vector(c)
        a=actual/scales if quantity=='d' else actual*scales[None,:]/scales[:,None]
        b=expected[quantity]/scales if quantity=='d' else expected[quantity]*scales[None,:]/scales[:,None]
        error=norm(a-b)/norm(b)
        assert math.isfinite(error) and error>1e-11
        rows.append(dict(id=name,rejection='INDEPENDENT_CHART_COMPARISON',relative_error=error))
    with localcontext() as decimal_context:
        decimal_context.prec=80;decimal_context.rounding=ROUND_HALF_EVEN
        _,_,K,_=covariance(c.coordinates,translations)
        eigenvalues,vectors=np.linalg.eigh(np.array(K,dtype=float))
        basis=increment_chart._orthogonal_seed(vectors)
        wrong=basis[0];lam=increment_chart._dot(wrong,[increment_chart._dot(row,wrong) for row in K])
        with pytest.raises(increment_chart.ChartBranchError):increment_chart._certify_branch(K,wrong,lam,basis)
        # Prevent a real Newton update, retaining all16 actual residual tests.
        def no_step(matrix,rhs):return [Decimal(0)]*len(rhs)
        with monkeypatch.context() as patch:
            patch.setattr(increment_chart,'_solve',no_step)
            altered=np.array(eigenvalues,copy=True);altered[-1]+=.1
            with pytest.raises(increment_chart.ChartNonconvergence):increment_chart._refine_eigenpair(K,altered,vectors)
    rows.extend([dict(id='wrong_davenport_branch',rejection='BRANCH_REJECTION'),
                 dict(id='wrong_polar_branch',rejection='BRANCH_REJECTION'),
                 dict(id='davenport_nonconvergence',rejection='NONCONVERGENCE_REJECTION')])
    # Registered SQUARE ZERO has diagonal planar covariance. A pi rotation
    # about its normal is exactly proper and stationary, but not maximizing.
    # Exercise the whole independent fit so residual/properness cannot mask
    # a missing maximizing-branch check.
    zero,zero_qa=registry.pose(c.construction_id,'ZERO')
    with monkeypatch.context() as patch:
        patch.setattr(checker.chart,'_proper_seed',lambda seed:[[Decimal(-1),Decimal(0),Decimal(0)],
            [Decimal(0),Decimal(-1),Decimal(0)],[Decimal(0),Decimal(0),Decimal(1)]])
        with pytest.raises(checker.chart.IncrementChartError,match='not maximizing'):
            checker.chart.evaluate(c.coordinates,zero,zero_qa)
    with monkeypatch.context() as patch:
        patch.setattr(checker.chart,'_cayley_step',lambda rotation,omega:rotation)
        with pytest.raises(checker.chart.IncrementChartError,match='did not converge'):
            checker.chart.evaluate(c.coordinates,q,qa)
    rows.append(dict(id='polar_nonconvergence',rejection='NONCONVERGENCE_REJECTION'))
    for name,index in (('eigen_first_derivative',4),('eigen_second_derivative',5)):
        changed=list(witness);changed[index]=np.array(changed[index],copy=True)
        changed[index][(0,)*changed[index].ndim]+=1.
        with pytest.raises(increment_chart.ChartEvaluationError,match='eigen-derivative identity failed'):verify(*changed)
        rows.append(dict(id=name,rejection='DERIVATIVE_IDENTITY_REJECTION'))
    return rows,dict(id='SQUARE::1::amplitude=1e-06',**derivative_receipt)


def test_affine_recovery_actual_mutation_rejection(monkeypatch):
    rows=[]
    for shape in registry.SHAPES:
        c,q,qa=context(shape+'::1');f=facade(c.construction_id);trial=evaluate(c,q,qa);expected=independent(c,q,qa)
        prepared=f._prepared;kin=trial.kinematics
        before=sha256(candidate.canonical(candidate._payload(prepared))).hexdigest()
        def forbidden(*args,**kwargs):raise AssertionError('mutation must not rebuild stationary reference')
        with monkeypatch.context() as protection:
            protection.setattr(candidate,'_prepare_reference',forbidden)
            def station_compare(value):
                stations,g,H=candidate._station_fields(value,kin)
                close(np.array([s.strain for s in stations]),expected['strains'])
                close(np.array([s.resultant for s in stations]),expected['resultants'])
                close(np.array([s.natural for s in stations]),expected['natural'])
                close(np.array([s.reference_position for s in stations]),expected['reference_positions'])
                close(np.array([s.weight for s in stations]),expected['weights'])
                close(np.array([s.numbered_frame for s in stations]),np.array([expected['frame']]*4))
                close(g,expected['physical_force']);close(H,expected['physical_hessian'])
            station_compare(prepared)
            for name in ('station_order','weight','frame','M','resultant'):
                changes={}
                if name=='station_order':changes['natural']=np.array(prepared.natural[::-1],copy=True)
                elif name=='weight':changes['weights']=np.array(prepared.weights)*2
                elif name=='frame':changes['frame']=np.array(prepared.frame)[:,[1,0,2]]
                elif name=='M':
                    value=np.array(prepared.mixed_maps,copy=True);value[0,0,0]+=.1;changes['mixed_maps']=value
                else:changes['constitutive']=np.array(prepared.constitutive)*1.1
                with pytest.raises(AssertionError):station_compare(replace(prepared,**changes))
                rows.append(dict(id=shape+'::'+name,rejection='INDEPENDENT_STATION_COMPARISON'))
            original=candidate._nonlinear_terms
            for mode in ('n','Dn','Hn'):
                def altered(*args,mode=mode):
                    values=list(original(*args));index={'n':0,'Dn':1,'Hn':2}[mode];values[index]=np.zeros_like(values[index]);return tuple(values)
                with monkeypatch.context() as patch:
                    patch.setattr(candidate,'_nonlinear_terms',altered)
                    with pytest.raises(AssertionError):station_compare(prepared)
                rows.append(dict(id=shape+'::'+mode,rejection='INDEPENDENT_STATION_COMPARISON'))
            original_schur=candidate._schur_terms
            def omit_geometric(*args):
                material,geometric=original_schur(*args);return material,np.zeros_like(geometric)
            with monkeypatch.context() as patch:
                patch.setattr(candidate,'_schur_terms',omit_geometric)
                with pytest.raises(AssertionError):station_compare(prepared)
            rows.append(dict(id=shape+'::force_weighted_Hessian',rejection='INDEPENDENT_HESSIAN'))
            badkin=replace(kin,second=np.zeros_like(kin.second))
            _,_,badH=candidate._station_fields(prepared,badkin)
            with pytest.raises(AssertionError):close(badH,expected['physical_hessian'])
            rows.append(dict(id=shape+'::chart_second',rejection='INDEPENDENT_HESSIAN'))
            C=trial.internal_coupling64;inverse=trial.internal_inverse64
            # Actual full station equilibrium catches a coupling sign error.
            badcoupling=-C
            solution=-inverse@C
            with pytest.raises(AssertionError):close(trial.internal_block64@solution,-badcoupling)
            rows.append(dict(id=shape+'::coupling_sign',rejection='STATION_EQUILIBRIUM'))
            badinverse=np.array(inverse,copy=True);badinverse[0,0]+=1
            with pytest.raises(AssertionError):close(trial.internal_block64@badinverse,np.eye(64))
            rows.append(dict(id=shape+'::inverse',rejection='STATION_INVERSE'))
            contaminated=trial.physical_energy+sum(channel.energy for channel in trial.numerical_channels)
            with pytest.raises(AssertionError):close(contaminated,expected['physical_energy'])
            rows.append(dict(id=shape+'::numerical_energy_leak',rejection='MATERIAL_ENERGY'))
            # Recreate the actual positional-zip defect from the measured
            # coordinate bijection, not an assumed swap. Reuse the immutable
            # stationary maps; only the registered BENDING chart is evaluated.
            bq,bqa=registry.pose(c.construction_id,'BENDING')
            bkin=candidate.deformation(c.coordinates,bq,bqa)
            good,_,_=candidate._station_fields(prepared,bkin)
            independent_strains=np.array([m@bkin.deformation+.5*np.einsum('aij,i,j->a',h,bkin.deformation,bkin.deformation)
                for m,h in zip(expected['M'],expected['nonlinear_hessians'])])
            close(np.array([s.strain for s in good]),independent_strains)
            raw_order=np.argsort(prepared.source_order)
            assert not np.array_equal(raw_order,np.arange(4))
            badjoin=replace(prepared,transverse_maps=np.array(prepared.transverse_maps)[raw_order],
                membrane_maps=np.array(prepared.membrane_maps)[raw_order],source_weights=np.array(prepared.source_weights)[raw_order])
            bad,_,_=candidate._station_fields(badjoin,bkin)
            with pytest.raises(AssertionError):close(np.array([s.strain for s in bad]),independent_strains)
            if shape=='SQUARE':
                assert any(float(a.nonlinear[2])*float(b.nonlinear[2])<0 for a,b in zip(good,bad))
            rows.append(dict(id=shape+'::nonlinear_point_join',rejection='INDEPENDENT_STATION_COMPARISON'))
        assert sha256(candidate.canonical(candidate._payload(prepared))).hexdigest()==before
    chart_mutations,derivatives=increment_mutations(monkeypatch)
    assert len(rows)==42;record('actual_mutation_rejection',mutations=rows,chart_mutations=chart_mutations,eigen_derivatives=[derivatives])


def test_affine_recovery_smoke_square_station_work():
    rows=[]
    for pose in ('ZERO','MIXED'):
        c,q,qa=context('SQUARE::1',pose);trial=evaluate(c,q,qa);expected=independent(c,q,qa)
        check_chart(c,trial,expected);checks=check_physical(c,q,qa,trial,expected,table='smoke',row_id='SQUARE::1::'+pose)
        rows.append(dict(id='SQUARE::1::'+pose,checks=checks))
    record('smoke_square_station_work',smoke=rows)
