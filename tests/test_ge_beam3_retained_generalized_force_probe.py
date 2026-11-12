"""Frozen research equivalence and actual uncondensed force solve tests."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pytest
from docs.reference_cases import ge_beam3_retained_generalized_force_probe as probe
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_p5.algebra import rotation
from test_ge_beam3_generalized_slenderness_diagnostic import make
from test_ge_beam3_schur_line_program import save

ARCHIVE=Path('C:/Users/AudunArnesenNyhus/AppData/Local/ANYrelease/ge-beam3-force-funnel-0197a34-20260908')
FORCE=np.array([.005,-.003,.002]);COUPLE=np.array([.002,-.001,.003])

def saved(curved):
    raw=(ARCHIVE/'archive-manifest.json').read_bytes()
    assert len(raw)==5012 and sha256(raw).hexdigest().upper()=='7AF4DAE6C13C8CDCD6A6CC289229F7B464153BFDAF308F2C1E659D5DA7115AF8'
    manifest=json.loads(raw);name='100.0-'+('curved' if curved else 'straight')+'-E/pytest/test_actual_generalized_force_0/result.json'
    data=(ARCHIVE/name).read_bytes();assert [len(data),sha256(data).hexdigest().upper()]==manifest[name]
    return json.loads(data)['states']['1']

def close(a,b):
    a=np.asarray(a);b=np.asarray(b);error=float(np.linalg.norm(a-b)/max(1.,np.linalg.norm(b)))
    assert error<=1e-11
    return error

@pytest.mark.parametrize('curved',(False,True))
def test_saved_native_full_operator(curved,tmp_path):
    model,_,_=make(100.,curved,np.eye(3));op=model.mesh.elements[1].operator;s=saved(curved)
    state=dict(positions=np.array(s['positions']),position_low=np.array(s['position_low']),
        nodal_frames=np.array(s['committed_nodal_rotation_matrices'])@op.reference.nodal_triads,
        cell_rotations=np.array(s['response']['rotations']),resultants=np.array(s['response']['resultants']))
    assert canonical(s['origins'])==canonical(op.cell.virgin())
    r,j,e,w=probe.assemble(op,state,op.cell.virgin(),FORCE,COUPLE,1.)
    errors=dict(residual=close(r,s['response']['full_residual']),jacobian=close(j,s['response']['full_spatial_jacobian']))
    save(tmp_path/'equivalence.json',dict(errors=errors,work=w.value,residual=r,jacobian=j,production_qualified=False))

@pytest.mark.parametrize('rho,curved',((100.,False),(100.,True),(10000.,False),(10000.,True),(1000000.,False),(1000000.,True)))
def test_retained_generalized_force(rho,curved,tmp_path):
    model,_,_=make(rho,curved,np.eye(3));op=model.mesh.elements[1].operator
    with (tmp_path/'progress.jsonl').open('xb') as log:
        def progress(row):
            log.write(canonical(row));log.flush()
            print({k:v for k,v in row.items() if k in ('stage','parameter','iteration','equilibrium','compatibility','message')},flush=True)
        result=probe.solve(op,FORCE,COUPLE,progress=progress)
    save(tmp_path/'retained.json',result)
    assert len(result['records'])==2 and result['status']=='COMPLETED'
    if rho==100.:
        old=saved(curved);s=result['state']
        errors=dict(positions=close(s['positions']+s['position_low'],np.array(old['positions'])+old['position_low']),
            nodal_frames=close(s['nodal_frames'],np.array(old['committed_nodal_rotation_matrices'])@op.reference.nodal_triads),
            cell_rotations=close(s['cell_rotations'],old['response']['rotations']),
            resultants=close(s['resultants'],old['response']['resultants']))
        save(tmp_path/'native-comparison.json',dict(errors=errors,production_qualified=False))

@pytest.mark.parametrize('rho',(100.,10000.,1000000.))
def test_retained_generalized_covariance(rho,tmp_path):
    q=rotation([.4,-.3,.2]);outputs=[]
    for matrix in (np.eye(3),q):
        model,_,_=make(rho,True,matrix)
        outputs.append(probe.solve(model.mesh.elements[1].operator,matrix@FORCE,matrix@COUPLE))
    a,b=[o['state'] for o in outputs]
    errors=dict(positions=close(b['positions']+b['position_low'],(a['positions']+a['position_low'])@q.T),
        nodal_frames=close(b['nodal_frames'],q@a['nodal_frames']),
        cell_rotations=close(b['cell_rotations'],q@a['cell_rotations']@q.T))
    save(tmp_path/'covariance.json',dict(errors=errors,outputs=outputs,production_qualified=False))
