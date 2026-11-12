"""Eight-macro displacement-controlled diagnostic; preserved eight-macro inputs."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import numpy as np
from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_native_translation import TranslationProgram,solve_translation
from anyfesolver._ge_beam3_native_translation_restart import decode_checkpoint,encode_checkpoint
from anyfesolver._ge_beam3_native_generalized_recovery import recover_native_fields
from docs.reference_cases.ge_beam3_uniform_arch_reference import solve as continuum
from docs.reference_cases.ge_beam3_uniform_arch_fields import fields
from docs.reference_cases.ge_beam3_preserved_arch_load_comparison import parse
from test_ge_beam3_native_translation import packet
from test_ge_beam3_schur_line_program import save

BASE_STATUS='docs/reference_cases/ge_beam3_uniform_arch_refinement_status.json'
DROPS=(.0009924835187201996,.002196594350540417,.004368321784681136)
SECTION=np.array([1000.,400.,400.,.008,.01,.01])


def make():
    model=FEModel('native-uniform-arch-eight-macro-displacement');t=np.linspace(-1.,1.,17)
    xyz=np.array([t,.1*(1-t*t),np.zeros(17)]).T;frames=[]
    for index,x in enumerate(t):
        axis=np.array([1.,-.2*x,0.]);axis/=np.linalg.norm(axis);second=np.array([0.,0.,1.])
        frames.append(np.column_stack((axis,second,np.cross(axis,second))));model.add_node(index+1,*xyz[index])
    law=EllipsoidalGeneralizedSection(np.diag(SECTION),np.eye(6),1e6,1.)
    for eid,start in enumerate(range(0,16,2),1):
        ref=Reference(xyz[start:start+3],np.array(frames[start:start+3]))
        e=NativeGeneralizedStaticElement(eid,tuple(range(start+1,start+4)),ref,law,order=4)
        model.add_element(eid,e);model.materials[e.material_name]=law
    model.add_boundary_condition(BoundaryCondition('clamped',[1,17],{k:0. for k in ('ux','uy','uz','rx','ry','rz')}))
    load=DistributedPattern(LinePattern(tuple((eid,0.,-1.,0.) for eid in range(1,9))),())
    return model,TranslationProgram(tuple(-v for v in DROPS),9,'uy',load)


def test_eight_macro_comparison(tmp_path):
    base_raw=Path(BASE_STATUS).read_bytes();base=parse(base_raw)
    assert tuple(r['drop'] for r in base['metrics'])==DROPS
    model,program=make()
    result=solve_translation(model,program,progress=lambda row:print(row,flush=True))
    save(tmp_path/'result.json',packet(result));save(tmp_path/'checkpoint.json',result.checkpoint)
    assert result.status=='completed',result.failure
    assert len(result.checkpoint)<2*1024**2
    fresh,_=make();chain,records=decode_checkpoint(fresh,program,result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    assert encode_checkpoint(fresh,program,chain,records)==result.checkpoint
    rows=[];previous=None
    for snapshot,record,coarse in zip(chain[1:],records,base['metrics'],strict=True):
        recovery=[];parameters=[]
        for eid,e in sorted(fresh.mesh.elements.items()):
            local=snapshot['states'][eid]
            recovered=recover_native_fields(e,fresh.mesh,local,expected_committed_total_u=snapshot['displacements'][list(e.get_dof_mapping(fresh.mesh))])
            assert all(sum(h.accumulated)==0. and not any(sum(v) for v in h.plastic) for h in local['response'].history.stations)
            for station in recovered['stations']:
                parameters.append(float(e.operator.reference.position(station['xi'])[0]));recovery.append(station)
        assert len(recovery)==64
        x=np.array(parameters);samples=np.unique(np.r_[np.linspace(-1.,0.,129),-np.abs(x)])
        previous=continuum(float(-record['target']),previous=previous,profile='BVP9',sample_parameters=samples.tolist())
        expected=fields(previous,x)
        force=np.array([r['resultants']+r['resultants_low'] for r in recovery]);strain=np.array([r['strain']+r['strain_low'] for r in recovery])
        measures=np.array([r['measure'] for r in recovery]);positions=np.array([r['current_position']+r['current_position_low'] for r in recovery])
        frames=np.array([r['current_frame'] for r in recovery])
        norm=float(np.sum(measures[:,None]*expected['resultants']**2/SECTION));assert norm>0
        field_error=float(np.sqrt(np.sum(measures[:,None]*(force-expected['resultants'])**2/SECTION)/norm))
        constitutive=float(np.max(np.abs(force-strain*SECTION)/np.maximum(1.,np.abs(force))))
        assert constitutive<=1e-11
        error=abs(record['parameter']/previous.density-1.)
        row=dict(step=record['index'],drop=-record['target'],density=record['parameter'],reference_density=previous.density,
            load_error=error,coarse_load_error=coarse['load_error'],load_error_improved=error<coarse['load_error'],
            resultant_compliance_norm_error=field_error,position_max_error=float(np.max(np.abs(positions-expected['position']))),
            frame_max_error=float(np.max(np.abs(frames-expected['frame']))),constitutive_error=constitutive,stations=len(recovery))
        save(tmp_path/('state-'+str(record['index'])+'.json'),dict(comparison=row,recovery=recovery,reference=asdict(previous),expected=expected))
        rows.append(row)
    assert Path(BASE_STATUS).read_bytes()==base_raw
    save(tmp_path/'comparison.json',dict(rows=rows,base_status_sha256=sha256(base_raw).hexdigest(),checkpoint_roundtrip=True,
        checkpoint_bytes=len(result.checkpoint),macros=8,native_arc_retried=False,full_spatial_stability=False,production_qualified=False))
