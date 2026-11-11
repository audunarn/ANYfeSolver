"""Bounded native arch smoke; no coarse-mesh qualification threshold."""
from dataclasses import asdict
from hashlib import sha256
import numpy as np
from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_native_arc import ArcProgram,solve_arc
from anyfesolver._ge_beam3_native_arc_restart import decode_checkpoint
from docs.reference_cases import ge_beam3_uniform_arch_reference as continuum
from test_ge_beam3_native_arc import packet
from test_ge_beam3_schur_line_program import save


def make():
    model=FEModel('native-uniform-arch-two-macro-smoke');t=np.linspace(-1.,1.,5)
    xyz=np.array([t,.1*(1-t*t),np.zeros(5)]).T;frames=[]
    for index,x in enumerate(t):
        axis=np.array([1.,-.2*x,0.]);axis/=np.linalg.norm(axis);second=np.array([0.,0.,1.])
        frames.append(np.column_stack((axis,second,np.cross(axis,second))))
        model.add_node(index+1,*xyz[index])
    law=EllipsoidalGeneralizedSection(np.diag([1000.,400.,400.,.008,.01,.01]),np.eye(6),1e6,1.)
    for eid,start in enumerate((0,2),1):
        ref=Reference(xyz[start:start+3],np.array(frames[start:start+3]))
        e=NativeGeneralizedStaticElement(eid,tuple(range(start+1,start+4)),ref,law,order=4)
        model.add_element(eid,e);model.materials[e.material_name]=law
    model.add_boundary_condition(BoundaryCondition('clamped',[1,5],{k:0. for k in ('ux','uy','uz','rx','ry','rz')}))
    load=DistributedPattern(LinePattern(((1,0.,-1.,0.),(2,0.,-1.,0.))),())
    return model,ArcProgram((.2,.2,.2,.2),.1,load,parameter_scale=.1)


def test_native_four_step_smoke(tmp_path):
    model,program=make();events=[]
    def progress(row):
        events.append(row)
        print(row,flush=True)
    result=solve_arc(model,program,progress=progress)
    save(tmp_path/'result.json',packet(result));save(tmp_path/'checkpoint.json',result.checkpoint)
    assert result.status=='completed',result.failure
    fresh,_=make();chain,records=decode_checkpoint(fresh,program,result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    rows=[];prior=None
    for snapshot,record in zip(chain[1:],records):
        u=snapshot['displacements'].reshape(5,6);drop=float(-u[2,1]);assert 0.<drop<=.2
        prior=continuum.solve(drop,previous=prior,profile='BVP9')
        zero_history=all(not any(sum(v) for v in s.plastic) and sum(s.accumulated)==0.
            for state in snapshot['states'].values() for s in state['response'].history.stations)
        assert zero_history
        symmetry=float(max(np.max(np.abs(u[:,0]+u[::-1,0])),np.max(np.abs(u[:,1]-u[::-1,1])),np.max(np.abs(u[:,2]))))
        rows.append(dict(drop=drop,native_density=record['parameter'],reference_density=prior.density,
            relative_load_error=abs(record['parameter']/prior.density-1),reference_slope=prior.slope,
            reflection_error=symmetry,zero_plastic_history=True,reference=asdict(prior)))
    save(tmp_path/'comparison.json',dict(rows=rows,coarse_diagnostic_only=True,production_qualified=False,
        full_spatial_stability=False,beam_postbuckling_qualified=False))


def test_reference_early_turning_region(tmp_path):
    rows=[];previous=None
    for drop in (0.,.001,.002,.004,.006,.008,.01):
        previous=continuum.solve(drop,previous=previous,profile='BVP9');rows.append(asdict(previous))
    save(tmp_path/'reference-early.json',dict(rows=rows,production_qualified=False,full_spatial_stability=False))
    assert rows[0]['slope']>0 and rows[-1]['slope']<0
