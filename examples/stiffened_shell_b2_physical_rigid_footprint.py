"""Development-only physical B2 rigid-footprint static / buckling workflow."""
import json
import numpy as np
from scipy import sparse
from anyfesolver import stiffened_shell as api
from examples.stiffened_shell_b2_reference_buckling import traction_sources

def definition():
    nodes={1:(0.,0.,0.),2:(1.,0.,0.),3:(2.,0.,0.),4:(0.,1.,0.),5:(1.,1.,0.),6:(2.,1.,0.)}
    shells=tuple(api.Q4PanelShell(f"shell-{i}",ids,1000.,.3,.15) for i,ids in enumerate(((1,2,5,4),(2,3,6,5))))
    section=api.LinearSection.isotropic(E=2000.,nu=.25,area=.04,Iy=.04*.2**2/12.,Iz=.04*.2**2/12.,J=2*.04*.2**2/12.,ky=.8,kz=.8)
    members=tuple(api.PhysicalB2Member(name,"explicit-square-"+name,(10+i,11+i),((.5+i,.5,0.),(1.5+i,.5,0.)),(0.,1.,0.),"square",section) for i,name in enumerate(("beam","beam-right")))
    group=api.B2PhysicalRigidFootprint("physical-left",(1,2,5,4),10,(0.,0.,1.),((1.,0.),(0.,1.),(0.,0.)),((1,2,5,4),))
    free=[i for i in range(54) if i not in range(36,42)]
    return api.B2PhysicalRigidFootprintDefinition(api.B2_E4_PL_FAST_LINEAR_V1,nodes,shells,members,(group,),sparse.eye(54,format="csr")[:,free],np.zeros(54))

def preload():
    load=np.zeros(54);load[[12,30,48]]=-1.
    return load

def workflow():
    owner=api.create_analysis(definition=definition())
    try:
        static=owner.solve_linear(preload(),selected_footprints=api.B2PhysicalRigidFootprintSelection(("physical-left",)))
        spectral=owner.buckling(api.B2PhysicalFootprintBucklingRequest(preload(),traction_sources()))
        return dict(static_status=static.status,status=spectral.status,load_factors=spectral.load_factors,failure_reason=spectral.failure_reason,policy=spectral.policy,production_qualified=False)
    finally:owner.close()

def batch_workflow(count=10):
    owner=api.create_analysis(definition=definition())
    try:
        selection=api.B2PhysicalRigidFootprintSelection(("physical-left",))
        result=owner.solve_linear_many(tuple((f"case-{i}",(i+1)*preload()) for i in range(count)),
            selected_dofs=(12,13,36,41),selected_footprints=selection,recover_members={"beam-right":(0.,.5,1.)})
        return dict(status=result.status,case_count=len(result.cases),factorization_count=result.factorization_count,
            solve_count=result.solve_count,maximum_rhs_chunk=result.maximum_rhs_chunk,
            member_resultant_envelope_count=len(result.member_resultant_envelopes),
            footprint_law_id=result.footprint_law_id,production_qualified=False)
    finally:owner.close()

if __name__=="__main__":print(json.dumps(workflow(),indent=2))
