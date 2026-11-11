"""Development reference-buckling workflow; no qualification claim."""
import json
import numpy as np
from scipy import sparse
from anyfesolver import stiffened_shell as api

def traction_source(member_id="beam", source_id="explicit-square-gauss-traction-v1"):
    a=.1/np.sqrt(3.)
    yz=np.array(((a,a),(a,-a),(-a,a),(-a,-a)))
    weights=np.full(4,.01); area=.04; moment=.04*.2**2/12.; polar=2*moment
    mapping=np.zeros((4,3,6))
    for i,(y,z) in enumerate(yz):
        mapping[i,0,(0,4,5)]=(1/area,z/moment,-y/moment)
        mapping[i,1,(1,3)]=(1/area,-z/polar)
        mapping[i,2,(2,3)]=(1/area,y/polar)
    return api.B2SectionTractionQuadrature(member_id,source_id,yz,weights,mapping)

def traction_sources():
    return (traction_source(),traction_source("beam-right","explicit-square-right-gauss-traction-v1"))

def definition(kind="q4"):
    nodes={1:(0.,0.,0.),2:(1.,0.,0.),3:(2.,0.,0.),4:(0.,1.,0.),5:(1.,1.,0.),6:(2.,1.,0.)}
    shells=[]
    for i,ids in enumerate(((1,2,5,4),(2,3,6,5))):
        if kind == "s3" or kind == "mixed" and i == 1:
            for j,tri in enumerate(((ids[0],ids[1],ids[2]),(ids[0],ids[2],ids[3]))):
                shells.append(api.S3V2DPanelShell(f"shell-{i}-{j}",tri,1000.,.3,.15,(0.,0.,1.)))
        else: shells.append(api.Q4PanelShell(f"shell-{i}-0",ids,1000.,.3,.15))
    section=api.LinearSection.isotropic(E=2000.,nu=.25,area=.04,Iy=.04*.2**2/12.,Iz=.04*.2**2/12.,J=2*.04*.2**2/12.,ky=.8,kz=.8)
    member=api.PhysicalB2Member("beam","explicit-square-member-v1",(10,11),((.5,.5,0.),(1.5,.5,0.)),(0.,1.,0.),"square",section)
    right=api.PhysicalB2Member("beam-right","explicit-square-right-member-v1",(11,12),((1.5,.5,0.),(2.5,.5,0.)),(0.,1.,0.),"square",section)
    points=(api.Q4B2Point("left","shell-0-0","beam",0.),api.Q4B2Point("right","shell-1-0","beam-right",0.))
    fixed=set(range(6))|set(range(18,24))
    # Keep membrane preload strictly separate from unsupported shell shear.
    for i in range(6): fixed.update(6*i+j for j in (2,3,4,5))
    free=[i for i in range(54) if i not in fixed]
    return api.B2MixedPanelDefinition(api.B2_E4_PL_FAST_LINEAR_V1,nodes,tuple(shells),(member,right),points,sparse.eye(54,format="csr")[:,free],np.zeros(54))

def workflow():
    owner=api.create_analysis(definition=definition())
    try:
        load=np.zeros(54);load[[12,30]]=-1.
        result=owner.buckling(api.B2ReferenceBucklingRequest(load,traction_sources()))
        return {"status":result.status,"load_factors":result.load_factors,"failure_reason":result.failure_reason,"policy":result.policy,"production_qualified":False}
    finally: owner.close()

if __name__ == "__main__": print(json.dumps(workflow(),indent=2))
