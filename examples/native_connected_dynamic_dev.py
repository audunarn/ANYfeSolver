"""Two genuinely connected native elements; source-only DEVELOPMENT example."""
import argparse
import json
import numpy as np
from anyfesolver.ge_beam3_native import define_beam, SELECTOR as BEAM, EllipsoidalGeneralizedSection
from anyfesolver.ge_beam3_native_connected_dynamics import create_analysis, SELECTOR


def build(clamped=False):
    section=EllipsoidalGeneralizedSection(np.diag((1e5,3e4,3e4,30.,20.,10.)),np.eye(6),1e6,1.)
    inertia=np.diag((2.,2.,2.,.02,.01,.01))
    definitions=tuple(define_beam(BEAM,k,nodes,np.array(points,dtype=np.float64),
        np.tile(np.eye(3),(3,1,1)),section,inertia,quadrature=4)
        for k,nodes,points in ((1,(10,20,30),((-1.,0.,0.),(-.5,0.,0.),(0.,0.,0.))),
                              (2,(30,40,50),((0.,0.,0.),(.5,0.,0.),(1.,0.,0.)))))
    boundaries=()
    if clamped:
        from anyfesolver.boundary import BoundaryCondition
        boundaries=(BoundaryCondition('root',[10],dict.fromkeys(('ux','uy','uz','rx','ry','rz'),0.)),)
    return create_analysis(SELECTOR,definitions,boundaries)


def velocity(owner):
    # Physical order: sorted-node translations then element/cell spatial spins.
    return np.array((.01,.02,0.)*owner.n+(2.,0.,0.)*(2*owner.e),dtype=np.float64)


def run(steps=4,dt=.00025):
    owner=build();state=owner.initialize(physical_velocity=velocity(owner));records=[]
    for _ in range(steps):
        candidate=owner.trial(state,dt=dt)
        prepared=dict(time=candidate.endpoint.time,accounting=candidate.accounting)
        # All serializable publication data prepare before commit.
        json.dumps(prepared,default=lambda x:x.tolist(),allow_nan=False)
        state=owner.commit(candidate);records.append(prepared)
    return owner,state,records


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--steps',type=int,default=4)
    args=parser.parse_args();owner,state,records=run(args.steps)
    print(json.dumps(dict(profile='GE_BEAM3_CONNECTED_CENTERED_ELASTIC_MIDPOINT_DEV_V1',
        elements=owner.e,shared_nodes=owner.n,accepted_time=state.time,records=records,
        checkpoint_bytes=len(owner.checkpoint()),engineering_accepted=False),default=lambda x:x.tolist()))


if __name__=='__main__':main()
