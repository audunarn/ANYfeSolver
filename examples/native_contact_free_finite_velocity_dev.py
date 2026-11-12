"""Two-step contact-free finite-velocity DEVELOPMENT example, not qualification.

Run with the development tree on PYTHONPATH; no old public selector is broadened.
"""
import json
import numpy as np
from anyfesolver.ge_beam3_native import (
    define_beam, SELECTOR as DEFINITION_SELECTOR, EllipsoidalGeneralizedSection)
from anyfesolver.ge_beam3_native_dynamics import create_analysis, SELECTOR, workflow_provenance


def build_definition():
    """Existing curved/cross-inertia fixture, explicit SI parameters."""
    x = np.array((-1.,0.,1.))
    points = np.column_stack((x,.2*(1-x*x),np.zeros(3)))
    frames = []
    for value in x:
        tangent = np.array((1.,-.4*value,0.)); tangent /= np.linalg.norm(tangent)
        second = np.array((0.,0.,1.))
        frames.append(np.column_stack((tangent,second,np.cross(tangent,second))))
    d = np.diag(np.sqrt((30000.,10000.,30000./3.5,2.,1.,1.5)))
    b = np.eye(6); b[0,3],b[1,5] = .02,.03
    f = b@d
    section = EllipsoidalGeneralizedSection(f.T@f,np.eye(6),1e6,1.)
    mass = 2.7; center = np.array((.035,-.025,.012))
    a,c,d0 = center
    skew = np.array(((0.,-d0,c),(d0,0.,-a),(-c,a,0.)))
    shift = -mass*(skew@skew); shift = np.triu(shift)+np.triu(shift,1).T
    inertia = np.block([[mass*np.eye(3),-mass*skew],
                        [mass*skew,np.diag((.12,.17,.19))+shift]])
    return define_beam(DEFINITION_SELECTOR,1,(1,2,3),points,np.array(frames),
                       section,inertia,quadrature=4), points


def initial_velocity(points):
    translation = np.array((.1,-.04,.03)); spin = np.array((.08,-.04,.05))
    return np.concatenate(((translation+np.cross(spin,points)).ravel(),spin,spin))


def main():
    definition, points = build_definition()
    analysis = create_analysis(SELECTOR,(definition,),())
    state = analysis.initialize(points,np.tile(np.eye(3),(2,1,1)),initial_velocity(points))
    for _ in range(2):
        candidate = analysis.trial(state,dt=.001)
        analysis.candidate_stage_audit(candidate)  # Evidence only, never credentials.
        state = analysis.commit(candidate)
    result = analysis.endpoint_accounting(state)
    print(json.dumps(dict(provenance=workflow_provenance(analysis),
        time_seconds=state.time,epoch=state.epoch,total_energy_j=result.total_energy)))


if __name__ == '__main__':
    main()
