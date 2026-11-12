"""One decision-oriented coupled failure trace; no mechanics-pass claim."""
import json,os
from pathlib import Path
import numpy as np
import pytest
from test_general_contact_native_material_plane import _analysis


@pytest.mark.skipif(not os.environ.get('CONTACT_MATERIAL_DIAGNOSIS_RECORD'),
                    reason='explicit diagnostic output required; preserve the failed run')
def test_capture_coupled_gap_merit_and_work_map_discriminator():
    owner=_analysis(normal_stiffness=1e5,equilibrium_tolerance=1e-8);records=[];swept=[]
    assemble=owner._assemble;certify=owner._swept_admissible
    def observed(mechanical,parameter,origins):
        result=assemble(mechanical,parameter,origins)
        r,j,metrics,_,contact=result
        records.append(dict(parameter=parameter,metrics=metrics,
            gap_lower_m=contact.minimum_gap_m,gap_estimate_m=contact.minimum_gap_estimate_m,
            force_n=contact.beam_force_n.tolist(),residual=r.tolist(),
            positions=mechanical.positions.tolist(),rotations=mechanical.cell_rotations.tolist(),
            descriptor=mechanical.descriptor()))
        return result
    def observed_sweep(start,increment,end,first,last):
        result=certify(start,increment,end,first,last)
        swept.append(dict(first_gap_m=first,last_gap_m=last,admissible=result,
                          maximum_correction=float(np.max(np.abs(increment)))))
        return result
    owner._assemble=observed;owner._swept_admissible=observed_sweep
    run=owner.solve()
    # Check the assembled residual/tangent at the last evaluated pose through
    # the native owner's own incremental coordinate map, not a new solver.
    state=owner.context.make(records[-1]['descriptor'])
    r,j,*_=assemble(state,records[-1]['parameter'],owner.initial.histories)
    direction=np.random.default_rng(862).normal(size=owner.context.count)*.001
    direction[list(owner.context.fixed)]=0.
    fd_errors=[]
    for step in (1e-5,2e-6):
        plus=owner.context.advance(state,step*direction)
        minus=owner.context.advance(state,-step*direction)
        difference=(assemble(plus,records[-1]['parameter'],owner.initial.histories)[0]
                    -assemble(minus,records[-1]['parameter'],owner.initial.histories)[0])/(2*step)
        fd_errors.append(float(np.linalg.norm(j@direction-difference)/np.linalg.norm(difference)))
    path=os.environ.get('CONTACT_MATERIAL_DIAGNOSIS_RECORD')
    assert path and records
    # JSON-safe native canonical payload, including full failed state for review.
    from anyfesolver._ge_beam3_p5_seeded.core import canonical
    Path(path).write_bytes(canonical(dict(scope='diagnosis, not accepted equilibrium',
        status=run.status,failure=run.failure,accepted_targets=run.accepted.completed_targets,
        gap_tolerance_m=owner.gap_tolerance,fd_errors=fd_errors,records=records,swept=swept)))
