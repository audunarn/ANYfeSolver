"""Private accepted physical recovery; never advances material history."""
import numpy as np
from ._ge_beam3_native_distributed_element import NativeDistributedFibreStaticElement,FORMULATION,SCHEMA
from ._ge_beam3_p5_seeded.core import canonical
from ._ge_beam3_p5.compensated_coordinates import split_sum
from ._native_reference_modal import _owned


def _sum(values):
    return split_sum(tuple(float(value) for value in values))


def recover_native_fields(element,mesh,state,*,expected_committed_total_u=None):
    if type(element) is not NativeDistributedFibreStaticElement: raise ValueError('exact native physical fibre recovery element')
    element._validate(mesh,state,expected_committed_total_u)
    before=canonical(state); response=state['response']; op=element.operator
    stations=op.recover(response.rotations,response.resultants,origin=state['origins']); rows=[]
    for station in stations:
        cell=station['cell']; t=station['xi']-(cell-1); lift=op.reference.half_cell_lift(cell,t)
        high=np.empty(3); low=np.empty(3)
        for axis in range(3):
            difference=_sum((state['positions'][cell+1,axis],state['position_low'][cell+1,axis],-state['positions'][cell,axis],-state['position_low'][cell,axis]))
            high[axis],low[axis]=_sum((state['positions'][cell,axis],state['position_low'][cell,axis],t*difference[0],t*difference[1],*(response.rotations[cell,axis,k]*lift[k] for k in range(3))))
        gh=np.empty(6); gl=np.empty(6); frame=station['current_frame']
        for block in (0,3):
            for axis in range(3):
                gh[block+axis],gl[block+axis]=_sum(tuple(frame[axis,k]*station['resultants'][block+k] for k in range(3))+tuple(frame[axis,k]*station['resultants_low'][block+k] for k in range(3)))
        rows.append({**station,'current_position':_owned(high),'current_position_low':_owned(low),
            'global_resultants':_owned(gh),'global_resultants_low':_owned(gl)})
    if canonical(tuple(row['history'] for row in rows))!=canonical(response.history.stations) or canonical(state)!=before:
        raise ValueError('native recovery changed or disagrees with accepted history')
    return dict(formulation_id=FORMULATION,state_schema=SCHEMA,load_pattern_sha256=state['load_pattern'].signature,state_sha256=state['state_sha256'],element_id=element.element_id,
        strain_order=('eps_x','gamma_xy','gamma_xz','kappa_x','kappa_y','kappa_z'),resultant_order=('N','V_y','V_z','T','M_y','M_z'),
        fibre_stress_status='PHYSICAL_SECTION_SUPPLIED_PAIRED_STRESSES',stations=tuple(rows),production_qualified=False)
