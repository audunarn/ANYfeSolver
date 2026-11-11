"""Original matched174 fixture plus64 explicit physical B4 coordinates.

Unchanged original builders, .04N load, two zero ux controls, all25 pressures.
This is a development response caller, never a transfer of the original seven
independent-reference gates or full D1 acceptance. No saved solution is used.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

from anyfesolver.e4_pl_enriched import EnrichedProfileError
from anyfesolver.general_contact.enriched_overlap import QuadraturePolicy
from anyfesolver.general_contact.enriched_nonlinear import (
    NonlinearEnrichedSystem,PairDefinition,MixedCriteria,IterationControl,
)

INPUT=Path(__file__).resolve().parents[1]/'docs/d1_review_support/d1_actual_cutback_single_press_20260929.json'
DOF={'ux':0,'uy':1,'uz':2,'rx':3,'ry':4,'rz':5}


def build(*,quadrature=QuadraturePolicy(1e-10,1e-8,1e-5,14144)):
    from scripts.general_contact_d1_interior_press_dev import build_saved_cycle
    saved=json.loads(INPUT.read_text(encoding='utf-8'))
    saved['source_reference_downshift_m']=0.
    model,definition,settings,program=build_saved_cycle(saved)
    n=model.mesh.dof_manager.total_dofs
    source,target=definition.surfaces
    if (n!=174 or len(source.element_ids)!=16 or len(target.element_ids)!=2
            or program.total_steps!=1 or saved['force_n']!=.04
            or len(program.controls)!=2 or np.any(program.target_at(1)!=0)):
        raise EnrichedProfileError('original registered matched inventory/control contract required')
    side={'bottom':-1.,'midsurface':0.,'top':1.}
    pairs=[]
    for si in source.element_ids:
        se=model.mesh.elements[si]
        source_offset=side[source.side]*se.thickness/2
        for ti in target.element_ids:
            te=model.mesh.elements[ti]
            target_offset=te.material_surface_offset_from_reference(side[target.side])
            pairs.append(PairDefinition(si,ti,float(source_offset),float(target_offset)))
    controls=np.zeros((len(program.controls),n))
    for i,control in enumerate(program.controls):
        if control.coordinate_profile!='translation_v1':
            raise EnrichedProfileError('original additive translation controls required')
        for node,name,weight in control.terms:
            controls[i,model.mesh.dof_manager.get_node_dofs(node)[DOF[name]]]+=weight
    system=NonlinearEnrichedSystem.from_model(model,pairs,
        normal_stiffness=definition.interactions[0].normal_stiffness,quadrature=quadrature,
        control_matrix=controls,control_values=np.asarray(program.target_at(1)))
    if (system.offset.size!=238 or system.transform.shape!=(238,184)
            or len(system.pressure_ids)!=25 or system.pressure_ids!=tuple(range(1,26))):
        raise EnrichedProfileError('174+64 coordinates,120+64 free and all25 pressures required')
    load=np.zeros(238)
    for force,value in zip(program.nodal_forces,program.force_values_at(1),strict=True):
        load[model.mesh.dof_manager.get_node_dofs(force.node_id)[DOF[force.dof]]]+=value
    if not np.isclose(np.sum(load[2:174:6]),-.04,rtol=0,atol=32*np.finfo(float).eps*.04):
        raise EnrichedProfileError('unchanged original .04N nodal load required')
    return model,system,load,dict(input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        source_reference_downshift_m=0.,physical_coordinates=174,added_coordinates=64,
        free_coordinates=184,pressure_ids=list(system.pressure_ids),control_matrix=controls.tolist(),
        control_values=list(program.target_at(1)),source_side=source.side,target_side=target.side,
        control_scales=np.maximum(1e-9,np.max(np.abs(np.asarray(program.waypoints)),axis=0)).tolist(),
        offsets=[vars(p) for p in pairs],original_reference_active_node=12,
        independent_reference_mapping='REQUIRED_UNVERIFIED',scientific_acceptance=False)


def solve_matched(*,cancel=lambda:False):
    model,system,load,descriptor=build()
    result=system.solve(load,criteria=MixedCriteria(.04,1e-6,1e-8),
                        control=IterationControl(18,8),cancel=cancel)
    accepted=system._accepted
    raw=system.linearize(accepted.coordinates,accepted.nodal_forces,cancel=cancel)
    internal,_,_=system.structural_response(accepted.coordinates,cancel=cancel)
    result.update(descriptor=descriptor,coordinates=accepted.coordinates.tolist(),
        nodal_forces=accepted.nodal_forces.tolist(),pressure=raw.pressure.tolist(),
        normalized_gap=raw.normalized_gap.tolist(),support=raw.support.tolist(),
        contact_force=raw.force.tolist(),structural_force=internal.tolist(),load=load.tolist(),
        element_states=system.recover_element_states(cancel=cancel),
        control_error=float(np.max(np.abs((np.asarray(descriptor['control_matrix']) @ accepted.coordinates[:174]
            -np.asarray(descriptor['control_values']))/np.asarray(descriptor['control_scales'])))),
        reference_active_branch_matches=bool(accepted.nodal_forces[11]>0 and
            np.all(np.delete(accepted.nodal_forces,11)==0)),
        old_seven_independent_gates='REQUIRED_UNVERIFIED',scientific_acceptance=False)
    witness=sampled_physical_gap(model,system,accepted.coordinates)
    result['independent_sampled_physical_gap']=witness
    if witness['original_target_plane_sample_minimum_m'] < -5e-5:
        raise EnrichedProfileError('independent sampled physical penetration exceeds original -50um floor')
    return model,system,result


def sampled_physical_gap(model,system,coordinates):
    """Independent one-way witness, never a global minimum/certificate.

    Retain the ORIGINAL independent_pairs local physical source normal and
    globalXY correspondence. Also evaluate the current production corner-frame
    correspondence independently; no production moments/AD calls. Both gap
    quantities are explicitly labelled, rather than replacing them by target
    plane distance. The old target-plane sample is separate diagnostic data.
    """
    from anyfesolver.general_contact.q4_skin_work import Q4SkinWorkAdapter
    from anyfesolver.general_contact.s3_skin_work import S3SkinWorkAdapter
    u=np.asarray(coordinates,dtype=float)
    if u.shape!=system.offset.shape or not np.all(np.isfinite(u)):
        raise EnrichedProfileError('finite complete sampled physical pose required')
    target_ids=sorted({p.definition.target_element for p in system.pairs})
    targets=[]
    for element_id in target_ids:
        adapter=S3SkinWorkAdapter.from_model(model,2,element_id,'top')
        vertices=adapter.current_positions(u[:system.physical_dofs])
        normal=np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0]);normal/=np.linalg.norm(normal)
        if not np.all(np.isfinite(normal)) or normal[2]<=.5:
            raise EnrichedProfileError('original physical target normal domain unsupported')
        targets.append((element_id,vertices,normal))
    minimum=float('inf');witness=None;plane_minimum=float('inf')
    minima={'ORIGINAL_GLOBAL_XY_LOCAL_SOURCE_NORMAL':float('inf'),
            'CURRENT_CORNER_FRAME_LOCAL_SOURCE_NORMAL':float('inf')}
    represented={key:0 for key in minima}
    for cell,indices in zip(system.cells,system.cell_indices):
        adapter=Q4SkinWorkAdapter.from_model(model,1,cell.element_id,'bottom')
        skin=adapter.current_positions(u[:system.physical_dofs])
        mid=cell.reference+u[np.asarray(indices[:24])].reshape(4,6)[:,:3]
        body=np.cross(mid[1]-mid[0],mid[3]-mid[0]);body/=np.linalg.norm(body)
        tangent=skin[1]-skin[0];tangent/=np.linalg.norm(tangent)
        corner_normal=np.cross(skin[1]-skin[0],skin[3]-skin[0]);corner_normal/=np.linalg.norm(corner_normal)
        projections={'ORIGINAL_GLOBAL_XY_LOCAL_SOURCE_NORMAL':np.eye(3)[:,:2],
            'CURRENT_CORNER_FRAME_LOCAL_SOURCE_NORMAL':np.column_stack((tangent,np.cross(corner_normal,tangent)))}
        beta=u[np.asarray(indices[24:])]
        for r in (-1.,-.5,0.,.5,1.):
            for s in (-1.,-.5,0.,.5,1.):
                hats=np.array(((1-r)*(1-s),(1+r)*(1-s),(1+r)*(1+s),(1-r)*(1+s)))/4
                nr=np.array((-(1-s),1-s,1+s,-(1+s)))/4
                ns=np.array((-(1-r),-(1+r),1+r,1-r))/4
                b=(1-r*r)*(1-s*s)
                br=np.array((-2*r*(1-s*s),(1-3*r*r)*(1-s*s),-2*r*s*(1-s*s),s*(1-3*r*r)*(1-s*s)))
                bs=np.array((-2*s*(1-r*r),-2*s*r*(1-r*r),(1-3*s*s)*(1-r*r),r*(1-3*s*s)*(1-r*r)))
                point=hats @ skin+b*(beta @ np.array((1.,r,s,r*s)))*body
                pr=nr @ skin+(br @ beta)*body;ps=ns @ skin+(bs @ beta)*body
                normal=np.cross(pr,ps);normal/=np.linalg.norm(normal)
                if not np.all(np.isfinite(normal)) or normal[2]<=0:
                    raise EnrichedProfileError('original sourceXY normal/projection domain unsupported')
                for label,frame in projections.items():
                    covered=False
                    for element_id,vertices,inward in targets:
                        jac=frame.T @ np.column_stack((vertices[1]-vertices[0],vertices[2]-vertices[0]))
                        inverse=np.linalg.inv(jac)
                        condition=np.linalg.norm(jac,ord=np.inf)*np.linalg.norm(inverse,ord=np.inf)
                        if not np.isfinite(condition) or condition>1e6:
                            raise EnrichedProfileError('original physical gap projection is ill-conditioned')
                        projected=inverse @ (frame.T @ (point-vertices[0]))
                        bary=np.array((1-np.sum(projected),*projected))
                        scale=max(1.,float(np.max(np.abs(point))),float(np.max(np.abs(vertices))))
                        bound=128*np.finfo(float).eps*(1+np.linalg.norm(inverse,ord=np.inf)*scale)
                        if np.min(bary)<-bound:continue
                        covered=True
                        target_point=bary @ vertices
                        gap=float((point-target_point) @ normal)
                        if not np.isfinite(gap):raise EnrichedProfileError('nonfinite independent physical gap witness')
                        minima[label]=min(minima[label],gap)
                        if label=='ORIGINAL_GLOBAL_XY_LOCAL_SOURCE_NORMAL':
                            plane_minimum=min(plane_minimum,float((point-vertices[0]) @ inward))
                        if gap<minimum:
                            minimum=gap;witness=dict(projection=label,source_element=cell.element_id,
                                target_element=element_id,r=r,s=s,point_m=point.tolist(),
                                local_source_normal=normal.tolist(),target_point_m=target_point.tolist(),
                                target_barycentric=bary.tolist())
                    if not covered:
                        raise EnrichedProfileError('independent source sample has no actual target projection')
                    represented[label]+=1
    return dict(minimum_sampled_gap_m=minimum,witness=witness,minimum_by_projection=minima,
        original_target_plane_sample_minimum_m=plane_minimum,
        represented_source_points=min(represented.values()),represented_by_projection=represented,
        samples_per_source=25,original_floor_m=-5e-5,
        predicate='ACTUAL_LOCAL_ENRICHED_SOURCE_NORMAL_DOT_SOURCE_MINUS_PROJECTED_TARGET',
        target_plane_sample='SEPARATE_ORIGINAL_INDEPENDENT_GAP_BOUND_PREDICATE; NOT_SOURCE_NORMAL_GAP',
        original_floor_predicate='TARGET_PLANE_SIGNED_DISTANCE_WITH_ORIGINAL_XY_FOOTPRINT',
        source_normal_samples='LABELLED_DIAGNOSTICS_ONLY; NO_ADDITIONAL_POINTWISE_FLOOR_GATE',
        full_global_gap='UNVERIFIED; samples can falsify only',scientific_acceptance=False)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    _,system,result=solve_matched()
    with args.output.open('x',encoding='utf-8')as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    args.output.with_suffix('.checkpoint.json').write_bytes(system.checkpoint())
