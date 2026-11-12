"""Explicit finite-profile rigid-frame seed; no free coordinate is constrained."""
from __future__ import annotations
import numpy as np
from anyfesolver.corotational import rotation_matrix_from_vector,rotation_vector_from_matrix

_DOF=dict(ux=0,uy=1,uz=2,rx=3,ry=4,rz=5)


def finite_surface_partitions(model, definition):
    """Explicit complete structural components, never geometric body discovery."""
    if len(definition.surfaces) != 2:
        raise ValueError('per_surface_reference_frames_dev requires two named surfaces')
    partitions=[];all_elements=set();all_nodes=set()
    for surface in sorted(definition.surfaces,key=lambda s:s.name):
        ids=set(surface.element_ids)
        if not ids or len(ids)!=len(surface.element_ids) or not ids<=set(model.mesh.elements):
            raise ValueError('finite frame partition has missing/repeated elements')
        if ids & all_elements:
            raise ValueError('finite frame partitions overlap in elements')
        elements={eid:set(model.mesh.elements[eid].node_ids) for eid in ids}
        nodes=set().union(*elements.values())
        if not nodes<=set(model.mesh.nodes) or nodes & all_nodes:
            raise ValueError('finite frame partitions have missing/shared nodes')
        reached={min(ids)};front=list(reached)
        while front:
            current=front.pop()
            for eid in ids-reached:
                if elements[eid] & elements[current]:
                    reached.add(eid);front.append(eid)
        if reached!=ids:
            raise ValueError('finite frame surface is not one connected structural component')
        partitions.append(dict(name=surface.name,element_ids=sorted(ids),node_ids=sorted(nodes)))
        all_elements.update(ids);all_nodes.update(nodes)
    if (len({p['name'] for p in partitions})!=2 or all_elements!=set(model.mesh.elements)
            or all_nodes!=set(model.mesh.nodes)):
        raise ValueError('finite frame partitions must uniquely cover every structural element and node')
    dofs=[int(d) for nid in sorted(all_nodes) for d in model.mesh.dof_manager.get_node_dofs(nid)]
    if len(dofs)!=len(set(dofs)) or set(dofs)!=set(range(model.mesh.dof_manager.total_dofs)):
        raise ValueError('finite frame partitions must cover every independent structural DOF')
    return tuple(partitions)


def per_surface_seed_descriptor(model, definition, program):
    partitions=finite_surface_partitions(model,definition)
    components={}
    for control in program.controls:
        if len(control.terms)!=1 or control.terms[0][2]!=1.:
            raise ValueError('per-surface frame seed requires unit nodal controls')
        nid,name,_=control.terms[0]
        if nid not in model.mesh.nodes or name not in _DOF:
            raise ValueError('per-surface frame control is not an owned nodal coordinate')
        component=_DOF[name]
        if component in components.setdefault(nid,set()):
            raise ValueError('per-surface frame has repeated coordinate controls')
        components[nid].add(component)
    descriptors=[]
    for partition in partitions:
        anchors=sorted(n for n in partition['node_ids'] if components.get(n)==set(range(6)))
        if not anchors:
            raise ValueError('each finite surface requires a complete six-coordinate frame anchor')
        descriptors.append(dict(**partition,anchor_node_id=anchors[0]))
    return tuple(descriptors)


def per_surface_rigid_seed(model, program, target, transform, offset, start_q, rows, descriptor):
    """LEFT transport committed free deformation by each declared body frame."""
    from .static_path import _control_projection_seed
    targets={}
    for control,value in zip(program.controls,target):
        nid,name,_=control.terms[0];targets.setdefault(nid,{})[_DOF[name]]=float(value)
    start_u=np.asarray(transform@start_q+offset).reshape(-1);transported=start_u.copy()
    gamma=512*np.finfo(float).eps/(1.-512*np.finfo(float).eps);history=[]
    for partition in descriptor:
        anchor=partition['anchor_node_id'];values=targets[anchor];node=model.mesh.nodes[anchor]
        x=np.array((node.x,node.y,node.z));theta=np.array([values[i] for i in (3,4,5)])
        rtarget=rotation_matrix_from_vector(theta)
        translation=x+np.array([values[i] for i in (0,1,2)])-rtarget@x
        for nid in partition['node_ids']:
            node=model.mesh.nodes[nid];reference=np.array((node.x,node.y,node.z))
            predicted=rtarget@reference+translation-reference
            for component,value in targets.get(nid,{}).items():
                if component<3:
                    magnitude=(np.abs(rtarget)@np.abs(reference)+np.abs(translation)+np.abs(reference))[component]
                    valid=abs(predicted[component]-value)<=gamma*max(1.,magnitude)
                else:
                    valid=abs(value-theta[component-3])<=gamma*max(1.,abs(value),abs(theta[component-3]))
                if not valid:
                    raise ValueError('finite surface controls do not declare one consistent reference frame')
        anchor_dofs=model.mesh.dof_manager.get_node_dofs(anchor)
        changed=not np.array_equal(start_u[anchor_dofs[3:6]],theta)
        increment=(rtarget@rotation_matrix_from_vector(start_u[anchor_dofs[3:6]]).T
                   if changed else np.eye(3))
        shift=x+np.array([values[i] for i in (0,1,2)])-increment@(x+start_u[anchor_dofs[:3]])
        translated=bool(np.any(shift!=0.))
        if changed or translated:
            for nid in partition['node_ids']:
                node=model.mesh.nodes[nid];reference=np.array((node.x,node.y,node.z))
                dofs=model.mesh.dof_manager.get_node_dofs(nid)
                transported[dofs[:3]]=(increment@(reference+start_u[dofs[:3]])+shift-reference
                                      if changed else start_u[dofs[:3]]+shift)
                if changed:
                    transported[dofs[3:6]]=rotation_vector_from_matrix(
                        increment@rotation_matrix_from_vector(start_u[dofs[3:6]]))
        history.append(dict(name=partition['name'],anchor_node_id=anchor,
                            transported_free_nodal_frames=changed,transported_free_midsurfaces=changed or translated))
    reduced=np.asarray(transform.T@(transported-offset)).reshape(-1)
    return _control_projection_seed(rows,reduced,target),dict(
        surfaces=history,method='per_surface_left_transport_project_controls_v1')


def finite_rigid_seed(model, program, target, transform, offset, start_q, rows):
    """Transport committed deformation by the declared common rigid frame.

    Every declaration must be a unit nodal control. A complete six-coordinate
    anchor fixes the declared frame/translation; the reference control targets
    must all correspond to that same frame. Free nodal Qi are LEFT transported,
    retaining their local deformation/drill rather than overwritten by anchor
    theta. The control projection changes only prescribed coordinates. The
    caller must certify the complete binary committed-to-projected-seed path.
    """
    from .static_path import _control_projection_seed
    targets={}
    for control,value in zip(program.controls,target):
        if len(control.terms)!=1 or control.terms[0][2]!=1.:
            raise ValueError('finite rigid seed requires unit nodal controls')
        node,name,_=control.terms[0];targets.setdefault(node,{})[_DOF[name]]=float(value)
    anchors=sorted(n for n,v in targets.items() if set(v)==set(range(6)))
    if not anchors:
        raise ValueError('finite rigid seed requires a complete six-coordinate anchor')
    anchor=anchors[0];values=targets[anchor];node=model.mesh.nodes[anchor]
    x=np.array((node.x,node.y,node.z));theta=np.array([values[i] for i in (3,4,5)])
    rtarget=rotation_matrix_from_vector(theta)
    translation=x+np.array([values[i] for i in (0,1,2)])-rtarget@x
    gamma=512*np.finfo(float).eps/(1.-512*np.finfo(float).eps)
    for nid,declared in targets.items():
        node=model.mesh.nodes[nid];reference=np.array((node.x,node.y,node.z))
        predicted=rtarget@reference+translation-reference
        for component,value in declared.items():
            if component<3:
                magnitude=(np.abs(rtarget)@np.abs(reference)+np.abs(translation)+np.abs(reference))[component]
                if abs(predicted[component]-value)>gamma*max(1.,magnitude):
                    raise ValueError('finite rigid seed controls do not declare one common reference frame')
            else:
                if abs(value-theta[component-3])>gamma*max(1.,abs(value),abs(theta[component-3])):
                    raise ValueError('finite rigid seed rotational controls are not one common frame')
    start_u=np.asarray(transform@start_q+offset).reshape(-1)
    anchor_dofs=model.mesh.dof_manager.get_node_dofs(anchor)
    rotation_changed=not np.array_equal(start_u[anchor_dofs[3:6]],theta)
    rstart=rotation_matrix_from_vector(start_u[anchor_dofs[3:6]])
    increment=rtarget@rstart.T if rotation_changed else np.eye(3)
    current_anchor=x+start_u[anchor_dofs[:3]]
    shift=x+np.array([values[i] for i in (0,1,2)])-increment@current_anchor
    transported=start_u.copy()
    # Identity pose updates during force phases retain the current free chart,
    # avoiding an unnecessary Exp/Log round trip of committed rotations.
    translation_changed=bool(np.any(shift!=0.))
    if rotation_changed or translation_changed:
        for nid,node in model.mesh.nodes.items():
            dofs=model.mesh.dof_manager.get_node_dofs(nid)
            reference=np.array((node.x,node.y,node.z))
            transported[dofs[:3]]=(increment@(reference+start_u[dofs[:3]])+shift-reference
                                  if rotation_changed else start_u[dofs[:3]]+shift)
            if rotation_changed:
                qi=rotation_matrix_from_vector(start_u[dofs[3:6]])
                transported[dofs[3:6]]=rotation_vector_from_matrix(increment@qi)
    reduced=np.asarray(transform.T@(transported-offset)).reshape(-1)
    seed=_control_projection_seed(rows,reduced,target)
    return seed,dict(anchor_node_id=anchor,transported_free_nodal_frames=rotation_changed,
                     transported_free_midsurfaces=rotation_changed or translation_changed,
                     common_translation_changed=translation_changed,
                     method='common_reference_frame_left_transport_project_controls_v1')
