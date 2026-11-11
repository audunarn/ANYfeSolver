"""Exact elastic Q4/V2D finite-contact state composition; no migration."""
from __future__ import annotations
import copy
from collections.abc import Mapping
import numpy as np

PROFILE='Q4_S3_V2D_COROTATIONAL_NODAL_DIRECTOR_MONOLITHIC_PATH_DEV_V0'
STATIC_PROFILE='Q4_S3_V2D_COROTATIONAL_NODAL_DIRECTOR_MONOLITHIC_STATIC_DEV_V0'
COORDINATES='typed_q4_reference_deformational24_v2d_global18_constitutive18_v1'
V2D_COORDINATES='v2d_global_committed18_corotational_constitutive18_v1'

def admitted_definition(model,definition,settings):
    """Exact opt-in pair; this does not broaden any existing finite profile."""
    from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
    from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
    if (settings.shell_skin_kinematics!='corotational_nodal_director_dev'
            or settings.general_s3_differentiation!='jet2_q4_s3_dev'
            or settings.enforcement_solver!='monolithic_active_set'
            or settings.shell_pair_orientation!='physical_sides_dev'
            or settings.q4_source_gap_certificate!='whole_face'
            or settings.q4_source_pressure_basis!='fine_nodal_q1'
            or settings.augmentation_support_resolution!='exact'
            or settings.augmentation_acceleration!='none'
            or settings.q4_s3_shared_edge_tie_step_m is not None
            or settings.path_predictor!='none' or settings.d1_interior_profile
            or definition.rigid_bodies or len(definition.surfaces)!=2
            or len(definition.interactions)!=1):return False
    interaction=definition.interactions[0];names={s.name:s for s in definition.surfaces}
    if interaction.normal_law!='augmented' or interaction.surface==interaction.target:return False
    source=names.get(interaction.surface);target=names.get(interaction.target)
    if source is None or target is None:return False
    if (source.side not in {'top','bottom'} or target.side not in {'top','bottom'}
            or not source.element_ids or not target.element_ids
            or source.beam_radius is not None or target.beam_radius is not None
            or set(source.element_ids)&set(target.element_ids)
            or set(source.element_ids+target.element_ids)!=set(model.mesh.elements)
            or not all(type(model.mesh.elements.get(i)) is QualifiedE4PLShellElement for i in source.element_ids)
            or not all(type(model.mesh.elements.get(i)) is NativeParityE4PLS3V2DShellElement for i in target.element_ids)):return False
    try:
        owner_contract(model)
        from .finite_rotation_predictor import finite_surface_partitions
        finite_surface_partitions(model,definition)
    except (ValueError,TypeError):return False
    return True

def owner_contract(model):
    from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
    from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
    from anymaterial import IsotropicMaterial
    from .finite_rotation_state import COORDINATES as Q4_COORDINATES
    owners=[]
    for eid,element in sorted(model.mesh.elements.items()):
        material=model.get_material(element.material_name)
        if (type(material) is not IsotropicMaterial or material.yield_stress!=0.
                or material.hardening_curve is not None or element.shell_section is not None):
            raise ValueError('mixed finite state requires isotropic elastic owners without sections')
        if type(element) is QualifiedE4PLShellElement:
            owner='qualified_q4';coordinates=Q4_COORDINATES
        elif type(element) is NativeParityE4PLS3V2DShellElement:
            owner='native_parity_s3_v2d';coordinates=V2D_COORDINATES
        else:raise ValueError('mixed finite state requires exact Q4/V2D owners')
        owners.append(dict(element_id=int(eid),owner=owner,coordinates=coordinates))
    if not any(v['owner']=='qualified_q4' for v in owners) or not any(v['owner']=='native_parity_s3_v2d' for v in owners):
        raise ValueError('mixed finite state requires both exact owner families')
    return owners

def mixed_state_contract(contract):
    from .finite_rotation_state import finite_state_contract,PROFILE as Q4_PROFILE,COORDINATES as Q4_COORDINATES
    if contract.get('profile')!=PROFILE:return False
    if contract.get('element_state_coordinates')!=COORDINATES:
        raise ValueError('mixed finite coordinate contract changed')
    owners=contract.get('finite_state_owners')
    if (not isinstance(owners,list) or not owners
            or any(not isinstance(v,dict) or set(v)!={'element_id','owner','coordinates'} for v in owners)
            or any(type(v['element_id']) is not int for v in owners)
            or [v['element_id'] for v in owners]!=sorted(set(v['element_id'] for v in owners))
            or any((v['owner'],v['coordinates']) not in {('qualified_q4',Q4_COORDINATES),('native_parity_s3_v2d',V2D_COORDINATES)} for v in owners)
            or not any(v['owner']=='qualified_q4' for v in owners)
            or not any(v['owner']=='native_parity_s3_v2d' for v in owners)):
        raise ValueError('mixed finite typed owner inventory changed')
    legacy=dict(contract,profile=Q4_PROFILE,element_state_coordinates=Q4_COORDINATES)
    return finite_state_contract(legacy)

def local_coordinates(model,element,q):
    from anyfesolver.corotational import _corotational_cache,_corotational_deformation_state
    reference=_corotational_cache(model,centered_shell=False).get(int(element.element_id))
    if reference is None or reference.category!='shell':raise ValueError('mixed owner shell reference missing')
    local,_,_=_corotational_deformation_state(reference,element,q)
    return local

def prepare_mixed_states(model,global_q,states,num_layers,info,*,restart):
    from .finite_rotation_state import prepare_finite_states
    owners=owner_contract(model);by_id={v['element_id']:v['owner'] for v in owners}
    if (set(states)!=set(by_id) if restart else not set(states)<=set(by_id)):
        raise ValueError('mixed finite state inventory incomplete or foreign')
    states={eid:states.get(eid) for eid in by_id}
    owned=prepare_finite_states(model,global_q,{i:s for i,s in states.items() if by_id[i]=='qualified_q4'},num_layers,info,restart=restart)
    for eid,state in states.items():
        if by_id[eid]=='qualified_q4':continue
        element=model.mesh.elements[eid];material=model.get_material(element.material_name)
        if state is None:
            if restart:raise ValueError('mixed V2D restart state missing')
            owned[eid]=None;continue
        if not isinstance(state,Mapping):raise ValueError('mixed V2D state must be a mapping')
        q=np.asarray(global_q)[np.asarray(element.get_dof_mapping(model.mesh),int)]
        if restart:
            validated=element.validate_model_bound_nonlinear_state(model.mesh,material,state,num_layers,expected_committed_total_u=q)
            if validated['solver_kinematics']!='COROTATIONAL' or not np.array_equal(validated['committed_constitutive_u'],local_coordinates(model,element,q)):
                raise ValueError('mixed V2D global/local corotational state mismatch')
        else:
            virgin=element.init_model_bound_nonlinear_state(model.mesh,material,num_layers)
            if element.serialize_nonlinear_state(model.mesh,material,state,num_layers)!=element.serialize_nonlinear_state(model.mesh,material,virgin,num_layers):
                raise ValueError('mixed V2D historical state requires a checkpoint')
            validated=virgin
        owned[eid]=copy.deepcopy(validated)  # Preserve V2D seals/coordinate metadata.
    info['element_state_coordinates']=COORDINATES;info['finite_state_owners']=owners
    return owned

def seal_mixed_states(model,global_q,states,num_layers,info):
    from .finite_rotation_state import seal_finite_states
    owners=owner_contract(model);by_id={v['element_id']:v['owner'] for v in owners}
    if set(states)!=set(by_id):raise ValueError('mixed final state inventory incomplete')
    sealed=seal_finite_states(model,global_q,{i:s for i,s in states.items() if by_id[i]=='qualified_q4'},num_layers,info)
    for eid,state in states.items():
        if by_id[eid]=='qualified_q4':continue
        if state is None:raise ValueError('mixed final V2D state missing')
        element=model.mesh.elements[eid];q=np.asarray(global_q)[np.asarray(element.get_dof_mapping(model.mesh),int)];material=model.get_material(element.material_name)
        if not np.array_equal(state['committed_constitutive_u'],local_coordinates(model,element,q)):
            raise ValueError('mixed final V2D local constitutive state mismatch')
        sealed[eid]=element.seal_solver_integrated_nonlinear_state(model.mesh,material,state,num_layers,q,kinematics='COROTATIONAL')
        element.validate_model_bound_nonlinear_state(model.mesh,material,sealed[eid],num_layers,expected_committed_total_u=q)
    info['element_state_coordinates']=COORDINATES;info['finite_state_owners']=owners
    return sealed
