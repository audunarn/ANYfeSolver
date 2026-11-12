"""Exact elastic finite V2D/V2D owner composition; no state migration."""
from __future__ import annotations
import copy
from collections.abc import Mapping
import numpy as np
from .finite_mixed_state import V2D_COORDINATES,local_coordinates

PROFILE='S3_V2D_S3_V2D_COROTATIONAL_NODAL_DIRECTOR_MONOLITHIC_PATH_DEV_V0'
STATIC_PROFILE='S3_V2D_S3_V2D_COROTATIONAL_NODAL_DIRECTOR_MONOLITHIC_STATIC_DEV_V0'
COORDINATES=V2D_COORDINATES

def owner_contract(model):
    from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
    from anymaterial import IsotropicMaterial
    owners=[]
    for eid,element in sorted(model.mesh.elements.items()):
        material=model.get_material(element.material_name)
        if (type(element) is not NativeParityE4PLS3V2DShellElement
                or type(material) is not IsotropicMaterial or material.yield_stress!=0.
                or material.hardening_curve is not None or element.shell_section is not None):
            raise ValueError('finite V2D pair requires exact elastic V2D owners without sections')
        owners.append(dict(element_id=int(eid),owner='native_parity_s3_v2d',coordinates=COORDINATES))
    if not owners:raise ValueError('finite V2D owner inventory empty')
    return owners

def admitted_definition(model,definition,settings):
    if (settings.shell_skin_kinematics!='corotational_nodal_director_dev'
            or settings.general_s3_differentiation!='jet2_s3_dev'
            or settings.enforcement_solver!='monolithic_active_set'
            or settings.shell_pair_orientation!='physical_sides_dev'
            or settings.q4_source_gap_certificate!='whole_face'
            or settings.q4_source_pressure_basis!='fine_nodal_q1'
            or settings.q4_source_pressure_knot_columns is not None
            or settings.augmentation_support_resolution!='exact'
            or settings.augmentation_acceleration!='none'
            or settings.q4_s3_shared_edge_tie_step_m is not None
            or settings.path_predictor!='none' or settings.d1_interior_profile
            or definition.rigid_bodies or len(definition.surfaces)!=2
            or len(definition.interactions)!=1):return False
    interaction=definition.interactions[0];names={s.name:s for s in definition.surfaces}
    source=names.get(interaction.surface);target=names.get(interaction.target)
    if (interaction.normal_law!='augmented' or source is None or target is None or source is target
            or any(not s.element_ids or s.side not in {'top','bottom'} or s.beam_radius is not None for s in (source,target))
            or set(source.element_ids)&set(target.element_ids)
            or set(source.element_ids+target.element_ids)!=set(model.mesh.elements)):return False
    try:
        owner_contract(model)
        from .finite_rotation_predictor import finite_surface_partitions
        finite_surface_partitions(model,definition)
    except (ValueError,TypeError):return False
    return True

def v2d_state_contract(contract):
    if contract.get('profile')!=PROFILE:return False
    owners=contract.get('finite_state_owners')
    if (contract.get('element_state_coordinates')!=COORDINATES
            or not isinstance(owners,list) or not owners
            or any(not isinstance(v,dict) or set(v)!={'element_id','owner','coordinates'}
                   or type(v['element_id']) is not int
                   or v['owner']!='native_parity_s3_v2d' or v['coordinates']!=COORDINATES for v in owners)
            or [v['element_id'] for v in owners]!=sorted(set(v['element_id'] for v in owners))):
        raise ValueError('finite V2D typed coordinate inventory changed')
    # Reuse the established finite control/seed policy, not Q4 state coordinates.
    from .finite_rotation_state import finite_state_contract,PROFILE as Q4_PROFILE,COORDINATES as Q4_COORDINATES
    return finite_state_contract(dict(contract,profile=Q4_PROFILE,element_state_coordinates=Q4_COORDINATES))

def prepare_v2d_states(model,global_q,states,num_layers,info,*,restart):
    ids={v['element_id'] for v in owner_contract(model)}
    if (set(states)!=ids if restart else not set(states)<=ids):
        raise ValueError('finite V2D state inventory incomplete or foreign')
    owned={}
    for eid in sorted(ids):
        state=states.get(eid);element=model.mesh.elements[eid];material=model.get_material(element.material_name)
        q=np.asarray(global_q)[np.asarray(element.get_dof_mapping(model.mesh),int)]
        if state is None:
            if restart:raise ValueError('finite V2D restart state missing')
            owned[eid]=None;continue
        if not isinstance(state,Mapping):raise ValueError('finite V2D state must be a mapping')
        if restart:
            validated=element.validate_model_bound_nonlinear_state(model.mesh,material,state,num_layers,expected_committed_total_u=q)
            if validated['solver_kinematics']!='COROTATIONAL' or not np.array_equal(validated['committed_constitutive_u'],local_coordinates(model,element,q)):
                raise ValueError('finite V2D global/local corotational state mismatch')
        else:
            virgin=element.init_model_bound_nonlinear_state(model.mesh,material,num_layers)
            if element.serialize_nonlinear_state(model.mesh,material,state,num_layers)!=element.serialize_nonlinear_state(model.mesh,material,virgin,num_layers):
                raise ValueError('finite V2D historical state requires checkpoint')
            validated=virgin
        owned[eid]=copy.deepcopy(validated)
    info['element_state_coordinates']=COORDINATES;info['finite_state_owners']=owner_contract(model)
    return owned

def seal_v2d_states(model,global_q,states,num_layers,info):
    if set(states)!={v['element_id'] for v in owner_contract(model)}:raise ValueError('finite V2D final inventory incomplete')
    sealed={}
    for eid,state in states.items():
        if state is None:raise ValueError('finite V2D final state missing')
        element=model.mesh.elements[eid];material=model.get_material(element.material_name)
        q=np.asarray(global_q)[np.asarray(element.get_dof_mapping(model.mesh),int)]
        if not np.array_equal(state['committed_constitutive_u'],local_coordinates(model,element,q)):
            raise ValueError('finite V2D final local constitutive state mismatch')
        sealed[eid]=element.seal_solver_integrated_nonlinear_state(model.mesh,material,state,num_layers,q,kinematics='COROTATIONAL')
        element.validate_model_bound_nonlinear_state(model.mesh,material,sealed[eid],num_layers,expected_committed_total_u=q)
    info['element_state_coordinates']=COORDINATES;info['finite_state_owners']=owner_contract(model)
    return sealed
