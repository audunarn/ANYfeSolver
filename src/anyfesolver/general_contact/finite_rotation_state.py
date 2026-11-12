"""Exact finite-contact interpretation of Qualified Q4 constitutive records.

The outer checkpoint retains global additive coordinates. Element records are
sealed at the corotational owner's reference deformational coordinates, not at
those global coordinates. This adapter admits no deletion or material migration.
"""
from __future__ import annotations
import copy
from collections.abc import Mapping
import numpy as np

PROFILE = "Q4_Q4_COROTATIONAL_NODAL_DIRECTOR_MONOLITHIC_PATH_DEV_V0"
COORDINATES = "qualified_q4_corotational_reference_deformational_u_d_v1"


def finite_state_contract(contract):
    """Recognize only the named profile, requiring its full coordinate contract."""
    from .finite_mixed_state import PROFILE as MIXED_PROFILE,mixed_state_contract
    if contract.get('profile')==MIXED_PROFILE:
        return mixed_state_contract(contract)
    from .finite_v2d_state import PROFILE as V2D_PROFILE,v2d_state_contract
    if contract.get('profile')==V2D_PROFILE:
        return v2d_state_contract(contract)
    if contract.get("profile") != PROFILE:
        if (contract.get("shell_skin_kinematics") == "corotational_nodal_director_dev"
                or contract.get("element_state_coordinates") == COORDINATES):
            raise ValueError("finite shell state coordinates require the exact named profile")
        return False
    expected = dict(control="contact_path", shell_skin_kinematics="corotational_nodal_director_dev",
                    structural_kinematics="corotational", corotational_tangent="consistent",
                    element_state_coordinates=COORDINATES,
                    enforcement_solver="monolithic_active_set")
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError("finite shell checkpoint state coordinate/profile contract is incompatible")
    method=contract.get('finite_rigid_seed')
    if method=='common_reference_frame_left_transport_project_controls_v1':
        if 'finite_frame_seed' in contract or 'finite_surface_frames' in contract:
            raise ValueError('common finite frame contract has incompatible selector metadata')
    elif method=='per_surface_left_transport_project_controls_v1':
        frames=contract.get('finite_surface_frames')
        if (contract.get('finite_frame_seed')!='per_surface_reference_frames_dev'
                or not isinstance(frames,list) or len(frames)!=2
                or any(not isinstance(p,dict) or set(p)!={'name','element_ids','node_ids','anchor_node_id'}
                       or p['anchor_node_id'] not in p['node_ids'] for p in frames)):
            raise ValueError('per-surface finite state requires explicitly bound partitions/anchors')
    else:
        raise ValueError('finite shell state seed method is incompatible')
    return True


def owner_local_coordinates(model, element, global_element_q):
    from anyfesolver.corotational import _corotational_cache, _corotational_deformation_state
    from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
    from anymaterial import IsotropicMaterial
    material = model.get_material(element.material_name)
    if (type(element) is not QualifiedE4PLShellElement or type(material) is not IsotropicMaterial
            or material.yield_stress != 0. or material.hardening_curve is not None):
        raise ValueError("finite shell state adapter requires exact elastic Qualified Q4")
    reference = _corotational_cache(model, centered_shell=False).get(int(element.element_id))
    if reference is None or reference.category != "shell":
        raise ValueError("finite shell state lacks its exact owner shell reference")
    local, _, _ = _corotational_deformation_state(reference, element, global_element_q)
    return local


def prepare_finite_states(model, global_q, states, num_layers, info, *, restart):
    """Validate all supplied local-bound records before returning trial copies."""
    owned = {}
    for eid, state in states.items():
        if state is None:
            if restart:
                raise ValueError("finite shell restart lacks a bound constitutive record")
            owned[eid] = None
            continue
        if not isinstance(state, Mapping) or "qualified_q4_activity_disposition" in state:
            raise ValueError("finite shell state is not an active constitutive record")
        element = model.mesh.elements[eid]
        q = np.asarray(global_q)[np.asarray(element.get_dof_mapping(model.mesh), dtype=int)]
        local = owner_local_coordinates(model, element, q)
        if not restart:
            raise ValueError("finite shell initial historical states require a bound checkpoint")
        element.validate_committed_current_tangent_state(
            model.mesh, model.get_material(element.material_name), local, state, num_layers)
        trial = copy.deepcopy(dict(state))
        for key in ("qualified_q4_committed_binding", "state_integrity_sha256", "state_digest"):
            trial.pop(key, None)
        owned[eid] = trial
    info["element_state_coordinates"] = COORDINATES
    return owned


def seal_finite_states(model, global_q, states, num_layers, info):
    sealed = {}
    for eid, state in states.items():
        if state is None or "qualified_q4_activity_disposition" in state:
            raise ValueError("finite shell final state lacks an active constitutive record")
        element = model.mesh.elements[eid]
        q = np.asarray(global_q)[np.asarray(element.get_dof_mapping(model.mesh), dtype=int)]
        local = owner_local_coordinates(model, element, q)
        material = model.get_material(element.material_name)
        sealed[eid] = element.seal_committed_current_tangent_state(
            model.mesh, material, local, state, num_layers)
        element.validate_committed_current_tangent_state(
            model.mesh, material, local, sealed[eid], num_layers)
    info["element_state_coordinates"] = COORDINATES
    return sealed
