"""Explicit immutable multibody selections for one native/S3 midpoint owner."""
from dataclasses import dataclass
from math import nextafter, inf, sqrt
import json
from anyfesolver._ge_beam3_connected_dynamic import MAX_COORDINATES
import numpy as np
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver.fe_core import FEModel
from anyfesolver._ge_beam3_native_multi_shell_contact import NativeMultiShellContactAnalysis
from .native_retained_line_s3 import NativeRetainedRoundLineS3Interior
from .native_retained_line_s3_work import NativeRetainedLineS3Work
from .s3_skin_work import S3SkinWorkAdapter
from .overlap import reference_shell_face
SELECTOR='ge_beam3_native_multi_s3_small_motion_transient_dev'

@dataclass(frozen=True)
class NativeContactBody:
    body_id: int
    definitions: tuple[NativeBeamDefinition, ...]
@dataclass(frozen=True)
class S3ContactBody:
    body_id: int
    model: FEModel
@dataclass(frozen=True, order=True)
class NativeTransientS3ContactPair:
    native_body_id: int
    native_element_id: int
    shell_body_id: int
    shell_element_id: int


def _selections(native_bodies,shell_bodies,pairs):
    for values,cls in ((native_bodies,NativeContactBody),(shell_bodies,S3ContactBody),(pairs,NativeTransientS3ContactPair)):
        if type(values) is not tuple or not values or any(type(v) is not cls for v in values):
            raise ValueError('nonempty immutable exact body/pair tuples required')
    ids=[b.body_id for b in native_bodies+shell_bodies]
    if any(type(i) is not int or i<=0 for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('globally unique positive body identities required')
    if any(type(b.model) is not FEModel for b in shell_bodies) or len({id(b.model) for b in shell_bodies})!=len(shell_bodies):
        raise ValueError('distinct exact shell model objects required; no body alias')
    if any(type(b.definitions) is not tuple or not b.definitions or any(type(d) is not NativeBeamDefinition for d in b.definitions) for b in native_bodies):
        raise ValueError('complete immutable native body definitions required')
    if any(any(type(i) is not int or i<=0 for i in (p.native_body_id,p.native_element_id,p.shell_body_id,p.shell_element_id)) for p in pairs):
        raise ValueError('positive exact pair identities required')
    if len(set(pairs))!=len(pairs):raise ValueError('duplicate contact pair selection')
    data=[json.loads(d.raw) for b in native_bodies for d in b.definitions]
    predicted=6*len({n for d in data for n in d['node_ids']})+24*len(data)+sum(b.model.mesh.dof_manager.total_dofs for b in shell_bodies)
    if predicted>MAX_COORDINATES:raise MemoryError('global retained coordinate budget exceeded before preparation')
    native=tuple(NativeContactBody(b.body_id,tuple(sorted(b.definitions,key=lambda d:json.loads(d.raw)['element_id'])))
                 for b in sorted(native_bodies,key=lambda b:b.body_id));shell=tuple(sorted(shell_bodies,key=lambda b:b.body_id))
    pairs=tuple(sorted(pairs));elements={};all_ids=set();used_nodes=set()
    for body in native:
        nodes=set()
        for definition in body.definitions:
            element,_=definition.instantiate()
            if element.element_id in all_ids:raise ValueError('native element identities must be globally unique')
            all_ids.add(element.element_id);nodes.update(element.node_ids);elements[(body.body_id,element.element_id)]=element
        if nodes.intersection(used_nodes):raise ValueError('native node alias between explicit bodies')
        used_nodes.update(nodes)
    targets={b.body_id:b.model for b in shell}
    for p in pairs:
        if (p.native_body_id,p.native_element_id) not in elements or p.shell_body_id not in targets or p.shell_element_id not in targets[p.shell_body_id].mesh.elements:
            raise ValueError('selected contact pair does not bind body/element inventory')
    return native,shell,pairs,elements,targets


def create_native_transient_multi_shell_analysis(formulation,native_bodies,shell_bodies,pairs,*,
        radius_m,line_stiffness_n_per_m2,patch_boundary_clearance_m,geometry_tolerance_m,
        energy_tolerance_j,force_tolerance_n,tangent_tolerance_n_per_m,maximum_geometry_evaluations,
        maximum_integration_intervals,maximum_penetration_m,maximum_swept_intervals,
        maximum_shell_displacement_m,maximum_shell_rotation_rad,inertia_order=24,max_iterations=12,max_backtracks=8):
    if formulation!=SELECTOR:raise ValueError('exact multibody native/S3 development selector required')
    native,shell,pairs,elements,targets=_selections(native_bodies,shell_bodies,pairs)
    if any(type(x) not in (int,float) or not np.isfinite(x) or x<=0 for x in
           (energy_tolerance_j,force_tolerance_n,tangent_tolerance_n_per_m)):
        raise ValueError('positive finite global dimensional error limits required')
    length=min(float(np.linalg.norm(e.operator.reference.coordinates[-1]-e.operator.reference.coordinates[0])) for e in elements.values())
    if not np.isfinite(length) or length<=0:raise ValueError('positive frozen native length scale required')
    works=[];count=len(pairs)
    for pair in pairs:
        element=elements[(pair.native_body_id,pair.native_element_id)];model=targets[pair.shell_body_id]
        skin=S3SkinWorkAdapter.from_model(model,pair.shell_body_id,pair.shell_element_id,'top')
        # Kernel errors are max component/entry in27 physical sample metres.
        # sqrt27 converts gradient;27 converts Hessian to Euclidean/operator norms.
        factor=nextafter(sqrt(27.)*max(4.,1.+2.*abs(skin.physical_offset)/length),inf)
        kernel=NativeRetainedRoundLineS3Interior(element.element_id,element.operator.reference,element.node_ids,
            pair.shell_element_id,reference_shell_face(model,pair.shell_body_id,pair.shell_element_id,'top'),skin.node_ids,
            radius_m=radius_m,line_stiffness_n_per_m2=line_stiffness_n_per_m2,
            rim_clearance_m=patch_boundary_clearance_m,geometry_tolerance_m=geometry_tolerance_m,
            energy_tolerance_j=nextafter(energy_tolerance_j/count,0.),
            force_tolerance_n=nextafter(force_tolerance_n/(count*factor),0.),
            tangent_tolerance_n_per_m=nextafter(tangent_tolerance_n_per_m/(count*factor*factor),0.),
            maximum_integration_intervals=maximum_integration_intervals)
        works.append(NativeRetainedLineS3Work(kernel,skin))
    return NativeMultiShellContactAnalysis(native,shell,pairs,tuple(works),maximum_penetration_m=maximum_penetration_m,
        maximum_geometry_evaluations=maximum_geometry_evaluations,maximum_integration_intervals=maximum_integration_intervals,
        maximum_swept_intervals=maximum_swept_intervals,maximum_shell_displacement_m=maximum_shell_displacement_m,
        maximum_shell_rotation_rad=maximum_shell_rotation_rad,energy_tolerance_j=energy_tolerance_j,
        force_tolerance_n=force_tolerance_n,tangent_tolerance_n_per_m=tangent_tolerance_n_per_m,
        inertia_order=inertia_order,max_iterations=max_iterations,max_backtracks=max_backtracks)


def solve_native_transient_multi_shell(analysis,config,*,checkpoint=None,expected_sha256=None,
        max_steps=10000,max_cutbacks=8,minimum_dt=None,cancellation_token=None):
    if type(analysis) is not NativeMultiShellContactAnalysis:raise TypeError('exact global native/S3 development owner required')
    from anyfesolver._ge_beam3_connected_trajectory import _solve_retained_transient
    return _solve_retained_transient(analysis,config,checkpoint=checkpoint,expected_sha256=expected_sha256,
        max_steps=max_steps,max_cutbacks=max_cutbacks,minimum_dt=minimum_dt,cancellation_token=cancellation_token)
