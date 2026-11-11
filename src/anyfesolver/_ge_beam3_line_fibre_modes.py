"""Private at-rest spectra of accepted conservative reference-line-load states.

Retain the external potential Hessian -lambda W'' of the lifted physical line.
The accepted load and plastic coordinates are fixed during elastic perturbation.
No finite-velocity dynamics, follower moments, nonlinear continuation, checkpoint
conversion or buckling load-factor qualification is provided by this module.
"""
from dataclasses import dataclass
from hashlib import sha256
from time import monotonic
import json
import numpy as np
from scipy.linalg import block_diag

from ._ge_beam3_fibre_line_program import Context
from ._ge_beam3_fibre_line_work import ReferenceLineForces, evaluate as line_work
from ._ge_beam3_retained_fibre_modes import (
    FROZEN, ALGORITHMIC, Packet as FibrePacket, Analysis as FibreAnalysis, compliance_factor,
)
from ._ge_beam3_lifted_kinetic_factor import current_lifted_kinetic_factor
from ._native_exact_shift_chain_modes import solve_factor_chain_modes
from ._native_reference_modal import _owned
from ._ge_beam3_p5_seeded.core import sha
from .control import cancellation_safe_point


SCHEMA='GE_BEAM3_CONSERVATIVE_LINE_FIBRE_FIXED_LOAD_SPECTRA_V1'


@dataclass(frozen=True)
class Packet(FibrePacket):
    external_potential_hessian: np.ndarray | None = None


def prepare(model,program,checkpoint,section_inertias,*,line_forces,material_policy,
            expected_checkpoint_sha256=None,cancellation_token=None,coordinate_limit=80,max_coordinates=256):
    if type(line_forces) is not ReferenceLineForces: raise ValueError('explicit conservative reference line-force pattern')
    if type(coordinate_limit) is not int or coordinate_limit not in (80,128,256):
        raise ValueError('explicit admitted spectral coordinate limit')
    if type(max_coordinates) is not int or max_coordinates not in (256,512):
        raise ValueError('explicit admitted retained coordinate limit')
    if material_policy not in (FROZEN,ALGORITHMIC): raise ValueError('explicit spectral material policy')
    started=monotonic()
    def local_check():
        cancellation_safe_point(cancellation_token,'line-fibre-modes.construct')
        if monotonic()-started>120.: raise TimeoutError('line-fibre spectral construction deadline')
    local_check()
    if type(checkpoint) is not bytes: raise ValueError('immutable canonical line-force checkpoint')
    context=Context(model,program,line_forces=line_forces,check=local_check,max_coordinates=max_coordinates)
    state,_=context.restore(checkpoint,expected_sha256=expected_checkpoint_sha256)
    parameter=0. if state.completed_targets==0 else program.targets[state.completed_targets-1]
    def guard():
        local_check(); context.guard()
    layout=context.layout; n=layout.nodal_count+6*len(context.probes)
    if n>coordinate_limit: raise ValueError('bounded standalone line-load spectrum required')
    if (type(section_inertias) is not dict or any(type(i) is not int for i in section_inertias)
            or set(section_inertias)!={i for i,_ in layout.elements}):
        raise ValueError('one explicit section inertia per element')
    inertias={i:_owned(v) for i,v in section_inertias.items()}
    forces={row[0]:row[1:] for row in line_forces.rows}
    g=np.zeros((n,n)); external=np.zeros((n,n)); lefts=[]; rights=[]; kinetics=[]
    mechanical=state.mechanical
    for index,((eid,element),probe) in enumerate(zip(layout.elements,context.probes)):
        guard(); nodes=layout.nodes[index]
        response=probe.evaluate(mechanical.positions[nodes],mechanical.position_low[nodes],
            mechanical.nodal_frames[nodes],mechanical.cell_rotations[index],mechanical.resultants[index],
            origin=state.origins[index],check=local_check)
        left=compliance_factor(probe,json.loads(response.material),state.origins[index],state.histories[index],
            mechanical.resultants[index],policy=material_policy,check=local_check)
        slots=list(element.get_dof_mapping(model.mesh))+list(range(layout.nodal_count+6*index,layout.nodal_count+6*index+6))
        j=response.hessian[24:,:24]; right=np.zeros((36,n)); right[:,slots]=np.vstack((j,j))
        g[np.ix_(slots,slots)]+=response.hessian[:24,:24]
        if eid in forces:
            work=line_work(probe.reference,mechanical.positions[nodes],mechanical.position_low[nodes],
                mechanical.cell_rotations[index],forces[eid],order=probe.order)
            external[np.ix_(slots,slots)]-=parameter*work.hessian[:24,:24]
        local=current_lifted_kinetic_factor(probe.reference,inertias[eid],mechanical.cell_rotations[index],order=probe.order)
        kinetic=np.zeros((len(local),n)); kinetic[:,slots]=local
        lefts.append(left); rights.append(right); kinetics.append(kinetic)
    g+=external
    nodal_free=tuple(i for i in range(layout.nodal_count) if i not in layout.fixed)
    free=nodal_free+tuple(range(layout.nodal_count,n)); algebraic=tuple(i for i in nodal_free if i%6>=3)
    left,right,kinetic=block_diag(*lefts),np.vstack(rights),np.vstack(kinetics)
    if np.any(kinetic[:,algebraic]): raise ValueError('nodal rotation traces must have zero inertia')
    if not np.isfinite(g).all() or not np.isfinite(external).all(): raise ValueError('finite total potential Hessian')
    identity=sha(dict(schema=SCHEMA,equilibrium_load_parameter=parameter,line_forces=line_forces.descriptor(),
        perturbation_load_policy='FIXED_ACCEPTED_REFERENCE_ARCLENGTH_SPATIAL_DEAD_FORCE',
        checkpoint_sha256=sha256(checkpoint).hexdigest(),material_policy=material_policy,
        operators=[p.identity for p in context.probes],left=left,right=right,geometric=g,
        external_potential_hessian=external,kinetic=kinetic,free=free,algebraic=algebraic,inertias=inertias,
        spectral_coordinate_limit=coordinate_limit,retained_coordinate_limit=max_coordinates))
    guard()
    return Packet(_owned(left),_owned(right),_owned(g),_owned(kinetic),free,algebraic,identity,material_policy,
        external_potential_hessian=_owned(external)),guard,parameter


@dataclass(frozen=True)
class Analysis(FibreAnalysis):
    equilibrium_load_parameter: float = 0.
    perturbation_load_policy: str = 'FIXED_ACCEPTED_REFERENCE_ARCLENGTH_SPATIAL_DEAD_FORCE'
    external_potential_hessian_included: bool = True
    checkpoint_converted: bool = False


def solve_modes(model,program,checkpoint,section_inertias,*,line_forces,material_policy,bounds,
                num_modes=6,root_width=1e-10,expected_checkpoint_sha256=None,cancellation_token=None,
                coordinate_limit=80,exact_dimension_limit=64,max_coordinates=256):
    if type(exact_dimension_limit) is not int or exact_dimension_limit not in (64,96,160):
        raise ValueError('explicit admitted exact-inertia dimension limit')
    packet,guard,parameter=prepare(model,program,checkpoint,section_inertias,line_forces=line_forces,
        material_policy=material_policy,expected_checkpoint_sha256=expected_checkpoint_sha256,
        cancellation_token=cancellation_token,coordinate_limit=coordinate_limit,max_coordinates=max_coordinates)
    modes=solve_factor_chain_modes(packet.left,packet.right,packet.geometric,packet.kinetic,
        packet.free,packet.algebraic,bounds=bounds,num_modes=num_modes,root_width=root_width,
        cancellation_token=cancellation_token,exact_dimension_limit=exact_dimension_limit)
    guard()
    interpretation=('ELASTIC_PERTURBATIONS_WITH_PLASTIC_COORDINATES_FIXED' if material_policy==FROZEN else
        'LAST_INCREMENT_OPERATOR_ONLY_NOT_PHYSICAL_VIBRATION_AUTHORITY')
    return packet,Analysis(material_policy,interpretation,modes.eigenvalues,modes.full_modes,modes.numerical_brackets,
        modes.spectral_residual,modes.original_ritz_residual,modes.full_backward_residual,packet.identity,
        sha256(checkpoint).hexdigest(),equilibrium_load_parameter=parameter)
