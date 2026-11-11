"""Private spatial dead-force successor; no nodal moments or public routing."""
from dataclasses import dataclass
from math import fsum,isfinite
import numpy as np
from ._ge_beam3_retained_generalized_state import Context as DistributedContext,Program as DistributedProgram
from ._ge_beam3_retained_generalized_program import _solve
from ._ge_beam3_retained_generalized_modal import _prepare
from ._native_paired_factor_chain_modes import solve_paired_factor_chain_modes
from ._ge_beam3_native_generalized_loading import DistributedPattern

SCHEMA='GE_BEAM3_RETAINED_NODAL_DEAD_FORCE_ACCEPTED_CHAIN_V1'
POLICY='CANDIDATE_GE_BEAM3_RETAINED_NODAL_DEAD_FORCE_V1'
MODAL_POLICY='GE_BEAM3_RETAINED_NODAL_DEAD_CURRENT_REST_FACTOR_CHAIN_V1'


@dataclass(frozen=True)
class NodalDeadForces:
    rows: tuple  # Sorted unique (physical node ID, Fx, Fy, Fz), spatial frame.

    def __post_init__(self):
        if type(self.rows) is not tuple or len(self.rows)>512:raise ValueError('bounded exact nodal force tuple')
        previous=None
        for row in self.rows:
            if (type(row) is not tuple or len(row)!=4 or type(row[0]) is not int
                    or (previous is not None and row[0]<=previous)
                    or any(type(v) is not float or not isfinite(v) for v in row[1:])):
                raise ValueError('sorted unique node IDs and three finite binary64 forces required')
            previous=row[0]

    def require(self,mesh):
        self.__post_init__()
        if any(row[0] not in mesh.nodes for row in self.rows):raise ValueError('unknown nodal force node')


@dataclass(frozen=True)
class Program:
    targets: tuple
    pattern: DistributedPattern
    nodal_forces: NodalDeadForces
    max_iterations: int=24
    max_backtracks: int=8

    def __post_init__(self):
        DistributedProgram(self.targets,self.pattern,self.max_iterations,self.max_backtracks)
        if type(self.nodal_forces) is not NodalDeadForces:raise ValueError('exact nodal dead forces required')
        self.nodal_forces.__post_init__()


class Context(DistributedContext):
    program_type=Program
    schema=SCHEMA
    policy=POLICY

    def __init__(self,model,program):
        if type(program) is not Program:raise ValueError('exact nodal retained program required')
        program.__post_init__();program.nodal_forces.require(model.mesh)
        super().__init__(model,program)

    def guard(self):
        super().guard();self.program.nodal_forces.require(self.model.mesh)

    def nodal_external(self,parameter):
        self.guard();force=np.zeros(self.nodal_count)
        for node,*value in self.program.nodal_forces.rows:
            slots=self.model.mesh.dof_manager.get_node_dofs(node)
            if len(slots)!=6:raise ValueError('six-DOF nodal force map required')
            force[list(slots[:3])]=parameter*np.array(value)
        if not np.isfinite(force).all():raise ValueError('nonfinite scaled nodal force')
        return force

    def nodal_work(self,state,parameter):
        force=self.nodal_external(parameter);terms=[]
        for node in self.node_ids:
            index=self.index[node];slots=self.model.mesh.dof_manager.get_node_dofs(node)
            for axis in range(3):
                delta=fsum((float(state.positions[index,axis]),-float(self.reference_positions[index,axis]),
                    float(state.position_low[index,axis])))
                terms.append(float(force[slots[axis]])*delta)
        value=fsum(terms)
        if not isfinite(value):raise ValueError('nonfinite nodal dead-load work')
        return value

    def assemble(self,state,parameter,origins):
        r,j,_,responses,works=super().assemble(state,parameter,origins)
        # Dead forces are assembled ONCE by global node, including shared nodes.
        # Their translation-only residual has exactly zero tangent; no rotation
        # pullback, follower term or fictitious nodal couple is introduced.
        r[:self.nodal_count]-=self.nodal_external(parameter)
        scaled=r/self.scale
        metrics=(float(np.linalg.norm(scaled[self.equilibrium])),float(np.linalg.norm(scaled[self.compatibility])))
        return r,j,metrics,responses,(*works,self.nodal_work(state,parameter))


def solve(model,program,*,checkpoint=None,expected_sha256=None,stop_after=None,cancellation_token=None,progress=None):
    return _solve(Context,model,program,checkpoint=checkpoint,expected_sha256=expected_sha256,
        stop_after=stop_after,cancellation_token=cancellation_token,progress=progress)


def prepare(model,program,checkpoint,section_inertias,*,expected_sha256,cancellation_token=None):
    return _prepare(Context,MODAL_POLICY,model,program,checkpoint,section_inertias,
        expected_sha256=expected_sha256,cancellation_token=cancellation_token)


def solve_modes(model,program,checkpoint,section_inertias,*,expected_sha256,bounds,
                num_modes=6,root_width=1e-10,relative_width=1e-12,cancellation_token=None):
    packet,guard=prepare(model,program,checkpoint,section_inertias,expected_sha256=expected_sha256,
        cancellation_token=cancellation_token)
    result=solve_paired_factor_chain_modes(packet.left,packet.right,packet.geometric,packet.kinetic,
        packet.free_dofs,packet.algebraic_dofs,bounds=bounds,num_modes=num_modes,
        root_width=root_width,relative_width=relative_width,cancellation_token=cancellation_token)
    guard();return packet,result
