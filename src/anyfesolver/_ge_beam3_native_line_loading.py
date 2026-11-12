"""Private load scope for actual native trial assembly, not a global solver.

All reference-line work uses the preserved objective lifted-line potential.
Effective loads are owned per trial; no mutable element load or nodal lumping.
"""
from contextvars import ContextVar
from dataclasses import dataclass,field
import numpy as np
from ._ge_beam3_fibre_line_work import ReferenceLineForces,evaluate,POLICY
from ._ge_beam3_p5_seeded.core import sha


@dataclass(frozen=True)
class LinePattern:
    rows: tuple
    signature: str=field(init=False)

    def __post_init__(self):
        if type(self.rows) is not tuple:raise ValueError('explicit native line pattern tuple')
        if self.rows:ReferenceLineForces(self.rows)
        object.__setattr__(self,'signature',sha(dict(policy=POLICY,rows=self.rows)))

    def require(self,mesh):
        if type(self.rows) is not tuple:raise ValueError('native line pattern tuple authority changed')
        if self.rows:ReferenceLineForces(self.rows)
        if self.signature!=sha(dict(policy=POLICY,rows=self.rows)) or any(row[0] not in mesh.elements for row in self.rows):
            raise ValueError('native line pattern authority changed or unknown element')

    def force(self,element_id):
        for eid,*force in self.rows:
            if eid==element_id:return np.array(force,dtype=float)
        return np.zeros(3)


@dataclass(frozen=True)
class _Scope:
    mesh: object
    store: object
    pattern: LinePattern
    identities: tuple

    def require(self,element,context,view):
        if context.store is not self.store or element._mesh is not self.mesh:
            raise ValueError('foreign native line load scope')
        context.require_view(view);self.pattern.require(self.mesh)
        if self.identities!=tuple((i,e.identity) for i,e in sorted(self.mesh.elements.items())):
            raise ValueError('native line scope element identities changed')
        return self.pattern


_ACTIVE=ContextVar('ge_beam3_native_line_scope',default=None)


def current_pattern(element,context,view):
    scope=_ACTIVE.get()
    if type(scope) is not _Scope:raise ValueError('live native line load scope required')
    return scope.require(element,context,view)


def nodal_force_vector(model,pattern):
    pattern.require(model.mesh);result=np.zeros(model.mesh.dof_manager.total_dofs)
    for eid,e in sorted(model.mesh.elements.items()):
        force=pattern.force(eid)
        if np.any(force):
            op=e.operator;work=evaluate(op.reference,op.reference.coordinates,np.zeros((3,3)),np.tile(np.eye(3),(2,1,1)),force,order=op.order)
            result[list(e.get_dof_mapping(model.mesh))]+=work.gradient[:18]
    return result


def assemble_line_trial(model,displacements,store,pattern,*,tangent=True):
    from ._ge_beam3_native_line_static_element import NativeLineFibreStaticElement
    from .nonlinear_state import NonlinearStateStore
    from .nonlinear_static import _assemble_nonlinear_system
    if type(pattern) is not LinePattern or type(store) is not NonlinearStateStore or _ACTIVE.get() is not None:
        raise ValueError('exact unnested native line trial authority required')
    if not model.mesh.elements or any(type(e) is not NativeLineFibreStaticElement for e in model.mesh.elements.values()):
        raise ValueError('native line trial requires its exact standalone elements')
    if model.mesh.element_activity is not None or model.mesh.point_masses:
        raise ValueError('native line trial activity/dynamics not admitted')
    for e in model.mesh.elements.values():e._check(model.mesh)
    owned=LinePattern(pattern.rows);pattern.require(model.mesh)
    scope=_Scope(model.mesh,store,owned,tuple((i,e.identity) for i,e in sorted(model.mesh.elements.items())))
    external=nodal_force_vector(model,owned);token=_ACTIVE.set(scope)
    try:
        internal,matrix,states=_assemble_nonlinear_system(model,displacements,store,1,tangent=tangent)
        owned.require(model.mesh)
        if scope.identities!=tuple((i,e.identity) for i,e in sorted(model.mesh.elements.items())):raise ValueError('native line model changed')
        return internal,matrix,states,external
    except BaseException:
        if store.has_active_trial:store.discard_trial(store.active_trial_token())
        raise
    finally:
        _ACTIVE.reset(token)
