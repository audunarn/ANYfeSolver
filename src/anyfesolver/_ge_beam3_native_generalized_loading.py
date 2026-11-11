"""Owned line-force/distributed-couple patterns for private native trials."""
from contextvars import ContextVar
from dataclasses import dataclass,field
from math import isfinite
import numpy as np
from ._ge_beam3_native_line_loading import LinePattern,nodal_force_vector
from ._ge_beam3_generalized_static_boundary import POLICY
from ._ge_beam3_p5_seeded.core import sha


def _rows(rows):
    if type(rows) is not tuple:raise ValueError('distributed couple tuple required')
    ids=[]
    for row in rows:
        if (type(row) is not tuple or len(row)!=4 or type(row[0]) is not int or row[0]<=0
                or any(type(v) is not float or not isfinite(v) for v in row[1:]) or not any(row[1:])):
            raise ValueError('positive element and nonzero finite distributed couple row required')
        ids.append(row[0])
    if ids!=sorted(set(ids)) or len(ids)>64:raise ValueError('bounded ordered distributed couple identities')


@dataclass(frozen=True)
class DistributedPattern:
    line: LinePattern
    couples: tuple
    signature: str=field(init=False)

    def __post_init__(self):
        if type(self.line) is not LinePattern:raise ValueError('exact distributed line pattern')
        _rows(self.couples)
        object.__setattr__(self,'signature',sha(dict(policy=POLICY,line=self.line,couples=self.couples)))

    def require(self,mesh):
        if type(self.line) is not LinePattern:raise ValueError('exact distributed line pattern')
        self.line.require(mesh);_rows(self.couples)
        if any(row[0] not in mesh.elements for row in self.couples) or self.signature!=sha(dict(policy=POLICY,line=self.line,couples=self.couples)):
            raise ValueError('distributed load authority changed or absent element')

    def force(self,eid):return self.line.force(eid)

    def density(self,eid):
        for node,*value in self.couples:
            if node==eid:return np.array(value,dtype=float)
        return np.zeros(3)


@dataclass(frozen=True)
class _Scope:
    mesh: object
    store: object
    pattern: DistributedPattern
    identities: tuple

    def require(self,element,context,view):
        if context.store is not self.store or element._mesh is not self.mesh:raise ValueError('foreign distributed load scope')
        context.require_view(view);self.pattern.require(self.mesh)
        if self.identities!=tuple((i,e.identity) for i,e in sorted(self.mesh.elements.items())):
            raise ValueError('distributed scope element identities changed')
        return self.pattern


_ACTIVE=ContextVar('ge_beam3_native_generalized_scope',default=None)


def current_pattern(element,context,view):
    scope=_ACTIVE.get()
    if type(scope) is not _Scope:raise ValueError('live distributed load scope required')
    return scope.require(element,context,view)


def assemble_distributed_trial(model,displacements,store,pattern,*,tangent=True):
    from ._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
    from .nonlinear_state import NonlinearStateStore
    from .nonlinear_static import _assemble_nonlinear_system
    if type(pattern) is not DistributedPattern or type(store) is not NonlinearStateStore or _ACTIVE.get() is not None:
        raise ValueError('exact unnested distributed trial authority required')
    if not model.mesh.elements or any(type(e) is not NativeGeneralizedStaticElement for e in model.mesh.elements.values()):
        raise ValueError('exact distributed standalone elements required')
    if model.mesh.element_activity is not None or model.mesh.point_masses:raise ValueError('distributed trial activity/dynamics not admitted')
    for e in model.mesh.elements.values():e._check(model.mesh)
    pattern.require(model.mesh);owned=DistributedPattern(LinePattern(pattern.line.rows),pattern.couples)
    scope=_Scope(model.mesh,store,owned,tuple((i,e.identity) for i,e in sorted(model.mesh.elements.items())))
    external=nodal_force_vector(model,owned.line);token=_ACTIVE.set(scope)
    try:
        internal,matrix,states=_assemble_nonlinear_system(model,displacements,store,1,tangent=tangent)
        owned.require(model.mesh);pattern.require(model.mesh)
        if scope.identities!=tuple((i,e.identity) for i,e in sorted(model.mesh.elements.items())):raise ValueError('distributed model changed')
        return internal,matrix,states,external
    except BaseException:
        if store.has_active_trial:store.discard_trial(store.active_trial_token())
        raise
    finally:_ACTIVE.reset(token)
