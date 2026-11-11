"""One-use, solve-local completed equation value; never an accepted state.

Only certified exact elastic element routes avoid the duplicate assembly.
The original trial has already closed/discarded its constitutive candidate.
No lease is renewed at consumption and no arrays are recomputed there.
"""
from dataclasses import dataclass
import numpy as np

from anyfesolver.e4_pl_enriched import EnrichedProfileError
from .enriched_static import check_cancel
from anyfesolver.matrix_assembly import AssemblyError
from anyfesolver.element_capabilities import ElementCapabilityError


def _callables(owner):
    result=[]
    for name in ('equations','structural_response','linearize','metrics','_mixed_direction'):
        bound=getattr(owner,name)
        function=getattr(bound,'__func__',bound)
        result.append((id(getattr(bound,'__self__',None)),function,getattr(function,'__code__',None)))
    for bound in (owner._nodal.verify,owner._nodal.evaluate,*(c.response for c in owner._nonlinear_cells)):
        function=getattr(bound,'__func__',bound)
        result.append((id(getattr(bound,'__self__',None)),function,getattr(function,'__code__',None)))
    from . import enriched_multitarget_moments,enriched_moment_batch
    for function in (enriched_multitarget_moments.selected_batch_moments,enriched_moment_batch.batch_moments):
        result.append((None,function,getattr(function,'__code__',None)))
    return tuple(result)


def _arrays(owner):
    return tuple((a.dtype.str,a.shape,a.tobytes()) for a in
        (owner.offset,owner.transform,owner.scales,owner.reference_support,
         *(a for p in owner.pairs for a in (p.reference_corners,p.reference_directors,p.reference_source))))


class _EmptyHistoryIdentity:
    """Strong ownership prevents address recycling; comparison is identity only."""
    def __init__(self,state):self.state=state
    def __eq__(self,other):
        return type(other) is _EmptyHistoryIdentity and self.state is other.state


@dataclass(frozen=True)
class _Admission:
    value: object
    status: str
    reason: str
    qualifier_error_type: object = None
    qualifier_error_argument_prefix: object = None
    qualifier_cause_type: object = None
    qualifier_cause_argument_prefix: object = None


def _decline(reason):return _Admission(None,'DECLINED',reason)


def _qualifier_decline(reason,error):
    # Existing error data only; no arbitrary str/repr/provider callbacks.
    def argument(exception):
        args=BaseException.args.__get__(exception)
        return args[0][:512] if type(args) is tuple and args and type(args[0]) is str else None
    cause=BaseException.__cause__.__get__(error)
    return _Admission(None,'DECLINED',reason,type(error).__name__,argument(error),
        None if cause is None else type(cause).__name__,None if cause is None else argument(cause))


def _history(owner):
    from anyfesolver.nonlinear_state import NonlinearStateStore
    state=owner._nodal.states
    if type(state) is dict:
        if dict.__len__(state)==0:
            return (_EmptyHistoryIdentity(state),'VIRGIN_ELASTIC_LEGACY_EMPTY',False,())
        return None
    if type(state) is not NonlinearStateStore:return None
    return (id(state),state.generation,state.has_active_trial,
            tuple((key,id(batch),batch.generation,batch.has_active_trial) for key,batch in state._batches.items()))


def _pure(owner, dependencies):
    from .enriched_multitarget import MultiTargetSystem,_REUSE_OPERATOR_FUNCTIONS,_Baseline,_Cell
    if (type(owner) is not MultiTargetSystem or type(owner._nodal) is not _Baseline
            or any(type(c) is not _Cell for c in owner._nonlinear_cells)): return False
    observed=dependencies[0]
    functions=observed[:7]+observed[7:8]+observed[-2:]
    return (len(functions)==10 and all((x[1],x[2])==_REUSE_OPERATOR_FUNCTIONS[7] for x in observed[7:-2])
            and all((x[1],x[2])==expected
        for x,expected in zip(functions,_REUSE_OPERATOR_FUNCTIONS)))


class AcceptedTrial:
    @staticmethod
    def dependencies(owner, reduced, z, load):
        inputs=tuple((a.dtype.str,a.shape,a.tobytes()) for a in (reduced,z,load))
        return _callables(owner), _arrays(owner), _history(owner), inputs, tuple(sorted(np.geterr().items()))

    @classmethod
    def capture(cls, owner, reduced, z, load, criteria, control, result, cancel, dependencies):
        try:_admission=cls._capture(owner,reduced,z,load,criteria,control,result,cancel,dependencies)
        except (AssemblyError,ElementCapabilityError) as error:
            _admission=_qualifier_decline('QUALIFIED_CAPTURE_ERROR',error)
        return _admission.value

    @classmethod
    def _capture(cls, owner, reduced, z, load, criteria, control, result, cancel, dependencies):
        from anyfesolver.matrix_assembly import _CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE
        from anyfesolver.current_state_tangent import _capture_qualified_v2d_trusted_authority
        from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
        from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
        from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement

        if not _pure(owner,dependencies):return _decline('UNSUPPORTED_OPERATOR_GRAPH')
        if cls.dependencies(owner,reduced,z,load) != dependencies:return _decline('PRETRIAL_DEPENDENCIES_CHANGED')
        if dependencies[2] is None:return _decline('UNSUPPORTED_HISTORY')
        if dependencies[2][2]:return _decline('ACTIVE_HISTORY_TRIAL')
        owner._nodal.verify()
        items = tuple(owner._nodal.model.mesh.elements.items())
        allowed = (QualifiedE4PLShellElement, QualifiedE4PLS3ShellElement,
                   NativeParityE4PLS3V2DShellElement)
        if any(type(element) not in allowed for _, element in items):
            return _decline('UNSUPPORTED_ELEMENT_ROUTE')  # Complete evaluation remains the fallback.
        v2d = _capture_qualified_v2d_trusted_authority(items, context='enriched accepted trial')
        if any(type(e) is NativeParityE4PLS3V2DShellElement for _, e in items) and v2d is None:
            return _decline('V2D_AUTHORITY_UNAVAILABLE')
        lease = _CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE(
            owner._nodal.model, context='enriched accepted trial', allow_q4_cached_stiffness=True)
        lease(owner._nodal.model, context='enriched accepted trial capture', final=True)
        if v2d is not None:
            v2d(context='enriched accepted trial capture')
        check_cancel(cancel)
        owner._nodal.verify()
        if cls.dependencies(owner,reduced,z,load) != dependencies:
            return _decline('POST_CALLBACK_DEPENDENCIES_CHANGED')
        lease(owner._nodal.model, context='enriched accepted trial post-callback', final=True)
        if v2d is not None:
            v2d(context='enriched accepted trial post-callback')
        made = cls()
        made.owner, made.identity = owner, owner.identity
        made.model, made.nodal = owner._nodal.model, owner._nodal
        made.layout = (owner.offset, owner.transform, owner.pairs, owner.quadrature)
        made.callables, made.arrays, made.history, made.input_bytes, made.error_modes = dependencies
        made.caches = tuple(getattr(owner._nodal.model.mesh,n,None) for n in ('_sparsity_cache','_topology_signature_cache'))
        made.inputs = tuple(np.array(a, copy=True) for a in (reduced, z, load))
        made.criteria, made.control = tuple(vars(criteria).items()), tuple(vars(control).items())
        made.lease, made.v2d, made.result, made.used = lease, v2d, result, False
        return _Admission(made,'CAPTURED','EXACT_ELIGIBLE_TRIAL')

    def consume(self, owner, reduced, z, load, criteria, control, cancel):
        try:_admission=self._consume(owner,reduced,z,load,criteria,control,cancel)
        except (AssemblyError,ElementCapabilityError) as error:
            self.used=True
            _admission=_qualifier_decline('QUALIFIED_CONSUMPTION_ERROR',error)
        return _admission.value

    def _consume(self, owner, reduced, z, load, criteria, control, cancel):
        from .runtime import active_contact_manager

        check_cancel(cancel)
        owner._nodal.verify()
        if active_contact_manager() is not None:
            raise EnrichedProfileError('nested contact manager would double-count physical contact')
        if (self.used or owner is not self.owner or owner.identity != self.identity
                or owner._nodal.model is not self.model or owner._nodal is not self.nodal
                or any(a is not b for a, b in zip(self.layout,
                    (owner.offset, owner.transform, owner.pairs, owner.quadrature)))
                or tuple(vars(criteria).items()) != self.criteria
                or tuple(vars(control).items()) != self.control
                or any(a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes()
                       for a, b in zip(self.inputs, (reduced, z, load)))):
            raise EnrichedProfileError('accepted trial consumption identity changed')
        if (_callables(owner) != self.callables or _arrays(owner) != self.arrays
                or _history(owner) != self.history or tuple(sorted(np.geterr().items())) != self.error_modes):
            self.used = True
            return _decline('CONSUMPTION_DEPENDENCIES_CHANGED')  # Complete evaluation on changed dependencies.
        if any(getattr(owner._nodal.model.mesh,n,None) is not old for n,old in
                zip(('_sparsity_cache','_topology_signature_cache'),self.caches)):
            self.used=True
            return _decline('CACHE_CONTAINER_CHANGED')  # Original guards decide compatibility.
        self.lease(owner._nodal.model, context='enriched accepted trial consumption', final=True)
        if self.v2d is not None:
            self.v2d(context='enriched accepted trial consumption')
        self.used = True
        return _Admission(self.result,'CONSUMED','EXACT_ELIGIBLE_TRIAL')
