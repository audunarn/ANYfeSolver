"""Private reference-linear mixed snapshot owner; no nonlinear/public admission."""
from typing import NamedTuple
from hashlib import sha256
import json
import inspect
import threading
import weakref

import numpy as np
from scipy.linalg import cho_solve

from ._ge_beam3_g1_analysis import runtime_digest
from ._ge_beam3_g1_elastic import canonical, owned, sha
from ._ge_beam3_g3b_reference import (
    B2TranslationReferenceProblem, B3TranslationReferenceProblem, invariant, progress)
from ._ge_beam3_g3b_q4_reference import Q4TranslationReferenceProblem
from ._ge_beam3_g3b_s3_reference import S3TranslationReferenceProblem
from ._ge_beam3_g3b_weighted_reference import WeightedQ4TranslationReferenceProblem
from ._ge_beam3_g3b_result_schema import validate_result

SCHEMA = "GE_BEAM3_G3B_REFERENCE_OWNER_RESTART_V1"
POLICY = "REFERENCE_LINEAR_SNAPSHOT_ONLY_NO_FINITE_POSE_OR_MATERIAL_HISTORY"
MAX_BYTES = 8*1024*1024
MAX_ENTRIES = 128
_TYPES = (B2TranslationReferenceProblem, B3TranslationReferenceProblem,
          Q4TranslationReferenceProblem, S3TranslationReferenceProblem,
          WeightedQ4TranslationReferenceProblem)
_CLAIMS = weakref.WeakKeyDictionary()
_PUBLICATIONS = weakref.WeakKeyDictionary()
_CLAIM_LOCK = threading.Lock()


class _Bundle(NamedTuple):
    data: bytes
    digest: str


def _parse(data):
    if type(data) is not bytes or not 0 < len(data) <= MAX_BYTES:
        raise ValueError("bounded exact checkpoint bytes required")
    def pairs(items):
        out = {}
        for key,value in items:
            if key in out: raise ValueError("duplicate checkpoint key")
            out[key] = value
        return out
    def bad(value): raise ValueError("nonfinite checkpoint")
    try:
        body = json.loads(data,object_pairs_hook=pairs,parse_constant=bad)
        if canonical(body) != data: raise ValueError("noncanonical checkpoint")
    except (UnicodeError, RecursionError, TypeError, OverflowError) as exc:
        raise ValueError("invalid checkpoint encoding/structure") from exc
    return body


def _body(problem, runtime, history):
    return dict(schema=SCHEMA, policy=POLICY, family=type(problem).__name__,
        definition=json.loads(problem._definition), operators=problem._operators,
        runtime=runtime, adapter_allowlist=[], generation=len(history), history=history)


def _bundle(body):
    data=canonical(body)
    if len(data) > MAX_BYTES: raise ValueError("checkpoint byte bound")
    return _Bundle(data,sha256(data).hexdigest())


class MixedReferenceOwner:
    """Own atomic reference snapshots, never an element nonlinear state store."""
    __slots__ = ("_problem","_definition","_runtime","_factor","_cache_key",
                 "_factor_digest","_lock","_dispatch","_anchor","__weakref__")

    def __setattr__(self,name,value):
        raise AttributeError("captured mixed owner attributes are immutable")

    def __init__(self,problem):
        if type(self) is not MixedReferenceOwner or type(problem) not in _TYPES:
            raise ValueError("exact reference-only mixed owner/problem required")
        with _CLAIM_LOCK:
            previous=_CLAIMS.get(problem)
            if previous is not None and previous() is not None:
                raise ValueError("reference problem already has an owner")
        type(problem)._guard(problem)
        factor=owned(np.linalg.cholesky(problem.reduced))
        runtime=runtime_digest()
        definition=problem._describe()
        dispatch=self._capture_dispatch(problem)
        cache_key=sha(dict(definition=json.loads(definition),operators=problem._operators,
                           runtime=runtime,policy=POLICY))
        initial=_bundle(_body(problem,runtime,[]))
        with _CLAIM_LOCK:
            previous=_CLAIMS.get(problem)
            if previous is not None and previous() is not None:
                raise ValueError("reference problem already has an owner")
            for name,value in dict(_problem=problem,_definition=definition,_runtime=runtime,
                    _factor=factor,_cache_key=cache_key,_factor_digest=sha(factor),
                    _lock=threading.Lock(),_dispatch=dispatch).items():
                object.__setattr__(self,name,value)
            _CLAIMS[problem]=weakref.ref(self)
            _PUBLICATIONS[self]=initial
        object.__setattr__(self,"_anchor",(self._problem,self._factor,self._lock,self._dispatch))
        _VALIDATE(self)

    @property
    def _bundle(self):
        return _PUBLICATIONS[self]

    @staticmethod
    def _capture_dispatch(problem):
        other=problem.legacy if hasattr(problem,"legacy") else problem.shell
        native=problem.native
        operator=native.operator
        # Anchor live edges as well as methods: retaining a detached old cell
        # must not let an equivalent replacement evade the method checks.
        edges=((problem,"native"),(problem,"legacy" if hasattr(problem,"legacy") else "shell"),
               (native,"operator"),(operator,"cell"),(operator,"section"),
               (operator,"reference"),(operator,"_ref"),(operator.cell,"section"))
        captured=[(obj,name,getattr(obj,name)) for obj,name in edges]
        objects=(problem,native,operator,operator.cell,operator.section,
                 operator.reference,operator._ref,other)
        # Closed object graph, including inherited recovery helpers and guard
        # dependencies. Inspect descriptors without evaluating properties.
        for obj in objects:
            names={name for cls in type(obj).__mro__ for name in vars(cls)
                   if not name.startswith("__")}
            for name in sorted(names):
                descriptor=inspect.getattr_static(type(obj),name)
                if isinstance(descriptor,property):
                    captured.append((type(obj),name,descriptor))
                elif inspect.isfunction(descriptor) or isinstance(descriptor,(staticmethod,classmethod)):
                    value=getattr(obj,name)
                    captured.append((obj,name,getattr(value,"__func__",value)))
        objects=((MixedReferenceOwner,("_guard","solve","checkpoint","restore","_capture_dispatch","_bundle")),)
        for obj,names in objects:
            for name in names:
                value=getattr(obj,name)
                captured.append((obj,name,getattr(value,"__func__",value)))
        return tuple(captured)

    def _guard(self, expected=None):
        if type(self) is not MixedReferenceOwner or type(self._problem) not in _TYPES:
            raise ValueError("reference owner identity changed")
        p=self._problem
        if any(a is not b for a,b in zip(self._anchor,(p,self._factor,self._lock,self._dispatch))):
            raise ValueError("captured owner/cache objects changed")
        with _CLAIM_LOCK:
            claim=_CLAIMS.get(p)
            if claim is None or claim() is not self: raise ValueError("foreign ownership claim")
        if expected is not None and self._bundle is not expected:
            raise ValueError("accepted bundle changed during preparation")
        if (type(self._bundle) is not _Bundle or type(self._bundle.data) is not bytes or
                sha256(self._bundle.data).hexdigest() != self._bundle.digest):
            raise ValueError("accepted bundle corrupted")
        for obj,name,original in self._dispatch:
            value=getattr(obj,name)
            if getattr(value,"__func__",value) is not original:
                raise ValueError("captured dispatch changed")
        type(p)._guard(p)
        if p._describe()!=self._definition:
            raise ValueError("captured reference graph changed")
        if (sha(self._factor)!=self._factor_digest or self._factor.flags.writeable or
            sha(dict(definition=json.loads(self._definition),operators=p._operators,
                     runtime=self._runtime,policy=POLICY))!=self._cache_key):
            raise ValueError("reference factor/cache identity changed")
        invariant(self._factor @ self._factor.T,p.reduced)

    def checkpoint(self):
        if type(self) is not MixedReferenceOwner:
            raise ValueError("exact mixed owner required")
        lock=self._lock
        if not lock.acquire(blocking=False): raise RuntimeError("mixed owner busy")
        try:
            _VALIDATE(self)
            if runtime_digest()!=self._runtime: raise ValueError("runtime changed")
            return self._bundle.data
        finally:
            lock.release()

    def solve(self, loads, *, mode="REFERENCE_LINEAR", rotation_targets=None,
              on_prepare=None, cancel=lambda:False):
        if (type(self) is not MixedReferenceOwner or mode!="REFERENCE_LINEAR" or
                rotation_targets is not None or type(loads) is not tuple or not 1<=len(loads)<=2 or
                (on_prepare is not None and not callable(on_prepare)) or not callable(cancel)):
            raise ValueError("one/two reference RHS vectors; no finite/rotational admission")
        lock=self._lock
        before=None
        captured=None
        if not lock.acquire(blocking=False): raise RuntimeError("mixed owner busy")
        try:
            # Bookkeeping itself can fail. Enter the protected region before
            # reading publication/snapshot state, and obtain rollback authority
            # from the registry, never an overridable property view.
            before=_PUBLICATIONS[self]
            captured={name:getattr(self,name) for name in MixedReferenceOwner.__slots__ if name!="__weakref__"}
            _VALIDATE(self,before)
            rhs=tuple(owned(f,(self._problem.size,)) for f in loads)
            body=_parse(before.data)
            if sum(len(row["loads"]) for row in body["history"])+len(rhs)>MAX_ENTRIES:
                raise ValueError("accepted entry bound")
            def check():
                _VALIDATE(self,before)
                stopped=cancel()
                _VALIDATE(self,before)
                if type(stopped) is not bool: raise ValueError("boolean cancellation result required")
                if stopped: raise InterruptedError("mixed preparation cancelled")
            check()
            results=[]
            for f in rhs:
                _VALIDATE(self,before)
                result=type(self._problem).solve(self._problem,f)
                p=self._problem
                cached=p.T @ cho_solve((self._factor,True),p.T.T @ f,check_finite=True)
                invariant(cached,result["u"])
                results.append(result)
                preview=canonical(result)
                for phase in ("native_prepared","other_prepared"):
                    if on_prepare is not None:
                        response=on_prepare(phase,preview)
                        _VALIDATE(self,before)
                        if response is not None: raise ValueError("preparation hook must not return authority")
                    check()
                    progress(phase)
            entry=dict(loads=rhs,results=results)
            body["history"].append(entry)
            body["generation"]+=1
            pending=_bundle(body)
            # Return allocations happen before the last user hook and guard.
            returned=json.loads(canonical(results))
            if on_prepare is not None:
                response=on_prepare("final_prepare",pending.data)
                _VALIDATE(self,before)
                if response is not None: raise ValueError("preparation hook must not return authority")
            check()
            if runtime_digest()!=self._runtime: raise ValueError("runtime changed")
            _VALIDATE(self,before)
            # Only this assignment publishes; no fallible user code follows it.
            _PUBLICATIONS[self]=pending
            return returned
        except BaseException:
            if captured is not None:
                for name,value in captured.items(): object.__setattr__(self,name,value)
            if before is not None: _PUBLICATIONS[self]=before
            raise
        finally:
            lock.release()

    @classmethod
    def restore(cls,problem,data,expected_sha256):
        if (cls is not MixedReferenceOwner or type(problem) not in _TYPES or
                type(expected_sha256) is not str or len(expected_sha256)!=64 or
                type(data) is not bytes or len(data)>MAX_BYTES or
                sha256(data).hexdigest()!=expected_sha256):
            raise ValueError("externally authenticated bounded mixed checkpoint required")
        body=_parse(data)
        if (type(body) is not dict or set(body)!={"schema","policy","family","definition",
                "operators","runtime","adapter_allowlist","generation","history"} or
                body["schema"]!=SCHEMA or body["policy"]!=POLICY or
                body["family"]!=type(problem).__name__ or body["adapter_allowlist"]!=[] or
                type(body["generation"]) is not int or type(body["history"]) is not list or
                body["generation"]!=len(body["history"]) or len(body["history"])>MAX_ENTRIES):
            raise ValueError("mixed checkpoint family/schema/count mismatch")
        type(problem)._guard(problem)
        if (canonical(body["definition"])!=problem._definition or body["operators"]!=problem._operators or
                body["runtime"]!=runtime_digest()):
            raise ValueError("mixed checkpoint frozen inputs mismatch")
        count=0
        for entry in body["history"]:
            if (type(entry) is not dict or set(entry)!={"loads","results"} or
                    type(entry["loads"]) is not list or not 1<=len(entry["loads"])<=2 or
                    type(entry["results"]) is not list or len(entry["results"])!=len(entry["loads"]) or
                    any(type(result) is not dict for result in entry["results"])):
                raise ValueError("mixed checkpoint entry schema")
            count+=len(entry["loads"])
            if count>MAX_ENTRIES: raise ValueError("mixed checkpoint entry bound")
            for f in entry["loads"]: owned(f,(problem.size,))
            for result in entry["results"]: validate_result(problem,result)
        made=cls(problem)
        try:
            for entry in body["history"]:
                results=made.solve(tuple(entry["loads"]))
                if canonical(results)!=canonical(entry["results"]):
                    raise ValueError("mixed checkpoint result replay mismatch")
            if made.checkpoint()!=data: raise ValueError("mixed checkpoint replay mismatch")
            progress("restart")
            return made
        except BaseException:
            with _CLAIM_LOCK:
                claim=_CLAIMS.get(problem)
                if claim is not None and claim() is made: del _CLAIMS[problem]
                _PUBLICATIONS.pop(made,None)
            raise


# Use the captured guard entry, so replacing the class guard cannot bypass the
# dispatch identity check at the next callback boundary.
_VALIDATE = MixedReferenceOwner._guard
