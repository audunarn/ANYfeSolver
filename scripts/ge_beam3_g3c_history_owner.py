"""Private graph checkpoint/replay composition; no public solver registration.

Import is inert. Numerical imports occur only when a reviewed runner creates
an owner, or AFTER complete checkpoint preflight when resuming. The inner owner
remains the sole publisher. No checkpoint payload is installed into its state.
"""
from hashlib import sha256
import inspect
import sys
import threading
from pathlib import Path

import ge_beam3_g3c_restart_preflight as packet

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'GE_BEAM3_G3C_STABLE_GRAPH_RESTART_V1'
POLICY = 'GE_BEAM3_G3C_STABLE_MIXED_ELASTIC_OWNER_V1'
ENVIRONMENT = '2ce154b866565e1d03019651b1ae7fdd070e5554f38d72adf21d8080558b6756'
HISTORY_SOURCES = (
    'scripts/ge_beam3_g3c_history_owner.py',
    'scripts/ge_beam3_g3c_restart_preflight.py',
    'scripts/run_ge_beam3_g3c_history.py',
)


def runtime_identity():
    """Inert identity, independently recomputed by the reviewed coordinator."""
    rows = [(p.relative_to(ROOT).as_posix(), sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest())
            for p in sorted((ROOT/'src'/'anyfesolver').rglob('*.py'))]
    rows += [(name, sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest())
             for name in HISTORY_SOURCES]
    return packet.digest(dict(environment=ENVIRONMENT, sources=rows))


def dispatch():
    rows = []
    for module in (packet, sys.modules[__name__]):
        for name, value in sorted(vars(module).items()):
            items = [(name, value)]
            if inspect.isclass(value) and value.__module__ == module.__name__:
                for cls in value.__mro__:
                    items += [(name+'.'+cls.__name__+'.'+k, v) for k,v in sorted(vars(cls).items())]
            for label, fn in items:
                if isinstance(fn, (classmethod, staticmethod)): fn = fn.__func__
                if inspect.isfunction(fn):
                    rows.append((module.__name__, label, fn, id(fn.__code__),
                                 repr(fn.__defaults__), repr(fn.__kwdefaults__)))
    rows += [('constant', name, globals()[name]) for name in ('ROOT','SCHEMA','POLICY','ENVIRONMENT','HISTORY_SOURCES')]
    rows += [('preflight_constant', name, getattr(packet,name)) for name in
             ('ROOT','CONTRACT','CONTRACT_SHA','REVIEW','REVIEW_SHA','DEFINITION','DEFINITION_SHA','MAP_SHA','HEX')]
    return tuple(rows)


def guard_signature(guard):
    fn=guard.__func__
    return (fn,fn.__code__,repr(fn.__defaults__),tuple((k,v,getattr(v,'__code__',None))
            for k,v in sorted((fn.__kwdefaults__ or {}).items())))


class HistoryOwner:
    """One immutable owned wrapper; no second accepted-state publication."""
    __slots__ = ('_owner','_owned_owner','_lock','_owned_lock','_dispatch','_runtime','_poisoned',
                 '_guard_call','_owned_guard','_guard_signature')

    def __setattr__(self, name, value): raise AttributeError('immutable history owner')
    def __delattr__(self, name): raise AttributeError('immutable history owner')

    def __init__(self, fixture_id='J_B2_PAIR', variant='BASE', common_motion='NONE'):
        if type(self) is not HistoryOwner: raise ValueError('exact history owner required')
        # The inner constructor still requires its independently reviewed lease.
        # Importing this wrapper alone never captures or manufactures that lease.
        packet.authorities()
        frozen = runtime_identity(); code = dispatch()
        from anyfesolver._ge_beam3_g3c_stable.owner import MixedGraphOwner
        owner = MixedGraphOwner(fixture_id, variant, common_motion)
        lock = threading.Lock()
        for name,value in dict(_owner=owner,_owned_owner=owner,_lock=lock,_owned_lock=lock,
                               _dispatch=code,_runtime=frozen,_poisoned=False).items():
            object.__setattr__(self,name,value)
        guard = self._guard
        object.__setattr__(self,'_guard_call',guard)
        object.__setattr__(self,'_owned_guard',guard)
        object.__setattr__(self,'_guard_signature',guard_signature(guard))
        guard(full=True)

    def _guarding(self):
        guard=self._owned_guard; signature=self._guard_signature
        def checked():
            # Check OUTSIDE the mutable guard callable, both before and after
            # invoking it. A captured bound method still has mutable __code__.
            for phase in (0,1):
                fn=guard.__func__
                actual=(fn,fn.__code__,repr(fn.__defaults__),tuple((k,v,getattr(v,'__code__',None))
                        for k,v in sorted((fn.__kwdefaults__ or {}).items())))
                if actual!=signature or self._guard_signature is not signature:
                    object.__setattr__(self,'_poisoned',True)
                    raise ValueError('captured history guard code/default identity changed')
                if phase==0: guard(full=True)
        return checked

    def _guard(self, *, full=False, _dispatch_fn=dispatch,
               _runtime_fn=runtime_identity, _authority_fn=packet.authorities):
        try:
            if (type(self) is not HistoryOwner or self._poisoned or self._owner is not self._owned_owner
                    or self._lock is not self._owned_lock or self._guard_call is not self._owned_guard
                    or _dispatch_fn() != self._dispatch):
                raise ValueError('history runtime/owner/lock identity changed')
            if full:
                _authority_fn()
                if _runtime_fn() != self._runtime: raise ValueError('history runtime source changed')
        except BaseException:
            object.__setattr__(self,'_poisoned',True)
            raise

    def solve(self, command, *, hook=None):
        """Delegate the only publication to the real accepted owner."""
        if hook is not None and not callable(hook): raise ValueError('callback required')
        lock = self._owned_lock
        if not lock.acquire(False): raise RuntimeError('history owner in use')
        try:
            guard = self._guarding()
            guard()
            # Copy command before callbacks, but retain the real owner's exact
            # sequence validation and nonce/sandbox transaction handling.
            copied = packet.strict(packet.canonical(command))
            def check(stage):
                guard()
                if hook is not None:
                    hook(stage)
                    guard()
            # No fallible wrapper code follows the inner publication point.
            return self._owned_owner.solve(copied, hook=check)
        finally:
            lock.release()

    def checkpoint_bytes(self):
        lock = self._owned_lock
        if not lock.acquire(False): raise RuntimeError('history owner in use')
        try:
            guard = self._guarding()
            guard()
            owner = self._owned_owner; inner = owner._owned_lock
            if not inner.acquire(False): raise RuntimeError('inner owner in use')
            try:
                owner._guard(full=True,lock=inner)
                if owner._active is not None: raise ValueError('checkpoint during active trial')
                generation = owner._published
                value = dict(schema=SCHEMA,policy=POLICY,
                    definition=packet.strict(owner._definition),
                    definition_sha256=sha256(owner._definition).hexdigest(),
                    final_state=packet.strict(generation.state),
                    history=packet.strict(generation.history),
                    initial_sha256=owner._initial,final_sha256=generation.state_sha256,
                    runtime_sha256=self._runtime)
                raw = packet.canonical(value)
                if len(raw) > 8*1024**2: raise ValueError('checkpoint byte bound')
                owner._guard(generation,full=True,lock=inner)
                guard()
                return raw
            finally:
                inner.release()
        finally:
            lock.release()

    def accepted_pose_diagnostic_bytes(self):
        """Fixed last-accepted-load diagnostic, never a new admissible command.

The immutable journal proves which command was accepted. Revalidate it against
its actual journal prefix, then evaluate only the actual published pose with a
genuine owned nonce/sandbox. There is no caller pose, origin, load or callback.
"""
        lock = self._owned_lock
        if not lock.acquire(False): raise RuntimeError('history owner in use')
        try:
            guard = self._guarding(); guard()
            owner = self._owned_owner; inner = owner._owned_lock
            if not inner.acquire(False): raise RuntimeError('inner owner in use')
            sandbox = None; nonce = None
            try:
                owner._guard(full=True,lock=inner)
                if owner._active is not None: raise ValueError('diagnostic during trial')
                origin = owner._published
                history = packet.strict(origin.history); state = packet.strict(origin.state)
                if not history: raise ValueError('accepted stage required for rebase diagnostic')
                from anyfesolver._ge_beam3_g3c_stable import owner as mechanics
                command = mechanics.command(history[-1]['command'],history[:-1],
                                             packet.strict(owner._definition)['common_motion'])
                if history[-1]['accepted_state_sha256'] != origin.state_sha256:
                    raise ValueError('accepted diagnostic journal mismatch')
                nonce = object(); object.__setattr__(owner,'_active',nonce)
                object.__setattr__(owner,'_serial',owner._serial+1)
                def check(stage):
                    guard(); owner._guard(origin,full=True,lock=inner)
                    if owner._active is not nonce: raise ValueError('diagnostic capability changed')
                    print('G3C CHECKPOINT accepted pose '+stage,flush=True)
                check('initialization')
                r,A,candidate,sandbox,_ = owner._evaluate(state['total_u'],state['multipliers'],command,origin,check)
                raw = mechanics.canonical(dict(kind='UNQUALIFIED_G3C_ACCEPTED_POSE_DIAGNOSTIC',
                    origin_sha256=origin.state_sha256,command=command,residual=r,tangent=A,
                    candidate=candidate,state_committed=False,physical_recovery_complete=False))
                check('complete')
                return raw
            finally:
                try:
                    if sandbox is not None: owner._discard(sandbox)
                finally:
                    if nonce is not None: object.__setattr__(owner,'_active',None)
                    inner.release()
        finally:
            lock.release()


def resume(raw, expected_sha256, *, expected_runtime_sha256):
    """Fresh genuine replay only; returns no owner on any mismatch.

The expected hashes are independent external authority. We never derive the
expected checkpoint digest from the untrusted input as an admission shortcut.
No caller owner/candidate/state injection or persistent numerical cache exists.
"""
    code = dispatch()
    if runtime_identity() != expected_runtime_sha256: raise ValueError('external runtime mismatch')
    checked = packet.preflight(raw, expected_sha256, expected_runtime_sha256=expected_runtime_sha256)
    if dispatch() != code: raise ValueError('preflight dispatch changed')
    value = packet.strict(checked.raw)
    # First possible numerical construction: complete nested preflight passed.
    owner = HistoryOwner(checked.fixture_id,checked.variant,checked.common_motion)
    current = packet.strict(owner.checkpoint_bytes())
    if current['initial_sha256'] != value['initial_sha256']:
        raise ValueError('genuine virgin state mismatch')
    print('G3C CHECKPOINT restart prefix 0',flush=True)
    for index, entry in enumerate(value['history']):
        owner.solve(entry['command'])
        current = packet.strict(owner.checkpoint_bytes())
        if (packet.canonical(current['history']) != packet.canonical(value['history'][:index+1])
                or current['final_sha256'] != entry['accepted_state_sha256']):
            raise ValueError('genuine replay prefix mismatch: '+str(index+1))
        print('G3C CHECKPOINT restart prefix '+str(index+1),flush=True)
    if owner.checkpoint_bytes() != raw: raise ValueError('complete replay checkpoint mismatch')
    if dispatch() != code or runtime_identity() != expected_runtime_sha256:
        raise ValueError('replay runtime changed')
    return owner
