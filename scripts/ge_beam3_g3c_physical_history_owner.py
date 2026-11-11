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

import ge_beam3_g3c_physical_restart_preflight as packet

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'GE_BEAM3_G3C_PHYSICAL_GRAPH_RESTART_V1'
POLICY = 'GE_BEAM3_G3C_PHYSICAL_MIXED_ELASTIC_OWNER_V1'
ENVIRONMENT = '2ce154b866565e1d03019651b1ae7fdd070e5554f38d72adf21d8080558b6756'
HISTORY_SOURCES = (
    'scripts/ge_beam3_g3c_correction_lease_binding.py',
    'scripts/ge_beam3_g3c_physical_history_owner.py',
    'scripts/ge_beam3_g3c_physical_restart_preflight.py',
    'scripts/run_ge_beam3_qualification.py',
)
GRAPHS=('J_B2_PAIR','J_B3_PAIR','J_Q4_PAIR','J_S3_PAIR','J_MULTIFAMILY_LOOP')
VARIANTS=('BASE','SHUFFLED_INSERTION','RENUMBERED','CONNECTIVITY_REVERSED','PROPER_GLOBAL_TRANSFORM')
SCALES=(.01,1.,10.)
MOTIONS=('NONE','CM0','CM1','CM2','CM3')
CORRECTION_COMPATIBILITY_SCHEMA='GE_BEAM3_G3C_PHYSICAL_RUNTIME_COMPATIBILITY_V3'
CORRECTION_PREDECESSOR_RUNTIME='41a0dddda886672479294953871e83be3073ed38e129e12f3fa210bfbf3ce87c'
CORRECTION_ADDENDUM_SHA='2c03f72a0e9c50c6a22fe5b7f47fd66f786aa3eadc23bedeb4dab1f2045a8696'
CORRECTION_DESIGN_REVIEW_SHA='0fbaedb533e501a5136407d5dc039b5393b4bbf53064c273c7e974917ba9fec9'
CORRECTION_REVISION_SHA='dae0dd3081ff45ab66a266b2706bdee6049e556d01c09003de2ba1749ce0be76'
CORRECTION_REVISION_REVIEW_SHA='339367400bc48eeb07eb09eca353097297e7dae2907f23ae53b53dab5199c26b'
CORRECTION_RECOVERY_SHA='faabb5d96fd7e2fbd74f115a46e050dcb27df6bf05db9f7d5ce91d8ee5017bdd'
CORRECTION_RECOVERY_REVIEW_SHA='f20539245dae880b332905a5e86cde3a42a840dbaa6a7b14731b7946a0b5498b'
CORRECTION_GUARD_SEGMENT_MANIFEST_SHA='8ae0c6560960ef9bf9035838abb5daae056372d7aae0f9426945330a898bf477'
CORRECTION_UNCHANGED_INPUTS_SHA='34a2f26c67aa71d7d443f514ca143657dc8a82fd0077108c53a922fb07230d1c'
CORRECTION_ALLOWED_CHANGED_PATHS=(
    'docs/GE_BEAM3_G3C_PHYSICAL_CORRECTION_INHERITANCE_ADDENDUM.md',
    'docs/GE_BEAM3_G3C_PHYSICAL_CORRECTION_INHERITANCE_V2.md',
    'docs/GE_BEAM3_G3C_PHYSICAL_CORRECTION_PARTITION_RECOVERY_V3.md',
    'docs/GE_BEAM3_QUALIFICATION_COMPLETION_REGISTER.md',
    'docs/reference_cases/ge_beam3_g3c_physical_correction_inheritance_review_v1.json',
    'docs/reference_cases/ge_beam3_g3c_physical_correction_inheritance_v2_review.json',
    'docs/reference_cases/ge_beam3_g3c_physical_correction_inheritance_v2_review_v2.json',
    'docs/reference_cases/ge_beam3_g3c_physical_correction_partition_implementation_review_v3_initial.json',
    'docs/reference_cases/ge_beam3_g3c_physical_correction_partition_recovery_review_v3.json',
    'docs/reference_cases/ge_beam3_g3c_physical_correction_partition_recovery_review_v3_correction.json',
    'docs/reference_cases/ge_beam3_g3c_physical_correction_partition_recovery_review_v3_initial.json',
    'scripts/ge_beam3_g3c_correction_lease_binding.py',
    'scripts/ge_beam3_g3c_physical_history_owner.py',
    'scripts/ge_beam3_g3c_rehearsal_mutations.py',
    'scripts/run_ge_beam3_qualification.py',
    'tests/test_ge_beam3_g3c_physical_correction_guards.py',
    'tests/test_ge_beam3_g3c_rehearsal_mutations_static.py',
    'tests/test_ge_beam3_qualification_runner.py',
)


def commands(motion,scale):
    if type(motion)is not str or motion not in MOTIONS or type(scale)is not float or scale not in SCALES:
        raise ValueError('registered physical history required')
    return ([] if motion=='NONE' else [dict(kind='PREPARE_COMMON_MOTION',step=i) for i in range(1,5)])+[
        dict(kind='LOAD_STAGE',load_factor=v,force_scale=scale,root_stage=i) for i,v in enumerate((0.,.5,1.,.25,0.))]


def history_matrix():
    return [dict(case_id=f'{g}::{v}::S{s}::{m}',graph=g,variant=v,force_scale=scale,common_motion=m,
                 accepted_stages=len(commands(m,scale)),commands_sha256=packet.digest(commands(m,scale)))
            for g in GRAPHS for v in VARIANTS for s,scale in enumerate(SCALES) for m in MOTIONS]


def work_inventory(lane):
    """Fixed inert assignment data; never imports a numerical package."""
    rows=history_matrix()
    if lane=='smoke':rows=[r for r in rows if r['variant']=='BASE' and r['force_scale']==.01 and r['common_motion']=='NONE']
    elif lane=='rehearsal':rows=[r for r in rows if r['variant']=='BASE' and ((r['force_scale']==.01 and r['common_motion']=='NONE') or (r['force_scale']==10. and r['common_motion']=='CM3'))]
    elif lane!='formal':raise ValueError('registered physical lane')
    history=[dict(kind='history',case=r,stages=2 if lane=='smoke' else r['accepted_stages']) for r in rows]
    prefixes=[] if lane=='smoke' else [dict(kind='prefix',case=r,prefix=k) for r in rows for k in range(r['accepted_stages']+1)]
    return history+prefixes


def mutation_inventory():
    """Original 142 actual probes; authority probes stay with the coordinator."""
    import ge_beam3_g3c_rehearsal_contract as inherited
    rows=inherited.expected()['mutation_probes']
    return [dict(row,executor='authority' if row['category']=='R10_NORMAL_SOURCE' or row['member']=='changed_implementation_review' else 'replay') for row in rows]


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
    rows += [('constant', name, globals()[name]) for name in ('ROOT','SCHEMA','POLICY','ENVIRONMENT','HISTORY_SOURCES','GRAPHS','VARIANTS','SCALES','MOTIONS')]
    rows += [('preflight_constant', name, getattr(packet,name)) for name in
             ('ROOT','CONTRACT','CONTRACT_SHA','REVIEW','REVIEW_SHA','DEFINITION','DEFINITION_SHA','MAP_SHA','HEX','PHYSICAL_CONTRACT','PHYSICAL_CONTRACT_SHA','PHYSICAL_AUTHORITY','PHYSICAL_AUTHORITY_SHA')]
    return tuple(rows)


def guard_signature(guard):
    fn=guard.__func__
    return (fn,fn.__code__,repr(fn.__defaults__),tuple((k,v,getattr(v,'__code__',None))
            for k,v in sorted((fn.__kwdefaults__ or {}).items())))


def validate_runtime_compatibility(value, expected_runtime, live_runtime):
    """Admit only the exact token retained by the captured validated lease."""
    keys={'schema','mode','predecessor','successor','addendum_sha256','design_review_sha256',
          'revision_sha256','revision_review_sha256','unchanged_inputs_sha256',
          'partition_recovery_sha256','partition_recovery_review_sha256',
          'guard_segment_manifest_sha256','allowed_changed_paths','self_sha256'}
    if type(value)is not dict or set(value)!=keys:raise ValueError('runtime compatibility schema')
    body={key:item for key,item in value.items()if key!='self_sha256'}
    predecessor=value.get('predecessor');successor=value.get('successor')
    if (value.get('schema')!=CORRECTION_COMPATIBILITY_SCHEMA or value.get('mode')!='R-GUARDS'
        or type(predecessor)is not dict or set(predecessor)!={'commit','tree','runtime_sha256','review_sha256'}
        or predecessor!={'commit':'f6a62518be52a414604aa5e1beddd4601093faca',
                         'tree':'1230eea2b64ca6e585ad23b389e514f1d5c493c4',
                         'runtime_sha256':CORRECTION_PREDECESSOR_RUNTIME,
                         'review_sha256':'52f8df02bd22c635bf828bf93190a34ec1463b20268a1a822e4e0f3c6c042eaf'}
        or type(successor)is not dict or set(successor)!={'commit','tree','runtime_sha256','review_sha256'}
        or successor.get('runtime_sha256')!=live_runtime
        or any(type(successor.get(key))is not str for key in successor)
        or len(successor.get('commit',''))!=40 or len(successor.get('tree',''))!=40
        or value.get('addendum_sha256')!=CORRECTION_ADDENDUM_SHA
        or value.get('design_review_sha256')!=CORRECTION_DESIGN_REVIEW_SHA
        or value.get('revision_sha256')!=CORRECTION_REVISION_SHA
        or value.get('revision_review_sha256')!=CORRECTION_REVISION_REVIEW_SHA
        or value.get('partition_recovery_sha256')!=CORRECTION_RECOVERY_SHA
        or value.get('partition_recovery_review_sha256')!=CORRECTION_RECOVERY_REVIEW_SHA
        or value.get('guard_segment_manifest_sha256')!=CORRECTION_GUARD_SEGMENT_MANIFEST_SHA
        or value.get('unchanged_inputs_sha256')!=CORRECTION_UNCHANGED_INPUTS_SHA
        or type(value.get('allowed_changed_paths'))is not list
        or value.get('allowed_changed_paths')!=list(CORRECTION_ALLOWED_CHANGED_PATHS)
        or value.get('self_sha256')!=packet.digest(body)
        or expected_runtime!=CORRECTION_PREDECESSOR_RUNTIME):
        raise ValueError('runtime compatibility authority')
    for digest in (successor['runtime_sha256'],successor['review_sha256'],
                   value['unchanged_inputs_sha256'],value['self_sha256']):
        packet.hash_value(digest)
    import ge_beam3_g3c_correction_lease_binding as binding
    if binding.compatibility_identity()!=packet.digest(value):
        raise ValueError('runtime compatibility not captured by lease')
    return live_runtime


def compatible_checkpoint_equal(predecessor_raw, successor_raw, predecessor_runtime, successor_runtime):
    """Compare immutable packets through one authenticated runtime-only view."""
    predecessor=packet.strict(predecessor_raw);successor=packet.strict(successor_raw)
    if (packet.canonical(predecessor)!=predecessor_raw or packet.canonical(successor)!=successor_raw
        or predecessor.get('runtime_sha256')!=predecessor_runtime
        or successor.get('runtime_sha256')!=successor_runtime
        or set(predecessor)!=set(successor)):
        return False
    view=dict(successor,runtime_sha256=predecessor_runtime)
    return packet.canonical(view)==predecessor_raw


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
        from anyfesolver._ge_beam3_g3c_physical_owner import MixedGraphOwner
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

    @property
    def size(self):
        lock=self._owned_lock
        if not lock.acquire(False):raise RuntimeError('history owner in use')
        try:
            guard=self._guarding();guard();value=self._owned_owner.size;guard();return value
        finally:lock.release()

    def snapshot_bytes(self):
        """Nonimportable diagnostic view used by directional qualification."""
        lock=self._owned_lock
        if not lock.acquire(False):raise RuntimeError('history owner in use')
        try:
            guard=self._guarding();guard();raw=self._owned_owner.snapshot_bytes();guard();return bytes(raw)
        finally:lock.release()

    def trial(self,total,multipliers,command,*,hook=None):
        """Nonpublishing trial under both owner guards; never restart input."""
        if hook is not None and not callable(hook):raise ValueError('callback required')
        lock=self._owned_lock
        if not lock.acquire(False):raise RuntimeError('history owner in use')
        try:
            guard=self._guarding();guard();copied=packet.strict(packet.canonical(command))
            def check(stage):
                guard()
                if hook is not None:hook(stage);guard()
            result=self._owned_owner.trial(total,multipliers,copied,hook=check)
            guard();return result
        finally:lock.release()

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
                from anyfesolver import _ge_beam3_g3c_physical_owner as mechanics
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


def resume(raw, expected_sha256, *, expected_runtime_sha256, runtime_compatibility=None):
    """Fresh genuine replay only; returns no owner on any mismatch.

The expected hashes are independent external authority. We never derive the
expected checkpoint digest from the untrusted input as an admission shortcut.
No caller owner/candidate/state injection or persistent numerical cache exists.
"""
    code = dispatch()
    live_runtime=runtime_identity()
    if live_runtime != expected_runtime_sha256:
        validate_runtime_compatibility(runtime_compatibility,expected_runtime_sha256,live_runtime)
    elif runtime_compatibility is not None:raise ValueError('runtime compatibility unnecessary')
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
    replayed=owner.checkpoint_bytes()
    if (replayed!=raw if runtime_compatibility is None else
            not compatible_checkpoint_equal(raw,replayed,expected_runtime_sha256,live_runtime)):
        raise ValueError('complete replay checkpoint mismatch')
    if dispatch() != code or runtime_identity() != live_runtime:
        raise ValueError('replay runtime changed')
    return owner
