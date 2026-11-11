"""Durable definition owner over unchanged, bounded coupled transactions.

No live 120-second Context is retained between operations. Each operation
reconstructs the same definition and mechanically replays authenticated history
into a fresh inner owner. Historical Context deadlines, identities and records
are never reset, extended or rebound. This private delivery facade is not a
qualification flag or a generic mixed-FEModel assembly interface.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import json
from threading import Lock

import numpy as np

from .control import cancellation_safe_point
from ._ge_beam3_native_analysis import NativeBeamAnalysis
from ._ge_beam3_native_definition import NativeBeamDefinition
from ._ge_beam3_retained_generalized_state import Program
from ._ge_beam3_coupled_beam_subdomain import CoupledBeamSubdomain
from ._ge_beam3_shell_joint_trial import ShellBeamTrialAssembly
from ._ge_beam3_shell_joint_state import CoupledShellBeamAnalysis, decode
from ._ge_beam3_p5_seeded.core import canonical
from ._ge_beam3_p5_seeded.codec import MAX_BYTES

SCHEMA = 'GE_BEAM3_DURABLE_COUPLED_DEFINITION_CHECKPOINT_V1'
# An ASCII byte needs at most six JSON string bytes, plus fixed envelope fields.
# This is an OUTER serialization bound; the inner 2MiB policy remains unchanged.
OUTER_MAX_BYTES = 6*MAX_BYTES + 4096
SHELL_KEYS = frozenset(('topology', 'coordinates', 'reference_normal', 'thickness',
    'elastic_modulus', 'poisson_ratio', 'shell_node', 'beam_node'))


def digest(raw):
    return sha256(raw).hexdigest()


def _outer(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= OUTER_MAX_BYTES:
        raise ValueError('bounded durable outer checkpoint required')
    quoted = escaped = False
    depth = 0
    for byte in raw:
        if quoted:
            if escaped: escaped = False
            elif byte == 92: escaped = True
            elif byte == 34: quoted = False
        elif byte == 34: quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > 32: raise ValueError('durable checkpoint nesting limit')
        elif byte in (93, 125): depth -= 1
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value: raise ValueError('duplicate durable checkpoint key')
            value[key] = item
        return value
    def forbidden(value): raise ValueError('nonfinite durable checkpoint token')
    try:
        value = json.loads(raw.decode('ascii'), object_pairs_hook=pairs, parse_constant=forbidden)
        if canonical(value) != raw: raise ValueError('noncanonical durable checkpoint')
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError('invalid durable checkpoint text') from error
    return value


@dataclass(frozen=True)
class DurableCoupledRun:
    status: str
    checkpoint: bytes
    cursor: int
    failure: str | None
    definition_sha256: str
    production_qualified: bool = field(default=False, init=False)

    @property
    def checkpoint_sha256(self):
        return digest(self.checkpoint)


class DurableCoupledBeamAnalysis:
    """Owned immutable inputs and canonical histories, not a retained Context.

Admits the reviewed single elastic Q4/S3 V2D and generalized-beam connection.
The nodal force vector is shell rows followed by ordered beam node rows; only
spatial dead forces are admitted. All six slave-beam joint DOFs must remain free
for modal/buckling operations, as required by the existing spectral adapter.
No checkpoint is stored implicitly: the caller explicitly owns each returned
accepted prefix and must supply its exact hash for subsequent operations.
"""

    def __init__(self, definitions, boundaries, beam_program, *, shell, targets,
                 shell_fixed, nodal_forces, shell_density=None,
                 max_iterations=24, max_backtracks=8):
        if (type(definitions) is not tuple or not definitions
                or any(type(row) is not NativeBeamDefinition for row in definitions)
                or type(beam_program) is not Program
                or type(shell) is not dict or set(shell) != SHELL_KEYS):
            raise ValueError('explicit durable coupled definitions, program and shell required')
        if shell_density is not None and (type(shell_density) is not float
                or not np.isfinite(shell_density) or shell_density <= 0.):
            raise ValueError('positive physical shell density or explicit static-only None required')
        self._definitions = tuple(row.raw for row in definitions)
        self._boundaries = deepcopy(boundaries)
        self._program = deepcopy(beam_program)
        self._shell = deepcopy(shell)
        self._controls = deepcopy(dict(targets=targets, shell_fixed=shell_fixed,
            nodal_forces=nodal_forces, max_iterations=max_iterations, max_backtracks=max_backtracks))
        self._density = shell_density
        self._inner_identity = None
        inner, _ = self._new()
        self._inner_identity = inner.identity
        self._spectral_size = inner.assembly.shell_count + inner.assembly.beam.nodal_count + 6*len(definitions)
        self.identity = digest(self._snapshot())
        self._identity = self.identity
        self._lock = Lock()
        # `inner` is deliberately discarded. Even initial recovery reconstructs.

    def _snapshot(self):
        return canonical(dict(schema=SCHEMA, definitions=[raw.decode('ascii') for raw in self._definitions],
            boundaries=[vars(row) for row in self._boundaries], program=self._program,
            shell=self._shell, controls=self._controls, shell_density=self._density,
            inner_identity=self._inner_identity, spectral_size=self._spectral_size, production_qualified=False))

    def _guard(self):
        if self.identity != self._identity or digest(self._snapshot()) != self.identity:
            raise ValueError('durable coupled definition changed')

    def _new(self, cancellation_token=None):
        cancellation_safe_point(cancellation_token, 'durable-coupled.before-reconstruction')
        native = NativeBeamAnalysis(tuple(NativeBeamDefinition(raw) for raw in self._definitions),
            deepcopy(self._boundaries))
        native._family_required('GENERALIZED_DISTRIBUTED')
        cancellation_safe_point(cancellation_token, 'durable-coupled.definitions-reconstructed')
        beam = CoupledBeamSubdomain(native.model, deepcopy(self._program))
        assembly = ShellBeamTrialAssembly(beam, **deepcopy(self._shell), variational_shell=True)
        cancellation_safe_point(cancellation_token, 'durable-coupled.assembly-reconstructed')
        owner = CoupledShellBeamAnalysis(assembly, **deepcopy(self._controls))
        if self._inner_identity is not None and owner.identity != self._inner_identity:
            raise ValueError('reconstructed coupled owner identity changed')
        cancellation_safe_point(cancellation_token, 'durable-coupled.owner-reconstructed')
        return owner, native._inertias

    @contextmanager
    def _operation(self, token):
        cancellation_safe_point(token, 'durable-coupled.start')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('concurrent durable coupled operation')
        try:
            self._guard()
            yield
            self._guard()
        finally:
            self._lock.release()

    def _backend(self, raw, expected_sha256):
        if (type(raw) is not bytes or type(expected_sha256) is not str
                or not 0 < len(raw) <= OUTER_MAX_BYTES or digest(raw) != expected_sha256):
            raise ValueError('bounded durable checkpoint and exact external hash required')
        value = _outer(raw)
        if (type(value) is not dict or set(value) != {'schema', 'definition_sha256',
                'backend', 'backend_sha256', 'production_qualified'}
                or value['schema'] != SCHEMA or value['definition_sha256'] != self.identity
                or value['production_qualified'] is not False or type(value['backend']) is not str):
            raise ValueError('durable checkpoint schema/definition mismatch')
        try:
            backend = value['backend'].encode('ascii')
        except UnicodeError as error:
            raise ValueError('ASCII coupled backend required') from error
        if digest(backend) != value['backend_sha256']:
            raise ValueError('durable backend hash mismatch')
        # Shape/record/equilibrium authority is supplied by full inner replay,
        # not by this envelope hash or a fabricated state-object identity.
        decode(backend)
        return backend, value['backend_sha256']

    def _envelope(self, backend):
        raw = canonical(dict(schema=SCHEMA, definition_sha256=self.identity,
            backend=backend.decode('ascii'), backend_sha256=digest(backend), production_qualified=False))
        self._backend(raw, digest(raw))
        return raw

    def solve(self, *, checkpoint=None, expected_sha256=None, stop_after=None,
              cancellation_token=None, progress=None):
        if (checkpoint is None) != (expected_sha256 is None):
            raise ValueError('durable checkpoint/hash pair required')
        if progress is not None and not callable(progress):
            raise ValueError('callable durable coupled progress observer required')
        end = len(self._controls['targets']) if stop_after is None else stop_after
        if type(end) is not int or not 0 <= end <= len(self._controls['targets']):
            raise ValueError('bounded durable coupled stop cursor required')
        with self._operation(cancellation_token):
            backend, expected = (None, None) if checkpoint is None else self._backend(checkpoint, expected_sha256)
            owner, _ = self._new(cancellation_token)
            def observed(row):
                self._guard()
                if progress is not None: progress(row)
                self._guard()
            run = owner.solve(checkpoint=backend, expected_sha256=expected, stop_after=end,
                cancellation_token=cancellation_token, progress=observed)
            # Preserve the genuine accepted prefix also on cancellation/failure.
            # Never convert its state into one issued by another owner.
            return DurableCoupledRun(run.status, self._envelope(run.checkpoint),
                run.state.cursor, run.failure, self.identity)

    def _restored(self, raw, expected, token):
        backend, digest_ = self._backend(raw, expected)
        owner, inertias = self._new(token)
        # This owner is fresh, private to this operation and never escapes.
        state, _ = owner._restore(backend, digest_, token)
        return owner, state, inertias

    def recover(self, checkpoint, *, expected_sha256, cancellation_token=None):
        with self._operation(cancellation_token):
            owner, state, _ = self._restored(checkpoint, expected_sha256, cancellation_token)
            result = owner.recover(state)
            cancellation_safe_point(cancellation_token, 'durable-coupled.recovery-complete')
            return result

    def modes(self, checkpoint, *, expected_sha256, bounds, num_modes=6,
              root_width=1e-10, relative_width=1e-12, cancellation_token=None):
        if self._density is None:
            raise ValueError('durable coupled modes require physical shell density')
        from ._ge_beam3_coupled_modes import modes
        with self._operation(cancellation_token):
            if (type(bounds) is not tuple or len(bounds) != 2
                    or any(type(x) not in (float, int) or not np.isfinite(x) for x in bounds)
                    or bounds[0] >= bounds[1] or type(num_modes) is not int
                    or not 1 <= num_modes <= self._spectral_size <= 256
                    or type(root_width) is not float or not np.isfinite(root_width) or root_width <= 0.
                    or type(relative_width) is not float or not np.isfinite(relative_width)
                    or not 0. < relative_width <= 1e-10):
                raise ValueError('explicit bounded durable modal controls required')
            owner, state, inertias = self._restored(checkpoint, expected_sha256, cancellation_token)
            return modes(owner, state, section_inertias=inertias, shell_density=self._density,
                bounds=bounds, num_modes=num_modes, root_width=root_width,
                relative_width=relative_width, cancellation_token=cancellation_token)

    def buckling(self, checkpoint, *, expected_sha256, bounds, num_modes=6, cancellation_token=None):
        from ._ge_beam3_coupled_modes import buckling
        from ._ge_beam3_native_buckling import _controls
        with self._operation(cancellation_token):
            _controls(bounds, num_modes)
            if self._spectral_size > 256:
                raise ValueError('durable coupled spectral coordinate bound')
            owner, state, _ = self._restored(checkpoint, expected_sha256, cancellation_token)
            return buckling(owner, state, bounds=bounds, num_modes=num_modes,
                cancellation_token=cancellation_token)
