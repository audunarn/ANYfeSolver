"""Immutable numeric snapshot for the prepared static evaluator.

The snapshot is captured once against the live model and then owns every
container.  Backing arrays are immutable buffers and exposed mappings are
read-only, so after preparation the evaluator never observes the source model,
mesh, elements, materials, session or caller load object. Canonical recovery
refinement uses separate, private local element/mesh/material copies. Mutating or
deleting the source model, or closing the session that happened to be used
during preparation, cannot change or invalidate evaluator state.

Capture compares the existing comprehensive structural witness before and after
preparation, including raw values, constraint coefficients and class facts.
Canonical assembly and recovery retain their own authority guards. No outer
trusted scope is granted to arbitrary preparation callbacks.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from .analysis_session import _immutable_array, _immutable_csr
from .prepared_static_loads import (
    CanonicalLoadMaps,
    compile_canonical_load_maps,
)
from .prepared_static_recovery import (
    OwnedLocalRecovery,
    PreparedRecoveryCompilation,
    ReactionMaps,
    capture_reaction_maps,
    compile_affine_recovery,
)

if TYPE_CHECKING:
    from .fe_core import FEModel
    from .recovery import RecoveryConfig, ResourceConfig


def _plain_copy(value: Any, *, depth: int = 0) -> Any:
    """Copy only plain snapshot-safe data; refuse anything live."""

    if depth > 12:
        raise ValueError("snapshot info nesting too deep")
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if type(value) is np.ndarray:
        if value.dtype.kind not in "biuf":
            raise ValueError(f"snapshot info array dtype {value.dtype!r} is not plain numeric")
        return _immutable_array(value)
    if type(value) is dict:
        return {str(key): _plain_copy(item, depth=depth + 1) for key, item in value.items()}
    if type(value) in (list, tuple):
        made = [_plain_copy(item, depth=depth + 1) for item in value]
        return type(value)(made)
    if isinstance(value, np.generic):
        return value.item()
    raise ValueError(f"snapshot info contains a live object of type {type(value).__name__}")


def _digest_array(digest: "hashlib._Hash", values: np.ndarray) -> None:
    made = np.ascontiguousarray(values)
    digest.update(str(made.dtype).encode("ascii"))
    digest.update(str(made.shape).encode("ascii"))
    digest.update(made.tobytes())


def _digest_matrix(digest: "hashlib._Hash", matrix: sparse.csr_matrix) -> None:
    digest.update(str(matrix.shape).encode("ascii"))
    _digest_array(digest, matrix.data)
    _digest_array(digest, matrix.indices)
    _digest_array(digest, matrix.indptr)


@dataclass(frozen=True)
class PreparedStaticSnapshot:
    """One fixed structure's immutable numerical state."""

    total_dofs: int
    num_nodes: int
    num_elements: int
    stiffness: sparse.csr_matrix
    transformation: sparse.csr_matrix
    prescribed_offset: np.ndarray
    independent_dofs: np.ndarray
    reduced_stiffness: sparse.csr_matrix
    constraint_info: Mapping[str, Any]
    rigid_modes: np.ndarray
    nullspace_info: Mapping[str, Any]
    load_maps: CanonicalLoadMaps
    reaction_maps: ReactionMaps
    recovery_compilation: Optional[PreparedRecoveryCompilation]
    selected_node_ids: Tuple[int, ...]
    selected_element_ids: Tuple[int, ...]
    memory_estimate: Mapping[str, Any]
    retained_bytes: int
    setup_seconds: float
    content_digest: str
    source_revision: Mapping[str, int]
    phase_timings: Mapping[str, float]
    envelope_plan: Any = None

    def nbytes(self) -> int:
        total = int(self.retained_bytes)
        for matrix in (self.stiffness, self.transformation, self.reduced_stiffness):
            total += int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
        total += int(self.prescribed_offset.nbytes + self.independent_dofs.nbytes
                     + self.rigid_modes.nbytes)
        total += int(self.load_maps.retained_bytes)
        if self.recovery_compilation is not None:
            total += int(self.recovery_compilation.retained_bytes)
        if self.envelope_plan is not None:
            total += self.envelope_plan.retained_bytes
        return int(total)

    def content_digest_hex(self) -> str:
        """Digest all owned mechanical operators and output/load metadata."""
        digest = hashlib.sha256()
        digest.update(b"prepared_static_snapshot_v4")
        for name in (
            "total_dofs", "num_nodes", "num_elements", "stiffness",
            "transformation", "prescribed_offset", "independent_dofs",
            "reduced_stiffness", "rigid_modes", "load_maps", "reaction_maps",
            "recovery_compilation", "selected_node_ids", "selected_element_ids", "envelope_plan",
        ):
            _digest_value(digest, name)
            _digest_value(digest, getattr(self, name))
        return digest.hexdigest()


def _digest_value(digest, value):
    # Only evaluator-owned records reach this walker. Length framing avoids
    # concatenation ambiguity; sparse metadata and all affine offsets count.
    if sparse.issparse(value):
        digest.update(b"sparse")
        _digest_matrix(digest, value)
    elif isinstance(value, np.ndarray):
        digest.update(b"array")
        _digest_array(digest, value)
    elif type(value) is OwnedLocalRecovery:
        # Private canonical execution objects own the same mechanics described
        # by this frozen recipe. Do not hash object addresses, mutable caches,
        # or allocator-dependent storage estimates as mechanical provenance.
        _digest_value(digest, "owned_local_recovery_v1")
        _digest_value(digest, value.numeric_identity)
    elif is_dataclass(value):
        _digest_value(digest, type(value).__name__)
        for field in fields(value):
            if field.name not in {"setup_seconds", "retained_bytes", "probe_count",
                                  "verification", "verification_relative_residual"}:
                _digest_value(digest, field.name)
                _digest_value(digest, getattr(value, field.name))
    elif isinstance(value, Mapping):
        _digest_value(digest, ("mapping", len(value)))
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _digest_value(digest, key)
            _digest_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence" + str(len(value)).encode() + b":")
        for item in value:
            _digest_value(digest, item)
    elif value is None or isinstance(value, (str, bytes, bool, int, float, np.generic)):
        made = (type(value).__name__ + ":" + repr(value)).encode("utf-8")
        digest.update(str(len(made)).encode() + b":" + made)
    else:
        raise TypeError(f"unowned snapshot digest value: {type(value).__name__}")


def capture_source_witness(model):
    """Internal preparation-only coherence witness for application adapters."""
    from .static_batch import _model_state
    return _model_state(model)


def capture_prepared_static_snapshot(
    model: "FEModel",
    *,
    recovery_config: "RecoveryConfig",
    envelope_recovery_config: Any = None,
    load_families: Optional[Sequence[str]],
    cancellation_token: Any = None,
    progress_callback: Any = None,
    _resource_config: Any = None,
) -> PreparedStaticSnapshot:
    """Capture the complete immutable snapshot from the live model."""

    from .assembly import (
        build_constraint_transformation,
        build_reduced_rigid_body_modes,
    )
    from .matrix_assembly import assemble_stiffness_matrix
    from .recovery import estimate_model_memory

    from .control import cancellation_safe_point, emit_progress

    started = time.perf_counter()
    cancellation_safe_point(cancellation_token, "prepared_static.prepare.start")
    model.apply_boundary_conditions()
    witness_before = capture_source_witness(model)
    def checkpoint(stage, completed):
        # Caller observations occur only between completed canonical operations.
        cancellation_safe_point(cancellation_token, stage)
        emit_progress(progress_callback, "prepared_static_preparation", stage,
                      completed=completed, total=5, suppress_callback_errors=False)
        cancellation_safe_point(cancellation_token, stage)
        if (progress_callback is not None or cancellation_token is not None) and capture_source_witness(model) != witness_before:
            raise RuntimeError("the model changed during prepared static snapshot capture; refusing a mixed snapshot")

    checkpoint("prepared_static.prepare.admitted", 0)
    revision_before = {
        str(category): int(value)
        for category, value in model.revision_signature().items()
    }
    phases = {"admission_and_witness": time.perf_counter() - started}
    phase_started = time.perf_counter()
    mesh = model.mesh
    total_dofs = int(mesh.dof_manager.total_dofs)

    stiffness, _stiffness_info = assemble_stiffness_matrix(model)
    K = _immutable_csr(stiffness)
    checkpoint("prepared_static.prepare.stiffness", 1)
    phases["stiffness"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()

    zero = np.zeros(total_dofs, dtype=float)
    K_red, _F_red, T, u0, independent, constraint_info = build_constraint_transformation(
        K, zero, model
    )
    T = _immutable_csr(T)
    u0 = _immutable_array(np.asarray(u0, dtype=float).reshape(-1))
    independent = _immutable_array(np.asarray(independent, dtype=np.intp).reshape(-1))
    K_red = _immutable_csr(K_red)

    rigid_modes, nullspace_info = build_reduced_rigid_body_modes(
        model, independent, total_dofs, transformation=T
    )
    rigid_modes = _immutable_array(np.asarray(rigid_modes, dtype=float))

    selected_node_ids = tuple(int(node_id) for node_id in recovery_config.selected_node_ids(model))
    selected_element_ids = tuple(
        int(element_id) for element_id in recovery_config.selected_element_ids(model)
    )

    checkpoint("prepared_static.prepare.constraints", 2)
    phases["constraints_and_modes"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    load_maps = compile_canonical_load_maps(model, families=load_families)
    checkpoint("prepared_static.prepare.loads", 3)
    phases["load_maps"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    reaction_maps = capture_reaction_maps(model)
    phases['reaction_map_capture'] = time.perf_counter()-phase_started
    recovery_compilation = None
    recovery_started = time.perf_counter()
    recovery_timings = {}
    if recovery_config.include_stresses and selected_element_ids:
        recovery_compilation = compile_affine_recovery(
            model, recovery_config, selected_element_ids,_preparation_timings=recovery_timings,
            _resource_config=_resource_config
        )
    phases['affine_recovery_compile'] = time.perf_counter()-recovery_started
    phases.update({'affine_recovery.'+(name[:-3]+'_seconds' if name.endswith('_ns') else name):
                   float(value)/1e9 if name.endswith('_ns') else float(value)
                   for name,value in recovery_timings.items()})

    from .prepared_static_envelopes import compile_envelope_plan
    envelope_started = time.perf_counter()
    envelope_plan = compile_envelope_plan(model, envelope_recovery_config)
    phases['envelope_plan_capture'] = time.perf_counter()-envelope_started

    checkpoint("prepared_static.prepare.recovery", 4)
    phases["recovery_and_reactions"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    revision_after = {
        str(category): int(value)
        for category, value in model.revision_signature().items()
    }
    if revision_after != revision_before or capture_source_witness(model) != witness_before:
        raise RuntimeError(
            "the model changed during prepared static snapshot capture; "
            "refusing a mixed snapshot"
        )

    estimate = estimate_model_memory(model, num_rhs=0, recovery_config=recovery_config)
    memory_estimate = _plain_copy(estimate.to_dict())

    checkpoint("prepared_static.prepare.complete", 5)
    phases["coherence_and_memory"] = time.perf_counter() - phase_started
    snapshot = PreparedStaticSnapshot(
        total_dofs=total_dofs,
        num_nodes=int(mesh.num_nodes),
        num_elements=int(mesh.num_elements),
        stiffness=K,
        transformation=T,
        prescribed_offset=u0,
        independent_dofs=independent,
        reduced_stiffness=K_red,
        constraint_info=MappingProxyType(_plain_copy(dict(constraint_info))),
        rigid_modes=rigid_modes,
        nullspace_info=MappingProxyType(_plain_copy(dict(nullspace_info))),
        load_maps=load_maps,
        reaction_maps=reaction_maps,
        recovery_compilation=recovery_compilation,
        selected_node_ids=selected_node_ids,
        selected_element_ids=selected_element_ids,
        memory_estimate=MappingProxyType(memory_estimate),
        retained_bytes=int(reaction_maps.retained_bytes),
        setup_seconds=float(time.perf_counter() - started),
        content_digest="",
        source_revision=MappingProxyType(dict(revision_before)),
        phase_timings=MappingProxyType(phases),
        envelope_plan=envelope_plan,
    )
    digest = snapshot.content_digest_hex()
    import dataclasses

    return dataclasses.replace(snapshot, content_digest=digest)
