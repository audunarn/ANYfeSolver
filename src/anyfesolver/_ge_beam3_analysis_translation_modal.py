"""Model-owned conservative modes of an actual accepted translation state.

Dispatches to each native state owner without converting checkpoints. The
physical inertia and signed factor pencil are mandatory; the controller is
not a physical support. Existing modal size and material bounds are unchanged.
"""
from dataclasses import dataclass, field
from math import isfinite

from ._ge_beam3_native_analysis import _require
from ._ge_beam3_analysis_translation import _owner, _operation, _backend, _check_backend_header, seeded
from ._native_paired_factor_chain_modes import solve_paired_factor_chain_modes
from ._native_modal_capacity import POLICY as LARGE_POLICY,solve_large_factor_chain_modes


@dataclass(frozen=True)
class NativeTranslationModes:
    definition_graph_sha256: str
    checkpoint_sha256: str
    packet: object
    modes: object
    production_qualified: bool = field(default=False, init=False)


@dataclass(frozen=True)
class NativeLargeTranslationModes(NativeTranslationModes):
    snapshot_sha256: str
    modal_capacity_policy: str=field(default=LARGE_POLICY,init=False)


def _controls(analysis, bounds, count, root_width, relative_width, modal_policy=None):
    _require(type(bounds) is tuple and len(bounds) == 2
        and all(type(x) in (float,int) and isfinite(x) for x in bounds) and bounds[0] < bounds[1]
        and type(count) is int and count > 0
        and type(root_width) is float and isfinite(root_width) and root_width > 0.
        and type(relative_width) is float and isfinite(relative_width) and 0. < relative_width <= 1e-10,
        'explicit finite physical modal controls required')
    # Reject unsupported dimensions before replay or factor construction.
    elements = len(analysis._elements)
    size = analysis.model.mesh.dof_manager.total_dofs+6*elements
    _require(modal_policy is None or type(modal_policy) is str and modal_policy==LARGE_POLICY,
        'registered modal capacity policy required')
    if modal_policy is not None:
        _require(getattr(analysis,'_retained_refinement',False) is True,
            'large modal policy requires explicit retained refinement ownership')
    coordinates,links=(256,512) if modal_policy is None else (640,640)
    _require(size <= coordinates and 18*elements <= links, 'accepted-state modal solver coordinate bound exceeded')


def solve(analysis, program, checkpoint, *, expected_sha256, bounds, num_modes=6,
          root_width=1e-10, relative_width=1e-12, seed=None, expected_seed_sha256=None,
          cancellation_token=None, compliance_guard_policy=None, modal_policy=None):
    owner, workflow, seed_digest = _owner(analysis, program, seed, expected_seed_sha256)
    _controls(analysis, bounds, num_modes, root_width, relative_width, modal_policy)
    if owner is seeded:
        from . import _ge_beam3_elastic_seed_modal as modal
    else:
        from . import _ge_beam3_retained_translation_modal as modal
        _require(compliance_guard_policy is None, 'compliance snapshot policy belongs to the elastic-seed owner')
    with _operation(analysis, cancellation_token):
        backend, digest = _backend(analysis, program, checkpoint, expected_sha256, workflow, seed_digest)
        _check_backend_header(backend, digest, program, owner, seed_digest)
        if owner is seeded:
            packet, guard = modal.prepare(analysis.model, program, seed, backend, analysis._inertias,
                expected_sha256=digest, expected_seed_sha256=seed_digest,
                cancellation_token=cancellation_token, compliance_guard_policy=compliance_guard_policy)
        else:
            packet, guard = modal.prepare(analysis.model, program, backend, analysis._inertias,
                expected_sha256=digest, cancellation_token=cancellation_token)
        solve_modes=solve_paired_factor_chain_modes
        if modal_policy is not None:
            from ._ge_beam3_analysis_modal_snapshot import seal
            snapshot,guard=seal(analysis,program,packet,guard,cancellation_token)
            solve_modes=solve_large_factor_chain_modes
        try:
            modes = solve_modes(packet.left, packet.right, packet.geometric, packet.kinetic,
                packet.free_dofs, packet.algebraic_dofs, bounds=bounds, num_modes=num_modes,
                root_width=root_width, relative_width=relative_width, cancellation_token=cancellation_token)
        finally:
            if modal_policy is not None:guard()
        if modal_policy is not None:
            return NativeLargeTranslationModes(analysis.identity,expected_sha256,packet,modes,snapshot)
        guard()
        return NativeTranslationModes(analysis.identity, expected_sha256, packet, modes)
