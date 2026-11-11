"""Regression coverage for spectral cancellation checkpoint guard routing."""

from __future__ import annotations

import pytest

import anyfesolver.buckling as buckling_module
import anyfesolver.modal as modal_module
from anyfesolver.boundary import BoundaryCondition, FixedSupport
from anyfesolver.control import CancellationToken
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel


def _modal_model() -> FEModel:
    model = FEModel("spectral_modal_checkpoint")
    model.add_material("steel", elastic_modulus=100.0, poisson_ratio=0.3, density=2.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(
        1,
        BeamElement(
            1,
            [1, 2],
            "steel",
            {"area": 1.0, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
        ),
    )
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(
        BoundaryCondition(
            "slider",
            [2],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    return model


def _buckling_model() -> tuple[FEModel, dict[int, dict[str, float]]]:
    model = FEModel("spectral_buckling_checkpoint")
    model.add_material("steel", elastic_modulus=210.0e9, poisson_ratio=0.3, density=7850.0)
    for node_id, x in enumerate((0.0, 2.0, 4.0), start=1):
        model.add_node(node_id, x, 0.0, 0.0)
    section = {"area": 0.02, "Iy": 3.0e-6, "Iz": 5.0e-6, "J": 2.0e-6}
    for element_id in (1, 2):
        model.add_element(
            element_id,
            BeamElement(element_id, [element_id, element_id + 1], "steel", section),
        )
    model.add_boundary_condition(
        BoundaryCondition(
            "suppress_unrelated_dofs",
            [1, 2, 3],
            {"ux": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0},
        )
    )
    model.add_boundary_condition(
        BoundaryCondition("pinned_lateral_ends", [1, 3], {"uy": 0.0})
    )
    model.apply_boundary_conditions()
    return model, {1: {"axial_compression": 1.0}, 2: {"axial_compression": 1.0}}


class _CustomCancellationToken(CancellationToken):
    """A caller-owned token subclass must not be treated as the no-op path."""


@pytest.mark.parametrize(
    ("module", "solve", "build", "stage_prefix"),
    [
        (
            modal_module,
            lambda model, states, token=None: modal_module.solve_free_vibration(
                model, num_modes=1, cancellation_token=token
            ),
            lambda: (_modal_model(), None),
            "modal",
        ),
        (
            buckling_module,
            lambda model, states, token=None: buckling_module.solve_eigenvalue_buckling(
                model, states, num_modes=1, cancellation_token=token
            ),
            _buckling_model,
            "buckling",
        ),
    ],
    ids=["modal", "buckling"],
)
def test_spectral_checkpoint_callback_aba_and_tokens_use_full_guards(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    solve: object,
    build: object,
    stage_prefix: str,
) -> None:
    """Only the captured original callback plus ``None`` may use trusted guards."""

    def trusted_runtime_guard(observed_model: FEModel, *, context: str) -> None:
        return None

    def runtime_guard(observed_model: FEModel, *, context: str) -> None:
        return None

    runtime_guard._qualified_trusted_require = trusted_runtime_guard  # type: ignore[attr-defined]

    def run_under_test_lease(
        observed_model: FEModel,
        *,
        context: str,
        operation: object,
    ) -> object:
        return operation(runtime_guard)  # type: ignore[operator]

    # The production lease supplies this trusted checkpoint; making it explicit
    # here lets the result diagnostics prove which guard path was selected.
    monkeypatch.setattr(module, "_run_with_qualified_assembly_runtime_lease", run_under_test_lease)

    model, states = build()  # type: ignore[operator]
    baseline = solve(model, states)  # type: ignore[operator]
    baseline_counts = baseline.diagnostics["spectral_guard_diagnostics"]

    seen_stages: list[str] = []
    original_callback = module._EXACT_CANCELLATION_SAFE_POINT

    def callback_that_restores_original(token: object, stage: str) -> None:
        seen_stages.append(stage)
        # An attacker can mutate the module alias before restoring the public
        # callback.  The solve must use its definition-time identity instead.
        monkeypatch.setattr(module, "_EXACT_CANCELLATION_SAFE_POINT", callback_that_restores_original)
        monkeypatch.setattr(module, "cancellation_safe_point", original_callback)

    monkeypatch.setattr(module, "cancellation_safe_point", callback_that_restores_original)
    model, states = build()  # type: ignore[operator]
    aba_result = solve(model, states)  # type: ignore[operator]
    aba_counts = aba_result.diagnostics["spectral_guard_diagnostics"]

    # The first captured callback was replaced even though it restored the
    # original immediately afterwards, so its following guard remains full.
    assert seen_stages == [f"{stage_prefix}.start"]
    assert aba_counts["full_guard_count"] == baseline_counts["full_guard_count"] + 1
    assert aba_counts["trusted_guard_count"] == baseline_counts["trusted_guard_count"] - 1

    monkeypatch.setattr(module, "_EXACT_CANCELLATION_SAFE_POINT", original_callback)
    model, states = build()  # type: ignore[operator]
    token_result = solve(model, states, _CustomCancellationToken())  # type: ignore[operator]
    token_counts = token_result.diagnostics["spectral_guard_diagnostics"]
    assert token_counts["full_guard_count"] > baseline_counts["full_guard_count"]
    assert token_counts["trusted_guard_count"] < baseline_counts["trusted_guard_count"]
