from __future__ import annotations

from typing import Any

import pytest

import anyfesolver.matrix_assembly as matrix_assembly
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.matrix_assembly import AssemblyError


def _install_rejecting_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    def capture(
        _model: Any,
        *,
        context: str,
        allow_q4_cached_stiffness: bool = False,
    ):
        del context
        assert not allow_q4_cached_stiffness

        def reject(
            _same_model: Any,
            *,
            context: str,
            final: bool = False,
        ) -> None:
            assert final
            raise AssemblyError(f"{context} rejected changed authority")

        return reject

    monkeypatch.setattr(
        matrix_assembly,
        "_CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE",
        capture,
    )


def test_precise_capability_error_survives_exceptional_lease_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_rejecting_lease(monkeypatch)
    expected = ElementCapabilityError("NUMERICAL_AUTHORITY_MISMATCH")

    def operation(_lease: Any) -> Any:
        raise expected

    with pytest.raises(ElementCapabilityError) as caught:
        matrix_assembly._run_with_qualified_assembly_runtime_lease(
            object(),
            context="precise classification",
            operation=operation,
        )
    assert caught.value is expected
    assert "NUMERICAL_AUTHORITY_MISMATCH" in str(caught.value)


def test_arbitrary_exception_is_replaced_when_exceptional_lease_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_rejecting_lease(monkeypatch)

    def operation(_lease: Any) -> Any:
        raise RuntimeError("arbitrary failure")

    with pytest.raises(AssemblyError, match="rejected changed authority") as caught:
        matrix_assembly._run_with_qualified_assembly_runtime_lease(
            object(),
            context="arbitrary classification",
            operation=operation,
        )
    assert isinstance(caught.value.__cause__, RuntimeError)
