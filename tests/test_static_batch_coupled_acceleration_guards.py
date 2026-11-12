"""Trusted acceleration admission for exact beam-shell coupling elements.

``_run_foreign_code_free_load_vector`` admits an untouched gravity load case to
the lease's constant-time trusted span only when every element is an exact
package class with untouched authority and plain constructed state.  These
tests pin the new ``CoupledBeamShellElement`` admission: the admitted span is
bit-identical to the public kernel (whose arithmetic is unchanged from the
8cb0de07 baseline, so equality with it is equality with the baseline), the
coupling element contributes exactly zero gravity load, the complete
lifecycle guard amortizes to one call per lease, and every adversarial
subclass/class-patch/descriptor/instance-shadow/foreign-state/foreign-provider
input is refused back to the fully guarded public path.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pytest

import anyfesolver.current_state_tangent as current_state_tangent
from anyfesolver import boundary
from anyfesolver.boundary import LoadCase
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.elements import BeamElement, CoupledBeamShellElement
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.fe_core import FEModel, FEMesh
from anyfesolver.matrix_assembly import assemble_load_matrix, assemble_load_vector

_GUARD_ERRORS = (Exception,)


def _model(with_coupled: bool = True) -> FEModel:
    """Eccentric beam over a shell patch, joined by one coupling element."""

    model = FEModel("coupled-acceleration-guards")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.5)
    model.add_node(2, 1.0, 0.0, 0.5)
    model.add_node(3, 0.0, 0.0, 0.0)
    model.add_node(4, 1.0, 0.0, 0.0)
    model.add_node(5, 0.5, 1.0, 0.0)
    beam = BeamElement(
        1, [1, 2], "steel", cross_section={"area": 0.004, "Iy": 1e-5, "Iz": 1e-5, "J": 1e-5}
    )
    shell = QualifiedE4PLS3ShellElement(
        2, [3, 4, 5], "steel", thickness=0.01, reference_normal=(0.0, 0.0, 1.0)
    )
    model.add_element(beam.element_id, beam)
    model.add_element(shell.element_id, shell)
    if with_coupled:
        coupled = CoupledBeamShellElement(
            3, 1, 3, "steel", coupling_stiffness=0.0,
            eccentricity=np.array([0.0, 0.0, 0.5]),
        )
        model.add_element(coupled.element_id, coupled)
    return model


def _gravity(name: str = "gravity") -> LoadCase:
    case = LoadCase(name)
    case.set_gravity(0.0, 0.0, -9.81)
    return case


def _owned(model: FEModel, case: LoadCase) -> Optional[np.ndarray]:
    return boundary._run_foreign_code_free_load_vector(
        case,
        model.mesh,
        model.mesh.dof_manager,
        model.get_material,
        None,
        None,
        stage_guard=lambda *, stage: None,
    )


def _public(model: FEModel, case: LoadCase) -> np.ndarray:
    return np.asarray(
        case.get_load_vector(
            model.mesh, model.mesh.dof_manager, model.get_material
        ),
        dtype=float,
    )


def _count_lifecycle_guard(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    original = current_state_tangent.require_exact_qualified_component_lifecycle_api

    def counting(model: Any, *, context: str) -> Any:
        calls.append(context)
        return original(model, context=context)

    monkeypatch.setattr(
        current_state_tangent,
        "require_exact_qualified_component_lifecycle_api",
        counting,
    )
    return calls


def test_exact_coupled_gravity_is_admitted_and_bit_identical() -> None:
    model = _model()
    case = _gravity()
    case.add_nodal_load(4, forces=[1.0e3, -2.0e3, 3.0e3])
    owned = _owned(model, case)
    assert owned is not None
    public = _public(model, case)
    assert owned.tobytes() == public.tobytes()
    assembled, info = assemble_load_vector(model, case)
    assert assembled.tobytes() == owned.tobytes()
    assert info["load_case"] == case.name
    batch, batch_info = assemble_load_matrix(model, [case, _gravity("gravity-b")])
    assert batch[:, 0].tobytes() == assembled.tobytes()
    assert batch_info["load_cases"] == ["gravity", "gravity-b"]


def test_coupled_element_contributes_exactly_zero_gravity_load() -> None:
    """Zero coupling mass means the baseline (8cb0de07) vector is unchanged.

    The public kernel's arithmetic is untouched by the admission change, and a
    coupling element's mass matrix is exactly zero, so the gravity load vector
    of the model with the coupling element must be bit-identical to the same
    model without it, on both the trusted span and the public path.
    """

    with_coupled = _model(with_coupled=True)
    without_coupled = _model(with_coupled=False)
    case = _gravity()
    coupled = with_coupled.mesh.elements[3]
    mass = coupled.compute_mass_matrix(with_coupled.mesh, with_coupled.get_material("steel"))
    assert mass.tobytes() == np.zeros((mass.shape[0], mass.shape[1]), dtype=float).tobytes()
    for model in (with_coupled, without_coupled):
        owned = _owned(model, case)
        assert owned is not None
        assert owned.tobytes() == _public(model, case).tobytes()
    assert _owned(with_coupled, case).tobytes() == _owned(without_coupled, case).tobytes()
    assert _public(with_coupled, case).tobytes() == _public(without_coupled, case).tobytes()


def test_coupled_gravity_batch_amortizes_guard_once_per_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    cases = [_gravity("gravity-a"), _gravity("gravity-b"), _gravity("gravity-c")]
    calls = _count_lifecycle_guard(monkeypatch)
    _, info = assemble_load_matrix(model, cases)
    assert info["load_cases"] == [case.name for case in cases]
    assert len(calls) == 1, calls
    calls.clear()
    assemble_load_vector(model, cases[0])
    assert len(calls) == 1, calls


def test_coupled_subclass_refuses_trusted_span_and_keeps_complete_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Observed(CoupledBeamShellElement):
        pass

    model = _model()
    coupled = Observed(3, 1, 3, "steel", eccentricity=np.array([0.0, 0.0, 0.5]))
    model.add_element(3, coupled)
    case = _gravity()
    assert _owned(model, case) is None
    calls = _count_lifecycle_guard(monkeypatch)
    assembled, _ = assemble_load_vector(model, case)
    assert len(calls) >= 3, calls
    assert assembled.tobytes() == _public(model, case).tobytes()


@pytest.mark.parametrize(
    "patch",
    [
        pytest.param(
            lambda monkeypatch: monkeypatch.setattr(
                CoupledBeamShellElement,
                "compute_mass_matrix",
                lambda self, mesh, material: np.eye(12),
            ),
            id="replaced-mass-method",
        ),
        pytest.param(
            lambda monkeypatch: monkeypatch.setattr(
                CoupledBeamShellElement,
                "beam_offset",
                property(lambda self: 1.0),
                raising=False,
            ),
            id="added-descriptor",
        ),
        pytest.param(
            lambda monkeypatch: monkeypatch.setattr(
                CoupledBeamShellElement, "__mro_entries__", lambda *a: (object(),),
                raising=False,
            ),
            id="patched-dunder",
        ),
    ],
)
def test_coupled_class_patch_or_descriptor_refuses_trusted_span(
    patch: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model()
    case = _gravity()
    patch(monkeypatch)
    assert _owned(model, case) is None
    with pytest.raises(ElementCapabilityError):
        assemble_load_vector(model, case)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda element: setattr(
                element, "compute_mass_matrix", lambda mesh, material: np.eye(12)
            ),
            id="instance-method-shadow",
        ),
        pytest.param(
            lambda element: setattr(element, "eccentricity", [0.0, 0.0, 0.5]),
            id="list-eccentricity",
        ),
        pytest.param(
            lambda element: setattr(element, "coupling_stiffness", np.float64(1.0)),
            id="numpy-scalar-stiffness",
        ),
        pytest.param(
            lambda element: setattr(element, "beam_node_id", np.int64(1)),
            id="numpy-node-id",
        ),
        pytest.param(
            lambda element: setattr(
                element, "eccentricity", np.zeros(3, dtype=np.complex128)
            ),
            id="complex-eccentricity",
        ),
    ],
)
def test_coupled_instance_shadow_or_foreign_state_refuses_trusted_span(
    mutate: Any,
) -> None:
    model = _model()
    case = _gravity()
    mutate(model.mesh.elements[3])
    assert _owned(model, case) is None


def test_foreign_state_keeps_complete_guards_and_correct_result() -> None:
    model = _model()
    case = _gravity()
    expected = _public(model, case).tobytes()
    model.mesh.elements[3].coupling_stiffness = np.float64(2.0)
    assert _owned(model, case) is None
    assembled, _ = assemble_load_vector(model, case)
    assert assembled.tobytes() == expected


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda model, monkeypatch: monkeypatch.setattr(
                model, "get_material", lambda name: model.materials["steel"]
            ),
            id="foreign-material-getter",
        ),
        pytest.param(
            lambda model, monkeypatch: monkeypatch.setattr(
                FEMesh, "get_node", lambda self, node_id: None
            ),
            id="patched-mesh-provider",
        ),
        pytest.param(
            lambda model, monkeypatch: monkeypatch.setattr(
                model.mesh, "get_node", lambda node_id: None
            ),
            id="instance-mesh-provider",
        ),
    ],
)
def test_foreign_provider_refuses_trusted_span(
    mutate: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fresh = _model()
    case = _gravity()
    mutate(fresh, monkeypatch)
    assert _owned(fresh, case) is None
