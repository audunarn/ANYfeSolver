"""Deep-copy regressions for runtime-authority-bound solver objects."""

from __future__ import annotations

import copy
import inspect
from typing import Any, Dict, Mapping

import numpy as np
import pytest

from anyfesolver import FEModel
from anyfesolver.elements import BeamElement, Element, ShellElement
from anyfesolver.imperfections import ImperfectionField, apply_imperfection
from anyfesolver.e4_pl_s3_state import resolved_material_descriptor
from anyfesolver.nonlinear_restart import (
    analysis_configuration_fingerprint,
    canonical_checkpoint_json_bytes,
    load_nonlinear_checkpoint,
    model_configuration_descriptor,
    model_configuration_fingerprint,
)
from anyfesolver.shell_sections import GeneralizedShellSection


class _SlottedBeamElement(BeamElement):
    __slots__ = ("slot_state",)


def _section() -> GeneralizedShellSection:
    return GeneralizedShellSection(
        A=np.diag([10.0, 8.0, 3.0]),
        B=np.zeros((3, 3), dtype=float),
        D=np.diag([2.0, 1.5, 0.75]),
        As=np.diag([4.0, 3.0]),
        name="deep-copy-section",
        mass_per_area=12.0,
        rotary_inertia_per_area=0.04,
    )


def _namespace_snapshot(cls: type[Any]) -> dict[str, Any]:
    return dict(type.__getattribute__(cls, "__dict__"))


def _assert_namespace_unchanged(
    cls: type[Any], before: dict[str, Any]
) -> None:
    after = type.__getattribute__(cls, "__dict__")
    assert set(after) == set(before)
    assert all(after[name] is value for name, value in before.items())
    assert "__slotnames__" not in after


def test_element_deepcopy_preserves_legacy_state_aliases_slots_and_memo() -> None:
    shared = {"history": [1.0, 2.0]}
    beam = _SlottedBeamElement(
        7,
        [1, 2],
        "steel",
        {"area": 0.02, "Iy": 1.0e-5, "Iz": 2.0e-5, "J": 3.0e-5},
    )
    beam.cross_section["shared"] = shared
    beam.shared_state = shared
    beam.slot_state = shared
    beam._stiffness_matrix = np.arange(144, dtype=float).reshape(12, 12)

    namespaces = {
        cls: _namespace_snapshot(cls)
        for cls in (Element, BeamElement, _SlottedBeamElement)
    }
    made = copy.deepcopy(beam)

    assert type(made) is _SlottedBeamElement
    assert made is not beam
    assert set(vars(made)) == set(vars(beam))
    assert made.element_id == beam.element_id
    assert made.node_ids == beam.node_ids
    assert made.node_ids is not beam.node_ids
    assert made._stiffness_matrix is not beam._stiffness_matrix
    np.testing.assert_array_equal(made._stiffness_matrix, beam._stiffness_matrix)
    assert made.shared_state is made.slot_state
    assert made.shared_state is made.cross_section["shared"]
    assert made.shared_state is not shared

    sentinel = object()
    assert copy.deepcopy(beam, {id(beam): sentinel}) is sentinel
    for cls, before in namespaces.items():
        _assert_namespace_unchanged(cls, before)


def test_generalized_section_deepcopy_is_equivalent_frozen_and_alias_safe() -> None:
    section = _section()
    before = _namespace_snapshot(GeneralizedShellSection)

    first, second = copy.deepcopy([section, section])

    assert type(first) is GeneralizedShellSection
    assert first is second
    assert first is not section
    assert set(vars(first)) == set(vars(section))
    assert first.to_dict() == section.to_dict()
    for name in ("A", "B", "D", "As"):
        original = object.__getattribute__(section, name)
        copied = object.__getattribute__(first, name)
        assert copied is not original
        np.testing.assert_array_equal(copied, original)
        assert not copied.flags.writeable
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
            copied.setflags(write=True)

    sentinel = object()
    assert copy.deepcopy(section, {id(section): sentinel}) is sentinel
    _assert_namespace_unchanged(GeneralizedShellSection, before)


def test_mixed_legacy_element_and_section_copy_has_no_class_namespace_delta() -> None:
    beam = BeamElement(
        3,
        [4, 5],
        "steel",
        {"area": 0.01, "Iy": 1.0e-6, "Iz": 2.0e-6, "J": 3.0e-6},
    )
    section = _section()
    beam.section_metadata = {"shell": section, "again": section}
    namespaces = {
        cls: _namespace_snapshot(cls)
        for cls in (Element, BeamElement, GeneralizedShellSection)
    }

    made_beam, made_section = copy.deepcopy((beam, section))

    assert type(made_beam) is BeamElement
    assert made_beam.section_metadata["shell"] is made_section
    assert made_beam.section_metadata["again"] is made_section
    for cls, before in namespaces.items():
        _assert_namespace_unchanged(cls, before)


def test_apply_imperfection_copies_legacy_element_and_section_without_class_delta() -> None:
    model = FEModel("namespace-neutral-imperfection")
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 2.0, 0.0, 0.0)
    beam = BeamElement(
        1,
        [1, 2],
        "default",
        {"area": 0.01, "Iy": 1.0e-6, "Iz": 2.0e-6, "J": 3.0e-6},
    )
    section = _section()
    beam.section_metadata = {"primary": section, "alias": section}
    model.add_element(1, beam)
    namespaces = {
        cls: _namespace_snapshot(cls)
        for cls in (Element, BeamElement, GeneralizedShellSection)
    }

    made = apply_imperfection(
        model,
        ImperfectionField({2: (0.0, 0.0, 0.1)}),
        copy_model=True,
    )

    assert made is not model
    assert model.mesh.get_node(2).z == 0.0
    assert made.mesh.get_node(2).z == pytest.approx(0.1)
    made_beam = made.mesh.get_element(1)
    assert type(made_beam) is BeamElement
    assert made_beam.section_metadata["primary"] is made_beam.section_metadata["alias"]
    assert made_beam.section_metadata["primary"] is not section
    assert not made_beam.section_metadata["primary"].A.flags.writeable
    for cls, before in namespaces.items():
        _assert_namespace_unchanged(cls, before)


def test_shell_element_private_guard_hooks_do_not_change_public_signatures() -> None:
    mass_signature = inspect.signature(ShellElement.compute_mass_matrix)
    nonlinear_signature = inspect.signature(ShellElement.compute_nonlinear_response)

    assert mass_signature == inspect.signature(Element.compute_mass_matrix)
    assert nonlinear_signature == inspect.signature(Element.compute_nonlinear_response)
    assert all(
        not name.startswith("_qualified") and name != "_return_tangent_components"
        for name in (*mass_signature.parameters, *nonlinear_signature.parameters)
    )


def _head_resolved_material_descriptor(material: Any) -> dict[str, Any]:
    raise NotImplementedError


def _head_canonical_checkpoint_json_bytes(value: Any) -> bytes:
    raise NotImplementedError


def _head_model_configuration_descriptor(model: Any) -> Dict[str, Any]:
    raise NotImplementedError


def _head_model_configuration_fingerprint(model: Any) -> str:
    raise NotImplementedError


def _head_analysis_configuration_fingerprint(
    contract: Mapping[str, Any],
) -> str:
    raise NotImplementedError


def _head_load_nonlinear_checkpoint(value: Any) -> Dict[str, Any]:
    raise NotImplementedError


def test_private_authority_hooks_preserve_public_helper_signatures() -> None:
    expected = (
        (resolved_material_descriptor, _head_resolved_material_descriptor),
        (
            canonical_checkpoint_json_bytes,
            _head_canonical_checkpoint_json_bytes,
        ),
        (
            model_configuration_descriptor,
            _head_model_configuration_descriptor,
        ),
        (
            model_configuration_fingerprint,
            _head_model_configuration_fingerprint,
        ),
        (
            analysis_configuration_fingerprint,
            _head_analysis_configuration_fingerprint,
        ),
        (load_nonlinear_checkpoint, _head_load_nonlinear_checkpoint),
    )
    for current, historical in expected:
        assert inspect.signature(current) == inspect.signature(historical)
        assert all(
            not name.startswith("_")
            for name in inspect.signature(current).parameters
        )
