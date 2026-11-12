from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from anyfesolver import (
    FEModel,
    QualifiedE4PLS3ShellElement,
    QualifiedE4PLShellElement,
)
import anyfesolver.e4_pl_element as q4_module
import anyfesolver.e4_pl_s3_element as s3_module


def _q4_case() -> tuple[FEModel, QualifiedE4PLShellElement, Any]:
    model = FEModel("q4-component-snapshot-integrity")
    for node_id, point in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.2, 0.1, 0.0),
            (1.0, 1.1, 0.0),
            (-0.1, 0.9, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *point)
    material = model.add_material(
        "steel",
        210.0e9,
        0.3,
        density=7850.0,
    )
    element = QualifiedE4PLShellElement(
        1,
        (1, 2, 3, 4),
        "steel",
        thickness=0.02,
        reference_normal=(0.0, 0.0, 1.0),
    )
    model.add_element(1, element)
    element.compute_stiffness_matrix(model.mesh, material)
    return model, element, material


def _s3_case() -> tuple[FEModel, QualifiedE4PLS3ShellElement, Any]:
    model = FEModel("s3-component-snapshot-integrity")
    for node_id, point in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.1, 0.05, 0.0),
            (0.42, 0.91, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *point)
    material = model.add_material(
        "steel",
        210.0e9,
        0.3,
        density=7850.0,
    )
    element = QualifiedE4PLS3ShellElement(
        1,
        (1, 2, 3),
        "steel",
        thickness=0.02,
        reference_normal=(0.0, 0.0, 1.0),
    )
    model.add_element(1, element)
    element.compute_stiffness_matrix(model.mesh, material)
    return model, element, material


def _snapshot_trace_lines() -> tuple[int, int]:
    lines = Path(q4_module.__file__).read_text(encoding="utf-8").splitlines()
    raw_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if 'raw = memoryview(value).cast("B").tobytes()' in line
    )
    length_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if index > raw_line and "if len(raw) != 24 * 24 * 8:" in line
    )
    return raw_line, length_line


def _s3_snapshot_trace_lines() -> tuple[int, int]:
    lines = Path(s3_module.__file__).read_text(encoding="utf-8").splitlines()
    raw_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if 'raw = memoryview(value).cast("B").tobytes()' in line
    )
    length_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if index > raw_line and "if len(raw) != expected_bytes:" in line
    )
    return raw_line, length_line


def test_q4_numerical_force_uses_operation_local_raw_byte_snapshots() -> None:
    _model, element, _material = _q4_case()
    displacement = np.linspace(-2.0e-4, 3.0e-4, 24, dtype=np.float64)
    expected = {
        name: value.copy()
        for name, value in element.numerical_internal_force(displacement).items()
    }
    components = object.__getattribute__(element, "__dict__")[
        "_qualified_components"
    ]
    pl = components["pl"]
    assert type(pl) is np.ndarray
    assert pl.dtype == np.dtype(np.float64)
    raw_line, length_line = _snapshot_trace_lines()
    changed = False

    def trace(frame: Any, event: str, _arg: Any):
        nonlocal changed
        if frame.f_code.co_name != "numerical_internal_force" or event != "line":
            return trace
        if frame.f_lineno == raw_line and frame.f_locals.get("name") == "pl":
            pl.dtype = np.int64
            changed = True
        elif changed and frame.f_lineno == length_line:
            pl.dtype = np.float64
        return trace

    sys.settrace(trace)
    try:
        actual = element.numerical_internal_force(displacement)
    finally:
        sys.settrace(None)
        pl.dtype = np.float64

    assert changed
    for name in ("pl", "hourglass", "numerical"):
        assert np.array_equal(actual[name], expected[name])


def test_q4_numerical_force_rejects_persistent_component_metadata_change() -> None:
    _model, element, _material = _q4_case()
    displacement = np.linspace(-2.0e-4, 3.0e-4, 24, dtype=np.float64)
    expected = {
        name: value.copy()
        for name, value in element.numerical_internal_force(displacement).items()
    }
    components = object.__getattribute__(element, "__dict__")[
        "_qualified_components"
    ]
    pl = components["pl"]
    raw_line, _length_line = _snapshot_trace_lines()
    changed = False

    def trace(frame: Any, event: str, _arg: Any):
        nonlocal changed
        if (
            not changed
            and frame.f_code.co_name == "numerical_internal_force"
            and event == "line"
            and frame.f_lineno == raw_line
            and frame.f_locals.get("name") == "pl"
        ):
            pl.dtype = np.int64
            changed = True
        return trace

    sys.settrace(trace)
    try:
        with pytest.raises(ValueError, match="component cache authority"):
            element.numerical_internal_force(displacement)
    finally:
        sys.settrace(None)
        pl.dtype = np.float64

    assert changed
    clean = element.numerical_internal_force(displacement)
    for name in ("pl", "hourglass", "numerical"):
        assert np.array_equal(clean[name], expected[name])


def test_s3_numerical_force_uses_operation_local_raw_byte_snapshot() -> None:
    _model, element, _material = _s3_case()
    displacement = np.linspace(-2.0e-4, 3.0e-4, 18, dtype=np.float64)
    expected = element.numerical_internal_force(displacement)["pl"].copy()
    components = object.__getattribute__(element, "__dict__")[
        "_qualified_components"
    ]
    pl = components["pl"]
    raw_line, length_line = _s3_snapshot_trace_lines()
    changed = False

    def trace(frame: Any, event: str, _arg: Any):
        nonlocal changed
        if (
            frame.f_code.co_name != "_snapshot_s3_float64_component"
            or event != "line"
            or frame.f_locals.get("label") != "pl"
        ):
            return trace
        if frame.f_lineno == raw_line:
            pl.dtype = np.int64
            changed = True
        elif changed and frame.f_lineno == length_line:
            pl.dtype = np.float64
        return trace

    sys.settrace(trace)
    try:
        actual = element.numerical_internal_force(displacement)["pl"]
    finally:
        sys.settrace(None)
        pl.dtype = np.float64

    assert changed
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    ("route", "component_name"),
    (
        ("stiffness", "total"),
        ("numerical", "pl"),
        ("recovery", "frame"),
        ("mass", "frame"),
        ("directions", "frame"),
        ("geometric", "frame"),
    ),
)
def test_s3_mechanics_consumers_reject_persistent_component_metadata_change(
    route: str,
    component_name: str,
) -> None:
    model, element, material = _s3_case()
    displacement = np.linspace(-2.0e-4, 3.0e-4, 18, dtype=np.float64)

    def evaluate() -> Any:
        if route == "stiffness":
            return element.compute_stiffness_matrix(model.mesh, material)
        if route == "numerical":
            return element.numerical_internal_force(displacement)["pl"]
        if route == "recovery":
            return element.compute_stresses(
                model.mesh,
                displacement,
                material,
            )["membrane_resultants"]
        if route == "mass":
            return element.compute_mass_matrix(model.mesh, material)
        if route == "directions":
            return element.dynamic_algebraic_directions(model.mesh, material)
        if route == "geometric":
            return element.compute_geometric_stiffness_matrix(
                model.mesh,
                material,
                None,
            )
        raise AssertionError(route)

    expected = np.asarray(evaluate(), dtype=np.float64).copy()
    components = object.__getattribute__(element, "__dict__")[
        "_qualified_components"
    ]
    component = components[component_name]
    raw_line, _length_line = _s3_snapshot_trace_lines()
    changed = False

    def trace(frame: Any, event: str, _arg: Any):
        nonlocal changed
        if (
            not changed
            and frame.f_code.co_name == "_snapshot_s3_float64_component"
            and event == "line"
            and frame.f_lineno == raw_line
            and frame.f_locals.get("label") == component_name
        ):
            component.dtype = np.int64
            changed = True
        return trace

    if route == "stiffness":
        # The total-only warm path deliberately bypasses the full component
        # snapshot helper, but still validates the exact total metadata.
        component.dtype = np.int64
        changed = True
    else:
        sys.settrace(trace)
    try:
        expected_error = (
            "cached stiffness authority"
            if route == "stiffness"
            else "component cache authority"
        )
        with pytest.raises(ValueError, match=expected_error):
            evaluate()
    finally:
        sys.settrace(None)
        component.dtype = np.float64

    assert changed
    clean = np.asarray(evaluate(), dtype=np.float64)
    assert np.array_equal(clean, expected)


@pytest.mark.parametrize("nested", (False, True))
def test_s3_component_mapping_replacement_rejects_before_mechanics(
    nested: bool,
) -> None:
    model, element, material = _s3_case()
    displacement = np.linspace(-2.0e-4, 3.0e-4, 18, dtype=np.float64)
    components = object.__getattribute__(element, "__dict__")[
        "_qualified_components"
    ]
    if nested:
        owner = components["assumed_shear_samples"]
        name = "A"
    else:
        owner = components
        name = "pl"
    backing = next(
        value for value in gc.get_referents(owner) if type(value) is dict
    )
    original = backing[name]
    replacement = np.frombuffer(
        original.tobytes(order="C"),
        dtype=np.float64,
    ).reshape(original.shape)
    backing[name] = replacement
    try:
        with pytest.raises(RuntimeError, match="mapping provenance changed"):
            if nested:
                element.compute_stresses(
                    model.mesh,
                    displacement,
                    material,
                )
            else:
                element.numerical_internal_force(displacement)
    finally:
        backing[name] = original
