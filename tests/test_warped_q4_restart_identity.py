"""Checkpoint identity under the 2026-11-24 warped qualified-Q4 drilling floor.

A checkpoint written before the floor (same serialized inputs, other drilling
mechanics on warped facets) must not resume.  Every other model keeps its
fingerprint unchanged.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from anyfesolver import FEModel
from anyfesolver.elements import create_shell_element
from anyfesolver.nonlinear_restart import (
    NonlinearCheckpointError,
    _applied_drilling_if_changed,
    _require_admitted_drilling_hook,
    _sha256,
    model_configuration_descriptor,
    model_configuration_fingerprint,
)


def _shell_model(warped: bool, alpha=None):
    model = FEModel("shell")
    model.add_material("mat", 29.0e6, 0.22)
    n, nw = 4, 2
    for i in range(n + 1):
        phi = (np.pi / 2 * i / n) if warped else 0.0
        for j in range(nw + 1):
            s = -0.55 + 1.1 * j / nw
            model.add_node(i * (nw + 1) + j + 1, 3.0 * i, s * np.cos(phi), s * np.sin(phi))
    kw = {} if alpha is None else {"drilling_stabilization": alpha}
    eid = 0
    for i in range(n):
        for j in range(nw):
            eid += 1
            a, b = i * (nw + 1) + j + 1, (i + 1) * (nw + 1) + j + 1
            model.add_element(eid, create_shell_element(eid, [a, b, b + 1, a + 1], "mat", thickness=0.32, **kw))
    return model


def _pre_floor_fingerprint(model):
    """The fingerprint the same model had before the floor (no applied-drilling field)."""
    descriptor = copy.deepcopy(model_configuration_descriptor(model))
    for record in descriptor["elements"]:
        record.pop("applied_drilling_stabilization", None)
    return _sha256(descriptor)


@pytest.mark.parametrize("alpha", [None, 1.0e-3, 0.0])
def test_floored_warped_q4_binds_the_applied_coefficient(alpha):
    model = _shell_model(warped=True, alpha=alpha)
    records = model_configuration_descriptor(model)["elements"]
    assert [r.get("applied_drilling_stabilization") for r in records] == [0.1] * len(records)
    # A pre-floor checkpoint carries the old fingerprint and is now rejected
    # (validate_nonlinear_checkpoint compares exactly this fingerprint).
    assert model_configuration_fingerprint(model) != _pre_floor_fingerprint(model)


@pytest.mark.parametrize(
    "warped, alpha", [(False, None), (False, 1.0e-3), (False, 0.5), (True, 0.1), (True, 0.5)]
)
def test_unfloored_models_keep_their_fingerprint(warped, alpha):
    model = _shell_model(warped=warped, alpha=alpha)
    records = model_configuration_descriptor(model)["elements"]
    assert all("applied_drilling_stabilization" not in r for r in records)
    assert model_configuration_fingerprint(model) == _pre_floor_fingerprint(model)


def test_applied_coefficient_uses_the_fingerprint_coordinate_snapshot(monkeypatch):
    # Review round 7: a second mesh read could describe geometry other than the
    # snapshotted node coordinates.  The fingerprint must not read the mesh again.
    model = _shell_model(warped=True)
    expected = model_configuration_fingerprint(model)

    def no_second_read(node_id):
        raise AssertionError(f"mesh.get_node({node_id}) read during fingerprinting")

    monkeypatch.setattr(model.mesh, "get_node", no_second_read)
    assert model_configuration_fingerprint(model) == expected


def _records_and_snapshot(model):
    descriptor = model_configuration_descriptor(model)
    snapshot = {r["id"]: r["coordinates"] for r in descriptor["nodes"]}
    return descriptor["elements"], snapshot


@pytest.mark.parametrize("warped", [True, False])
@pytest.mark.parametrize("alpha", [None, 0.0, 1.0e-3, 0.0999, 0.1, 0.5])
def test_recorded_marker_matches_the_element_hook(warped, alpha):
    model = _shell_model(warped=warped, alpha=alpha)
    records, snapshot = _records_and_snapshot(model)
    for record in records:
        element = model.mesh.elements[record["id"]]
        live = element._effective_drilling_stabilization(element.get_node_coordinates(model.mesh))
        expected = None if live == element.drilling_stabilization else live
        assert _applied_drilling_if_changed(record, snapshot) == expected


def test_marker_reads_only_recorded_values():
    # Review round 8: the marker must not reread live element inputs.
    model = _shell_model(warped=True)
    records, snapshot = _records_and_snapshot(model)
    recorded = copy.deepcopy(records)
    for element in model.mesh.elements.values():
        element.drilling_stabilization = 0.5  # live change after recording
    assert [_applied_drilling_if_changed(r, snapshot) for r in recorded] == [0.1] * len(recorded)


def test_degenerate_recorded_facet_is_a_typed_checkpoint_error():
    model = _shell_model(warped=True)
    records, snapshot = _records_and_snapshot(model)
    record = records[0]
    collinear = {n: [float(k), 0.0, 0.0] for k, n in enumerate(record["configuration"]["node_ids"])}
    with pytest.raises(NonlinearCheckpointError, match="drilling identity"):
        _applied_drilling_if_changed(record, collinear)


def test_unknown_recorded_node_is_refused_whatever_the_coefficient():
    # Review round 9 P3: node ids are resolved before the alpha early return.
    model = _shell_model(warped=True, alpha=0.5)
    records, snapshot = _records_and_snapshot(model)
    missing = dict(snapshot)
    del missing[records[0]["configuration"]["node_ids"][0]]
    with pytest.raises(NonlinearCheckpointError, match="drilling identity"):
        _applied_drilling_if_changed(records[0], missing)


def test_shell_subclass_overriding_the_drilling_hook_fails_closed():
    # Review round 9 P2 (owner option C): restart identity cannot describe a
    # custom applied-drilling hook, so such a model is refused.
    from anyfesolver.elements import ShellElement

    class CustomDrillingShell(ShellElement):
        def _effective_drilling_stabilization(self, coordinates=None):
            return 0.25

    model = FEModel("custom")
    model.add_material("mat", 29.0e6, 0.22)
    for k, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)], start=1):
        model.add_node(k, float(x), float(y), 0.0)
    model.add_element(1, CustomDrillingShell(1, [1, 2, 3, 4], "mat", thickness=0.1))
    with pytest.raises(NonlinearCheckpointError, match="overrides the drilling coefficient hook"):
        model_configuration_fingerprint(model)


def _all_shell_classes():
    import importlib
    import pathlib

    import anyfesolver
    from anyfesolver.elements import ShellElement

    root = pathlib.Path(anyfesolver.__file__).parent
    for path in sorted(root.rglob("*.py")):
        if "ShellElement" in path.read_text(encoding="utf-8", errors="ignore"):
            name = ".".join(("anyfesolver",) + path.relative_to(root).with_suffix("").parts)
            if name.endswith(".__init__"):
                name = name[: -len(".__init__")]
            try:
                importlib.import_module(name)
            except Exception:  # dev or optional modules that need extras
                continue
    seen, stack = [], [ShellElement]
    while stack:
        cls = stack.pop()
        if cls not in seen:
            seen.append(cls)
            stack.extend(cls.__subclasses__())
    return [c for c in seen if c.__module__.startswith(("anyfesolver", "anysolver"))]


def test_every_shell_class_in_the_package_is_admitted():
    classes = _all_shell_classes()
    assert len(classes) >= 3
    for cls in classes:
        element = object.__new__(cls)
        _require_admitted_drilling_hook(element)  # must not raise
