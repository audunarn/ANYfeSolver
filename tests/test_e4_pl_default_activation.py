from __future__ import annotations

import ast
from pathlib import Path

from anyfesolver import (
    DEFAULT_Q4_FORMULATION,
    DEFAULT_S3_FORMULATION,
    LegacyShellElement,
    NativeParityE4PLS3V2DShellElement,
    QualifiedE4PLShellElement,
    create_element,
    create_shell_element,
    generate_simple_panel_mesh,
)
from anyfesolver.anystructure_fem_mode import build_fe_model_from_generated_geometry
from anyfesolver.cylinder_benchmarks import (
    CylinderBenchmarkConfig,
    build_cylindrical_shell_benchmark_model,
)


def test_q4_default_and_explicit_rollback_are_topology_closed() -> None:
    assert DEFAULT_Q4_FORMULATION == "e4-pl"
    assert DEFAULT_S3_FORMULATION == "e4-pl-s3-v2d"
    default = create_shell_element(1, [1, 2, 3, 4], "steel")
    factory = create_element("shell", 2, [1, 2, 3, 4], "steel")
    rollback = create_shell_element(
        3, [1, 2, 3, 4], "steel", formulation="legacy"
    )
    rollback_alias = create_element("legacy-shell", 4, [1, 2, 3, 4], "steel")
    tri3 = create_element(
        "shell",
        5,
        [1, 2, 3],
        "steel",
        reference_normal=(0.0, 0.0, 1.0),
    )
    q8 = create_element("shell", 6, list(range(1, 9)), "steel")
    assert type(default) is QualifiedE4PLShellElement
    assert type(factory) is QualifiedE4PLShellElement
    assert type(rollback) is LegacyShellElement
    assert type(rollback_alias) is LegacyShellElement
    assert type(tri3) is NativeParityE4PLS3V2DShellElement
    assert type(q8) is LegacyShellElement


def test_primary_panel_and_generated_geometry_builders_select_q4_default() -> None:
    panel = generate_simple_panel_mesh(
        2.0, 1.0, 0.02, num_divisions_x=2, num_divisions_y=1
    )
    assert all(
        type(element) is QualifiedE4PLShellElement
        for element in panel.mesh.elements.values()
    )
    generated = build_fe_model_from_generated_geometry(
        {
            "nodes": [
                {"id": 1, "coords": [0.0, 0.0, 0.0]},
                {"id": 2, "coords": [1.0, 0.0, 0.0]},
                {"id": 3, "coords": [1.0, 1.0, 0.0]},
                {"id": 4, "coords": [0.0, 1.0, 0.0]},
            ],
            "shells": [
                {"id": 1, "node_ids": [1, 2, 3, 4], "thickness": 0.02}
            ],
        }
    )
    assert type(generated.mesh.elements[1]) is QualifiedE4PLShellElement


def test_cylinder_builder_selects_q4_but_retains_q8_compatibility() -> None:
    q4, _ = build_cylindrical_shell_benchmark_model(
        CylinderBenchmarkConfig(
            num_circumferential=4, num_height=1, use_8node_elements=False
        )
    )
    q8, _ = build_cylindrical_shell_benchmark_model(
        CylinderBenchmarkConfig(
            num_circumferential=4, num_height=1, use_8node_elements=True
        )
    )
    assert all(
        type(element) is QualifiedE4PLShellElement
        for element in q4.mesh.elements.values()
    )
    assert all(type(element) is LegacyShellElement for element in q8.mesh.elements.values())


def test_explicit_e4_request_rejects_non_q4_topology() -> None:
    try:
        create_shell_element(
            1, [1, 2, 3], "steel", formulation="e4-pl"
        )
    except ValueError as exc:
        assert "only for four-node" in str(exc)
    else:
        raise AssertionError("explicit E4-PL request accepted non-Q4 topology")


def test_in_package_shell_construction_is_closed_world_through_selector() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "anyfesolver"
    direct_calls: list[str] = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ShellElement"
            ):
                direct_calls.append(f"{path.relative_to(source_root)}:{node.lineno}")
    assert direct_calls == []
