"""Focused public-contract probe for supported ANYmesher endpoints."""

from __future__ import annotations

import argparse
import json
import os
import re
import tomllib
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name


EXPECTED_REQUIREMENT = "ANYmesher>=0.4,<0.6"
EXPECTED_SPECIFIER = SpecifierSet(">=0.4,<0.6")


def _assert_expected_anymesher_requirement(requirement: str) -> None:
    parsed = Requirement(requirement)
    assert canonicalize_name(parsed.name) == "anymesher"
    assert not parsed.extras
    assert parsed.marker is None
    assert parsed.url is None
    assert parsed.specifier == EXPECTED_SPECIFIER


def _installed_anymesher_requirements() -> list[str]:
    requirements = metadata.requires("ANYfeSolver") or []
    return [
        requirement
        for requirement in requirements
        if re.match(r"(?i)^anymesher(?:\s|[<>=!~])", requirement)
    ]


def _source_anymesher_requirements() -> list[str]:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return [
        requirement
        for requirement in project["project"]["dependencies"]
        if canonicalize_name(Requirement(requirement).name) == "anymesher"
    ]


def _is_beneath(path: Path, root: Path) -> bool:
    path_text = os.path.normcase(str(path.resolve()))
    root_text = os.path.normcase(str(root.resolve()))
    try:
        return os.path.commonpath((path_text, root_text)) == root_text
    except ValueError:
        return False


def test_is_beneath_treats_commonpath_value_error_as_not_beneath(
    monkeypatch,
) -> None:
    def reject_cross_drive(_paths: tuple[str, str]) -> str:
        raise ValueError("Paths don't have the same drive")

    monkeypatch.setattr(os.path, "commonpath", reject_cross_drive)
    assert _is_beneath(Path("installed-origin"), Path("workspace")) is False


def test_is_beneath_propagates_non_value_error(monkeypatch) -> None:
    import pytest

    def fail_unexpectedly(_paths: tuple[str, str]) -> str:
        raise RuntimeError("unexpected commonpath failure")

    monkeypatch.setattr(os.path, "commonpath", fail_unexpectedly)
    with pytest.raises(RuntimeError, match="unexpected commonpath failure"):
        _is_beneath(Path("installed-origin"), Path("workspace"))


def _assert_probe_identities(
    modules: dict[str, ModuleType], *, metadata_mode: str
) -> None:
    expected_variables = {
        "ANYfeSolver": "EXPECTED_ANYSOLVER_VERSION",
        "ANYmaterial": "EXPECTED_ANYMATERIAL_VERSION",
        "ANYgeometry": "EXPECTED_ANYGEOMETRY_VERSION",
        "ANYmesher": "EXPECTED_ANYMESHER_VERSION",
        "ANYfileio": "EXPECTED_ANYFILEIO_VERSION",
    }
    workspace_raw = os.environ.get("GITHUB_WORKSPACE")
    if metadata_mode == "installed":
        assert workspace_raw is not None
    for distribution_name, module in modules.items():
        expected = os.environ.get(expected_variables[distribution_name])
        if metadata_mode == "installed":
            assert expected is not None, expected_variables[distribution_name]
        if expected is not None:
            assert module.__version__ == expected
        if metadata_mode == "installed":
            distribution = metadata.distribution(distribution_name)
            assert distribution.version == expected
            distribution_root = Path(distribution.locate_file("")).resolve()
            origin = Path(module.__file__).resolve()
            assert _is_beneath(origin, distribution_root)
            assert not _is_beneath(origin, Path(workspace_raw))
            assert not _is_beneath(distribution_root, Path(workspace_raw))


def _metadata_requirement(metadata_mode: str) -> str:
    if metadata_mode == "source":
        requirements = _source_anymesher_requirements()
    else:
        assert metadata_mode == "installed"
        requirements = _installed_anymesher_requirements()
    assert len(requirements) == 1
    _assert_expected_anymesher_requirement(requirements[0])
    return requirements[0]


def probe_anymesher_public_contract(
    *, metadata_mode: str = "source"
) -> dict[str, Any]:
    """Exercise the exact neutral-meshing surface consumed by ANYfeSolver."""

    import anyfileio
    import anygeometry
    import anymaterial
    import anymesher
    import anyfesolver

    modules = {
        "ANYfeSolver": anyfesolver,
        "ANYmaterial": anymaterial,
        "ANYgeometry": anygeometry,
        "ANYmesher": anymesher,
        "ANYfileio": anyfileio,
    }
    _assert_probe_identities(modules, metadata_mode=metadata_mode)

    required_names = (
        "Mesh",
        "PanelMeshConfig",
        "StiffenedPanel",
        "StiffenerCrossSection",
        "panel_edge_nodes",
        "stiffened_panel_mesh",
        "verify_mesh_quality",
    )
    assert all(hasattr(anymesher, name) for name in required_names)
    assert anyfesolver.StiffenerCrossSection is anymesher.StiffenerCrossSection

    panel = anymesher.StiffenedPanel(
        length=2.4,
        width=1.8,
        plate_thickness=0.012,
        stiffener_type="T-bar",
        stiffener_spacing=0.55,
        stiffener_height=0.18,
        stiffener_web_thickness=0.008,
        stiffener_flange_width=0.10,
        stiffener_flange_thickness=0.012,
        num_stiffeners=2,
    )
    config = anymesher.PanelMeshConfig(
        shell_num_divisions_x=5,
        shell_num_divisions_y=4,
        beam_num_divisions=3,
        use_coupling_elements=True,
        tolerance=1.0e-9,
        use_8node_shells=True,
        align_mesh_to_stiffeners=True,
    )
    mesh = anymesher.stiffened_panel_mesh(panel, config)
    edge_nodes = anymesher.panel_edge_nodes(mesh)
    quality = anymesher.verify_mesh_quality(mesh)

    assert isinstance(mesh, anymesher.Mesh)
    assert mesh.nodes
    assert mesh.quads
    assert mesh.beams
    assert mesh.couplings
    assert edge_nodes["all"]
    assert isinstance(quality, anymesher.MeshQuality)
    shell_element_count = len(mesh.quads) + len(mesh.tris)
    assert shell_element_count > 0

    requirement = _metadata_requirement(metadata_mode)

    return {
        "anymesher_version": anymesher.__version__,
        "anymesher_origin": str(Path(anymesher.__file__).resolve()),
        "anygeometry_version": anygeometry.__version__,
        "anygeometry_origin": str(Path(anygeometry.__file__).resolve()),
        "anyfileio_version": anyfileio.__version__,
        "anyfileio_origin": str(Path(anyfileio.__file__).resolve()),
        "anymaterial_version": anymaterial.__version__,
        "anymaterial_origin": str(Path(anymaterial.__file__).resolve()),
        "anyfesolver_version": anyfesolver.__version__,
        "anyfesolver_origin": str(Path(anyfesolver.__file__).resolve()),
        "requirement": requirement,
        "metadata_mode": metadata_mode,
        "node_count": len(mesh.nodes),
        "quad_count": len(mesh.quads),
        "beam_count": len(mesh.beams),
        "coupling_count": len(mesh.couplings),
        "edge_node_count": len(edge_nodes["all"]),
        "quality_element_count": shell_element_count,
    }


def test_source_declares_exact_anymesher_compatibility_range() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [
        requirement
        for requirement in project["project"]["dependencies"]
        if requirement.lower().startswith("anymesher")
    ]
    assert requirements == [EXPECTED_REQUIREMENT]


def test_installed_requirement_accepts_canonicalized_specifier_order() -> None:
    _assert_expected_anymesher_requirement("ANYmesher<0.6,>=0.4")


def test_anymesher_public_contract_used_by_anyfesolver() -> None:
    result = probe_anymesher_public_contract()
    assert result["node_count"] > 0
    assert result["quality_element_count"] > 0


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument(
        "--metadata-mode", choices=("source", "installed"), required=True
    )
    arguments = parser.parse_args()
    if not arguments.probe:
        parser.error("the standalone entry point requires --probe")
    print(
        json.dumps(
            probe_anymesher_public_contract(
                metadata_mode=arguments.metadata_mode
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
