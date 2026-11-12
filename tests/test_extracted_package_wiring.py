"""Cross-package ownership and compatibility guarantees for ANYfeSolver 0.5."""

from __future__ import annotations

import anyfileio
import anymaterial
import anymesher

import anyfesolver
from anyfesolver import external_references
from anyfesolver.mesh_gen import InterpolatedBeamShellMPCElement


def test_material_public_api_is_canonical_anymaterial() -> None:
    assert anyfesolver.Material is anymaterial.IsotropicMaterial
    assert anyfesolver.OrthotropicMaterial is anymaterial.OrthotropicMaterial
    assert anyfesolver.Hill48Yield is anymaterial.Hill48Yield
    assert anyfesolver.DNVC208MaterialCurve is anymaterial.DNVC208MaterialCurve
    assert anyfesolver.elastic_compliance_matrix is anymaterial.elastic_compliance_matrix


def test_mesh_geometry_public_api_is_canonical_anymesher() -> None:
    assert anyfesolver.StiffenerCrossSection is anymesher.StiffenerCrossSection


def test_interpolated_beam_shell_mpc_is_available_from_the_root_api() -> None:
    assert anyfesolver.InterpolatedBeamShellMPCElement is InterpolatedBeamShellMPCElement


def test_file_document_and_result_api_is_canonical_anyfileio() -> None:
    assert anyfesolver.SesamFemDocument is anyfileio.SesamFemDocument
    assert anyfesolver.FemRawRecord is anyfileio.FemRawRecord
    assert anyfesolver.CalculixParsedResults is anyfileio.CalculixParsedResults
    assert external_references.parse_calculix_frd is anyfileio.parse_frd
    assert external_references.parse_calculix_dat is anyfileio.parse_dat
    assert external_references.merge_calculix_results is anyfileio.merge_results


def test_distribution_version_marks_the_extraction_boundary() -> None:
    assert anyfesolver.__version__ == "0.5.0"
