"""Focused authority tests for the flat qualified-Q4/V2A model boundary."""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.e4_pl_s3_v2_element import (
    BLOCKED_OPERATIONS,
    SUPPORTED_OPERATIONS,
    StrictFlatLinearCapabilityError,
    StrictFlatLinearE4PLS3V2ShellElement,
)
from anyfesolver.elements import BeamElement, ShellElement
from anyfesolver.fe_core import FEMesh, Material


NORMAL = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)


def _material() -> Material:
    return Material("steel", 210.0e9, 0.3, density=7850.0)


def _mixed_mesh(
    *,
    q4_reference_normal: object = NORMAL,
    q4_director_polarity: int = 1,
    out_of_plane: bool = False,
) -> tuple[
    FEMesh,
    StrictFlatLinearE4PLS3V2ShellElement,
    QualifiedE4PLShellElement,
]:
    mesh = FEMesh()
    coordinates = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 1.0, 0.125 if out_of_plane else 0.0),
    )
    for node_id, coordinate in enumerate(coordinates, start=1):
        mesh.add_node(node_id, *coordinate)
    v2 = StrictFlatLinearE4PLS3V2ShellElement(
        1,
        (1, 2, 4),
        "steel",
        thickness=0.08,
        reference_normal=NORMAL,
    )
    q4 = QualifiedE4PLShellElement(
        2,
        [2, 5, 6, 3],
        "steel",
        thickness=0.08,
        reference_normal=q4_reference_normal,
        director_polarity=q4_director_polarity,
        warped_formulation="reject",
    )
    mesh.add_element(v2.element_id, v2)
    mesh.add_element(q4.element_id, q4)
    return mesh, v2, q4


def test_exact_flat_qualified_q4_v2a_mix_is_admitted_without_matrix_change() -> None:
    mesh, v2, _q4 = _mixed_mesh()

    mixed = v2.compute_stiffness_matrix(mesh, _material())

    standalone = FEMesh()
    for node_id, coordinate in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        start=1,
    ):
        standalone.add_node(node_id, *coordinate)
    reference = StrictFlatLinearE4PLS3V2ShellElement(
        1,
        (1, 2, 3),
        "steel",
        thickness=0.08,
        reference_normal=NORMAL,
    ).compute_stiffness_matrix(standalone, _material())

    np.testing.assert_array_equal(mixed, reference)
    assert "flat_qualified_q4_v2a_mixed_mesh" in SUPPORTED_OPERATIONS
    assert "mixed_element_mesh" not in BLOCKED_OPERATIONS
    assert "mixed_shell_mesh" not in BLOCKED_OPERATIONS


def test_physical_director_not_stored_normal_controls_positive_alignment() -> None:
    mesh, v2, _q4 = _mixed_mesh(
        q4_reference_normal=-NORMAL,
        q4_director_polarity=-1,
    )

    stiffness = v2.compute_stiffness_matrix(mesh, _material())

    assert stiffness.shape == (18, 18)
    assert np.all(np.isfinite(stiffness))


def test_missing_or_opposite_q4_physical_director_fails_closed() -> None:
    missing_mesh, missing_v2, _missing_q4 = _mixed_mesh(
        q4_reference_normal=None,
    )
    with pytest.raises(
        StrictFlatLinearCapabilityError,
        match="authoritative reference_normal",
    ):
        missing_v2.compute_stiffness_matrix(missing_mesh, _material())

    opposite_mesh, opposite_v2, _opposite_q4 = _mixed_mesh(
        q4_reference_normal=NORMAL,
        q4_director_polarity=-1,
    )
    with pytest.raises(
        StrictFlatLinearCapabilityError,
        match="positively aligned",
    ):
        opposite_v2.compute_stiffness_matrix(opposite_mesh, _material())


def test_global_coplanarity_covers_q4_and_unreferenced_registered_nodes() -> None:
    warped_mesh, warped_v2, _warped_q4 = _mixed_mesh(out_of_plane=True)
    with pytest.raises(StrictFlatLinearCapabilityError, match="globally coplanar"):
        warped_v2.compute_stiffness_matrix(warped_mesh, _material())

    orphan_mesh, orphan_v2, _orphan_q4 = _mixed_mesh()
    orphan_mesh.add_node(7, 0.5, 0.5, 0.01)
    with pytest.raises(StrictFlatLinearCapabilityError, match="globally coplanar"):
        orphan_v2.compute_stiffness_matrix(orphan_mesh, _material())


@pytest.mark.parametrize("kind", ("legacy", "v1", "beam"))
def test_unqualified_or_nonshell_mixed_elements_fail_closed(kind: str) -> None:
    mesh, v2, q4 = _mixed_mesh()
    mesh.elements.pop(q4.element_id)
    if kind == "legacy":
        other = ShellElement(2, [2, 5, 6, 3], "steel", thickness=0.08)
    elif kind == "v1":
        other = QualifiedE4PLS3ShellElement(
            2,
            [2, 5, 3],
            "steel",
            thickness=0.08,
            reference_normal=NORMAL,
        )
    else:
        other = BeamElement(2, [2, 5], "steel")
    mesh.add_element(2, other)

    with pytest.raises(StrictFlatLinearCapabilityError, match="mixed element"):
        v2.compute_stiffness_matrix(mesh, _material())


def test_registry_identity_and_exact_formulation_identity_fail_closed() -> None:
    malformed, v2, _q4 = _mixed_mesh()
    malformed.elements[99] = malformed.elements.pop(v2.element_id)
    with pytest.raises(StrictFlatLinearCapabilityError, match="registry identity"):
        v2.compute_stiffness_matrix(malformed, _material())

    shadowed, shadowed_v2, q4 = _mixed_mesh()
    object.__setattr__(q4, "formulation_id", "FORGED_Q4")
    with pytest.raises(
        StrictFlatLinearCapabilityError,
        match="formulation identity.*shadowed",
    ):
        shadowed_v2.compute_stiffness_matrix(shadowed, _material())


def test_scope_cache_is_invalidated_by_bound_q4_and_node_mutations() -> None:
    mesh, v2, q4 = _mixed_mesh()
    v2.compute_stiffness_matrix(mesh, _material())

    q4.reference_normal = -NORMAL
    with pytest.raises(StrictFlatLinearCapabilityError, match="positively aligned"):
        v2.compute_stiffness_matrix(mesh, _material())

    restored, restored_v2, _restored_q4 = _mixed_mesh()
    restored_v2.compute_stiffness_matrix(restored, _material())
    restored.set_node_coordinates(6, 2.0, 1.0, 0.1)
    with pytest.raises(StrictFlatLinearCapabilityError, match="globally coplanar"):
        restored_v2.compute_stiffness_matrix(restored, _material())
