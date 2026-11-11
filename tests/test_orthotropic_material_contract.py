"""Solver integration tests for the ANYmaterial structural material contract."""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import (
    FEModel,
    Hill48Yield,
    OrthotropicMaterial,
    StructuralMaterial,
    validate_production_model,
)


def _orthotropic(**overrides) -> OrthotropicMaterial:
    properties = {
        "name": "ud",
        "elastic_modulus_1": 150.0e9,
        "elastic_modulus_2": 10.0e9,
        "elastic_modulus_3": 8.0e9,
        "poisson_ratio_12": 0.25,
        "poisson_ratio_13": 0.20,
        "poisson_ratio_23": 0.30,
        "shear_modulus_12": 5.0e9,
        "shear_modulus_13": 4.0e9,
        "shear_modulus_23": 3.0e9,
        "density": 1600.0,
    }
    properties.update(overrides)
    return OrthotropicMaterial(**properties)


class _ExternalOrthotropicMaterial:
    def __init__(self, name: str = "external") -> None:
        self.name = name
        self.density = 1234.0
        self.elastic_symmetry = "orthotropic"
        self._compliance = _orthotropic().elastic_compliance_matrix()

    def elastic_compliance_matrix(self) -> np.ndarray:
        return self._compliance.copy()


def test_register_material_accepts_external_protocol_and_advances_revision() -> None:
    model = FEModel("external_material")
    initial_revision = model.revision_signature()["material"]
    first = _ExternalOrthotropicMaterial()

    returned = model.register_material(first)

    assert isinstance(first, StructuralMaterial)
    assert returned is first
    assert model.get_material("external") is first
    assert model.revision_signature()["material"] == initial_revision + 1

    replacement = _ExternalOrthotropicMaterial()
    model.register_material(replacement)
    assert model.get_material("external") is replacement
    assert model.revision_signature()["material"] == initial_revision + 2


def test_add_orthotropic_material_and_anisotropic_rejection_are_explicit() -> None:
    model = FEModel("orthotropic")
    material = model.add_orthotropic_material(
        "panel",
        150.0e9,
        10.0e9,
        8.0e9,
        0.25,
        0.20,
        0.30,
        5.0e9,
        4.0e9,
        3.0e9,
        density=1600.0,
        hill_yield=Hill48Yield(400e6, 300e6, 250e6, 180e6, 170e6, 160e6),
    )
    assert model.get_material("panel") is material
    assert material.hill48_yield is material.hill_yield

    anisotropic = _ExternalOrthotropicMaterial("general")
    anisotropic.elastic_symmetry = "anisotropic"
    with pytest.raises(ValueError, match="general anisotropic elasticity is not supported"):
        model.register_material(anisotropic)

    # Production validation also catches unsupported objects injected by
    # deserializers or legacy callers that bypass registration.
    model.materials["general"] = anisotropic
    report = validate_production_model(model, allow_free_mechanisms=True)
    messages = " ".join(issue.message for issue in report.errors)
    assert report.status == "invalid"
    assert "General anisotropic elasticity is not supported" in messages
