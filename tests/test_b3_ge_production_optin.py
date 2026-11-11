"""G7 public-boundary tests; the accepted native mechanics are not repeated."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from anyfesolver import b3_ge
from anyfesolver._ge_beam3_native_analysis import NativeBeamAnalysis
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver.elements import QuadraticBeamElement, create_element
from test_ge_beam3_native_generalized import problem


def _definition():
    model, element = problem(curved=True, plastic=False)
    inertia = np.diag((2.0, 2.0, 2.0, 0.07, 0.09, 0.11))
    return model, element, inertia


def test_public_identity_is_explicit_and_hash_stable():
    record = json.loads(b3_ge.ADMISSION_BYTES)
    assert b3_ge.SELECTOR == "b3-ge"
    assert b3_ge.NAME == "B3-GE"
    assert record["selection"] == "EXPLICIT_OPT_IN_ONLY"
    assert record["legacy_b3_default"] is True
    assert record["default_changed"] is False
    assert record["historical_ge_beam3_redirected"] is False
    assert sha256(b3_ge.ADMISSION_BYTES).hexdigest() == b3_ge.ADMISSION_SHA256


def test_g7_contract_binds_the_accepted_g6_record():
    root = Path(__file__).resolve().parents[1]
    contract_raw = (root / "docs/reference_cases/b3_ge_g7_contract_v1.json").read_bytes()
    contract = json.loads(contract_raw)
    assert contract_raw == (
        json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    g6_raw = (root / "docs/reference_cases/ge_beam3_g6_confirmation_v4.json").read_bytes()
    g6_payload = (
        json.dumps(json.loads(g6_raw), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    assert sha256(g6_payload).hexdigest() == contract["g6_confirmation_payload_sha256"]
    assert contract["g6_confirmation_file_sha256"] == b3_ge.G6_CONFIRMATION_FILE_SHA256
    assert contract["g6_confirmation_payload_sha256"] == b3_ge.G6_CONFIRMATION_PAYLOAD_SHA256
    assert contract["legacy_b3_default"] is True
    assert contract["default_changed"] is False


def test_b3_ge_delegates_to_exact_native_definition_and_owner():
    model, element, inertia = _definition()
    made = b3_ge.define_beam(
        b3_ge.SELECTOR,
        element.element_id,
        element.node_ids,
        element.operator.reference.coordinates,
        element.operator.reference.nodal_triads,
        element.section,
        inertia,
        quadrature=element.operator.order,
    )
    direct = NativeBeamDefinition.capture(element, inertia)
    assert type(made) is NativeBeamDefinition
    assert made.raw == direct.raw
    owner = b3_ge.create_analysis(
        b3_ge.SELECTOR, (made,), tuple(model.boundary_conditions)
    )
    assert type(owner) is NativeBeamAnalysis
    provenance = json.loads(b3_ge.workflow_provenance(owner))
    assert provenance["selector"] == "b3-ge"
    assert provenance["legacy_b3_default"] is True
    assert provenance["native"]["profile_id"] == b3_ge.NATIVE_PROFILE_ID


@pytest.mark.parametrize("selector", (None, "B3-GE", "ge-beam3", "ge-beam3-native", "b3"))
def test_public_api_rejects_every_noncanonical_selector(selector):
    model, element, inertia = _definition()
    with pytest.raises(ValueError, match="explicit b3-ge"):
        b3_ge.define_beam(
            selector,
            element.element_id,
            element.node_ids,
            element.operator.reference.coordinates,
            element.operator.reference.nodal_triads,
            element.section,
            inertia,
        )


def test_legacy_b3_factory_default_is_unchanged_and_b3_ge_is_not_an_element_alias():
    legacy = create_element("quadratic_beam", 1, [1, 2, 3])
    assert type(legacy) is QuadraticBeamElement
    with pytest.raises(ValueError, match="Unknown element type"):
        create_element("b3-ge", 2, [1, 2, 3])


def test_historical_ge_beam3_is_not_reinterpreted():
    from anyfesolver.ge_beam3_element import GeometricallyExactBeam3D3NElement

    historical = create_element(
        "ge-beam3",
        1,
        [1, 2, 3],
        section={"stiffness": np.eye(6), "mass_matrix": np.eye(6)},
        reference_orientation=(0.0, 1.0, 0.0),
    )
    assert type(historical) is GeometricallyExactBeam3D3NElement
    assert historical.selector == "ge-beam3"
    assert historical is not b3_ge
