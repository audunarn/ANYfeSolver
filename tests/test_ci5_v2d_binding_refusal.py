"""Focused CI5 coverage for the exact V2D committed-displacement binding.

Direct fact: the CI5 native arc/chart nodes fail at the exact
``committed_total_u`` binding (``np.array_equal`` against the value recomputed
from the restored global state) during historical fixture restore on ubuntu.
The recomputation path can depend on BLAS kernel dispatch. These tests only
pin the refusal semantics of the binding itself at the state level:
exact acceptance without mutation, single-ulp refusal with bounded exact-hex
evidence, and refusal of a smallest-subnormal component.  No model, fixture
or solve is used, so the coverage is stable on every platform.
"""
from __future__ import annotations

import numpy as np
import pytest

from anyfesolver.e4_pl_s3_v2d_state import (
    V2DStateError,
    canonical_json_bytes,
    initialize_v2d_state,
    seal_v2d_state,
    validate_v2d_state,
)

IDENTITY = "CI5_SYNTHETIC_ELEMENT_IDENTITY_SHA256"
ELEMENT_ID = 7
NODE_IDS = (1, 2, 5)
LAYERS = 3
MODE = "LAYERED_PLANE_STRESS"
# Deterministic binary64 values near the CI5 failure scale; the exact binding
# must hold for arbitrary magnitudes, including the tiny entries below.
COMMITTED = (
    -3.37529282e-07, -1.10004785e-07, 6.77626358e-21,
    7.84722210e-07, 1.94628790e-05, 2.02382517e-07,
    -1.10004785e-07, 3.60706120e-09, 1.69406589e-21,
    -1.40093140e-06, -6.96138371e-06, 1.56832069e-07,
    0.0, -4.04765038e-21, 8.81316331e-07,
    -9.73143950e-06, 2.54881045e-08, -7.19263499e-22,
)


def _state(**overrides):
    state = initialize_v2d_state(
        element_id=ELEMENT_ID,
        node_ids=NODE_IDS,
        element_identity_sha256=IDENTITY,
        num_layers=LAYERS,
        material_mode=MODE,
        initial_generalized_prestrain=np.zeros((3, 8)),
        initial_generalized_resultant=np.zeros((3, 8)),
    )
    state["committed_total_u"] = np.asarray(COMMITTED, dtype=np.float64)
    state.update(overrides)
    return seal_v2d_state(state)


def _validate(state, *, expected=None, element_id=ELEMENT_ID,
              node_ids=NODE_IDS, identity=IDENTITY, layers=LAYERS, mode=MODE):
    return validate_v2d_state(
        state,
        element_id=element_id,
        node_ids=node_ids,
        element_identity_sha256=identity,
        num_layers=layers,
        material_mode=mode,
        expected_committed_total_u=expected,
    )


def test_exact_binding_accepts_and_does_not_mutate_the_state():
    state = _state()
    before = canonical_json_bytes(state)
    validated = _validate(state, expected=COMMITTED)
    np.testing.assert_array_equal(validated["committed_total_u"], COMMITTED)
    assert canonical_json_bytes(state) == before


def test_single_ulp_perturbation_refuses_with_exact_evidence():
    state = _state()
    perturbed = np.asarray(COMMITTED, dtype=np.float64)
    perturbed[3] = np.nextafter(perturbed[3], np.inf)
    tampered = _state(committed_total_u=perturbed)
    with pytest.raises(V2DStateError) as caught:
        _validate(tampered, expected=COMMITTED)
    message = str(caught.value)
    assert "V2D committed displacement binding mismatch" in message
    assert "; element_id=7" in message
    assert "; node_ids=[1, 2, 5]" in message
    assert "; differing_entries=1/18" in message
    assert "; first_index=3" in message
    assert f"; stored={float(perturbed[3]).hex()}" in message
    assert f"; expected={float(COMMITTED[3]).hex()}" in message


def test_smallest_subnormal_component_is_not_discarded():
    # A zero entry perturbed to the smallest subnormal must still refuse:
    # the exact binding admits no small-component tolerance.
    perturbed = np.asarray(COMMITTED, dtype=np.float64)
    perturbed[12] = np.nextafter(0.0, 1.0)
    tampered = _state(committed_total_u=perturbed)
    with pytest.raises(V2DStateError, match="committed displacement binding mismatch"):
        _validate(tampered, expected=COMMITTED)
