"""Request correspondence and stale-input rejection without admission claims."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from anyfesolver import (ContactRequestIdentity, ContactRequestIdentityError,
    identify_general_contact_request, verify_general_contact_request,
    ContactPathControl, ContactPathProgram)
from anyfesolver.nonlinear_restart import canonical_checkpoint_json_bytes
from test_general_contact_q4_s3_patch_static import _case


def _request():
    model, definition, settings = _case()
    program = ContactPathProgram((ContactPathControl.node(1, "uz"),), ((0.,), (-.01,)))
    return model, definition, settings, {"contact_path_program": program}


def test_fresh_owner_model_identity_is_deterministic_and_never_admits():
    first = identify_general_contact_request(*_request()[:3], **_request()[3])
    other = identify_general_contact_request(*_request()[:3], **_request()[3])
    assert first == other
    assert not first.admitted_for_application
    record = first.to_dict()
    assert record["purpose"] == "request_identity_only"
    assert not record["admitted_for_application"]
    actual = Path(sys.modules["anyfesolver"].__file__).read_bytes()
    import hashlib
    assert record["request"]["solver_package_files"]["__init__.py"] == hashlib.sha256(actual).hexdigest()
    record["request"]["settings"]["pair_motion"] = "forged"
    assert first.to_dict()["request"]["settings"]["pair_motion"] == "general"


@pytest.mark.parametrize("change", ["model", "contact", "settings", "path", "controls", "source"])
def test_recomputed_identity_rejects_changed_inputs(change):
    model, definition, settings, controls = _request()
    expected = identify_general_contact_request(model, definition, settings, **controls)
    if change == "model":
        model.mesh.get_node(1).x += .01
    elif change == "contact":
        definition = replace(definition, interactions=(replace(definition.interactions[0], normal_stiffness=2000.),))
    elif change == "settings":
        settings = replace(settings, general_s3_difference_step=4e-5)
    elif change == "path":
        controls["contact_path_program"] = replace(controls["contact_path_program"], waypoints=((0.,), (-.02,)))
    elif change == "controls":
        controls["tolerance"] = 1e-7
    else:
        request = expected.to_dict()["request"]
        request["solver_package_files"]["__init__.py"] = "0"*64
        expected = ContactRequestIdentity(canonical_checkpoint_json_bytes(request))
    with pytest.raises(ContactRequestIdentityError, match="contact request changed") as caught:
        verify_general_contact_request(expected, model, definition, settings, **controls)
    assert caught.value.code == "request_mismatch"


def test_mixed_loaded_package_source_is_rejected(monkeypatch, tmp_path):
    module = ModuleType("anyfesolver.foreign_fixture")
    module.__file__ = str(tmp_path/"foreign.py")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ContactRequestIdentityError) as caught:
        identify_general_contact_request(*_request()[:3], **_request()[3])
    assert caught.value.code == "mixed_source"


@pytest.mark.parametrize("change", ["name", "force"])
def test_dead_load_identity_retains_name_and_actual_force(change):
    from anyfesolver import LoadCase
    model, definition, settings, _ = _request()
    load = LoadCase("named load")
    load.add_nodal_load(1, forces=(0., 0., -.1))
    expected = identify_general_contact_request(model, definition, settings, load_case=load)
    request = expected.to_dict()["request"]
    assert request["loading"]["load_case"]["name"] == "named load"
    assert request["loading"]["program"]["nodal_forces"][0]["values_n"] == [0., -.1]
    if change == "name":
        load.name = "different persisted case"
    else:
        load.nodal_loads[1][2] = -.2
    with pytest.raises(ContactRequestIdentityError) as caught:
        verify_general_contact_request(expected, model, definition, settings, load_case=load)
    assert caught.value.code == "request_mismatch"


@pytest.mark.parametrize("bad", [bytearray(b"{}"), b"{}", b'{"schema":"ANYSOLVER_STATIC_CONTACT_REQUEST_V1","x":NaN}', b"[]"])
def test_identity_cannot_retain_mutable_or_noncanonical_data(bad):
    with pytest.raises(ContactRequestIdentityError):
        ContactRequestIdentity(bad)


@pytest.mark.parametrize("controls", [{}, {"contact_path_program": lambda: None},
    {"contact_path_program": True}, {"max_iterations": True}, {"tolerance": float("nan")}])
def test_invalid_request_controls_refuse_without_solve(controls):
    model, definition, settings, valid = _request()
    supplied = controls if not controls else {**valid, **controls}
    with pytest.raises(ContactRequestIdentityError):
        identify_general_contact_request(model, definition, settings, **supplied)
