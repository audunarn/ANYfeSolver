"""Read data emitted by the solver immediately before the public namespace move."""

import hashlib
import json
from pathlib import Path

import pytest

from anyfesolver.general_contact import (
    read_arc_contact_results,
    read_static_contact_results,
    read_transient_contact_results,
)
from anyfesolver.nonlinear_static import solve_static_nonlinear
from test_nonlinear_restart_checkpoint import _model, _static_kwargs


FIXTURES = Path(__file__).parent / "data" / "namespace_migration"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_bytes())


def historical_bytes(name):
    data = (FIXTURES / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == MANIFEST["sha256"][name]
    return data


@pytest.mark.parametrize(("name", "reader"), [
    ("static.json", read_static_contact_results),
    ("arc.json", read_arc_contact_results),
    ("transient.json", read_transient_contact_results),
])
def test_detached_archive_preserves_historical_wire_bytes(name, reader):
    payload = historical_bytes(name)
    assert reader(payload).to_bytes() == payload


def test_resume_pre_rename_checkpoint_matches_original_continuation():
    model, load = _model(activity=0.5)
    result = solve_static_nonlinear(
        model, load, max_load_factor=0.30, num_steps=2,
        restart_checkpoint=historical_bytes("restart.json"), **_static_kwargs(),
    )
    assert result.status == "completed"
    assert result.displacements.tobytes().hex() == MANIFEST["uninterrupted_displacement_hex"]
    assert hashlib.sha256(result.restart_checkpoint_bytes()).hexdigest() == MANIFEST["uninterrupted_checkpoint_sha256"]
