"""Formatting of native cached generalized responses, without recovery replay."""
from dataclasses import replace
import json
import numpy as np
import pytest

from anyfesolver._ge_beam3_generalized_cell import GeneralizedCellConjugate
from anyfesolver._ge_beam3_fibre_section import canonical
from anyfesolver._ge_beam3_retained_generalized import RetainedGeneralizedOperator
from anyfesolver._stiffened_shell_generalized_recovery import build
from test_stiffened_shell_bl5_material import model


@pytest.fixture
def cached():
    element, _ = model(hardening=1e8).beams[0].instantiate()
    operator = element.operator
    from scipy.spatial.transform import Rotation
    cells = Rotation.from_rotvec([[.2,-.1,.05],[-.13,.07,.1]]).as_matrix()
    response = operator.evaluate(operator.reference.coordinates,
        np.zeros((3, 3)), operator.reference.nodal_triads, cells,
        np.linspace(-.1, .2, 18), origin=operator.cell.virgin())
    return element, cells, response


def test_cached_station_order_pairs_and_detachment(cached, monkeypatch):
    element, cells, response = cached
    def forbidden(*args, **kwargs):
        raise AssertionError("cached recovery must not recompute mechanics")
    monkeypatch.setattr(RetainedGeneralizedOperator, "evaluate", forbidden)
    monkeypatch.setattr(RetainedGeneralizedOperator, "recover", forbidden)
    monkeypatch.setattr(GeneralizedCellConjugate, "response", forbidden)
    records = build(element, cells, response.material, response.history)
    raw = json.loads(response.material)
    assert records.element_id == element.element_id
    assert not records.production_qualified
    for record, location, station, history in zip(records.stations, element.operator.stations,
                                                raw["stations"], response.history.stations):
        cell, index, xi, measure, frame = location
        assert (record.cell, record.station, record.xi, record.measure) == (cell, index, xi, measure)
        np.testing.assert_array_equal(record.local_strain, station["strain"][0])
        np.testing.assert_array_equal(record.local_strain_low, station["strain"][1])
        np.testing.assert_array_equal(record.local_resultants, station["resultants"][0])
        np.testing.assert_array_equal(record.local_resultants_low, station["resultants"][1])
        np.testing.assert_array_equal(record.reference_frame, frame)
        np.testing.assert_array_equal(record.current_frame, cells[cell] @ frame)
        np.testing.assert_array_equal(record.plastic_strain, np.asarray(history.plastic)[:, 0])
        assert record.accumulated == history.accumulated
        for name, key in (("stored_energy_per_length", "stored_energy"),
                          ("dissipation_per_length", "dissipation"),
                          ("plastic_increment", "plastic_increment")):
            assert getattr(record, name) == tuple(part[0] for part in station[key])
        assert not record.reference_frame.flags.writeable
        assert not np.shares_memory(record.reference_frame, frame)
        record.reference_frame.setflags(write=True)
        record.reference_frame[:] = 123.
        assert np.max(abs(frame)) <= 1.
    cells[:] = 0.
    assert np.linalg.det(records.stations[0].current_frame) == pytest.approx(1.)


@pytest.mark.parametrize("mutation", (
    lambda r: r.update(cell_identity="foreign"),
    lambda r: r["stations"].pop(),
    lambda r: r["stations"][0].update(strain=[[0.]*5, [0.]*5]),
    lambda r: r["stations"][0].update(branch="unsupported"),
    lambda r: r["stations"][0].update(dissipation=[[-1.], [0.]]),
))
def test_invalid_cached_fields(cached, mutation):
    element, cells, response = cached
    data = json.loads(response.material); mutation(data)
    with pytest.raises(ValueError):
        build(element, cells, canonical(data), response.history)


def test_bad_frames_history_and_nonfinite_fields(cached):
    element, cells, response = cached
    with pytest.raises(ValueError, match="history"):
        build(element, cells, response.material, replace(response.history, cell_identity="foreign"))
    with pytest.raises(ValueError, match="frame"):
        build(element, np.zeros_like(cells), response.material, response.history)
    data = json.loads(response.material)
    data["stations"][0]["resultants"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build(element, cells, json.dumps(data).encode(), response.history)
