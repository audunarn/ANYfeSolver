"""Predictor admission never substitutes a branch for a genuinely ambiguous pose."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from anysolver.general_contact.corner_contact_indentation import CoupledIndentation
from anysolver.general_contact.corner_contact_geometry import closest_panel_point
from anysolver.general_contact.corner_contact_certificate import CornerContactRefusal


def test_actual_captured_44_45_tie_remains_refused():
    z = np.load(Path(__file__).parent/'data/general_contact/s11_faces_44_45_tie.npz',
                allow_pickle=False)
    with pytest.raises(CornerContactRefusal) as exc:
        closest_panel_point(z['x'],z['panel_positions'],z['face_slices'],z['face_nodes'],
                            z['panel_director_gradient'],faces=z['faces'])
    assert exc.value.code == 'nonunique_closest_point'


def test_refused_speculative_predictor_uses_an_exact_copy_of_accepted_iterate():
    accepted = np.array([0.4,0.3,0.2,0.1])
    seen = []
    def geometry(full):
        seen.append(full.copy())
        raise CornerContactRefusal('nonunique_closest_point','captured seam type')
    sim = SimpleNamespace(q=accepted.copy(),n_s=2,rigid_fields=np.array([[0,0,1],[0,0,2.]]),
                          target=np.array([0.,0.,1.]),_trial_target=np.array([0.,0.,0.75]),
                          bridge=SimpleNamespace(expand=lambda q:q.copy()),_contact_geometry=geometry)
    result = CoupledIndentation._initial_guess(sim)
    np.testing.assert_array_equal(seen[0],accepted+np.array([-.25,-.5,0,0]))
    np.testing.assert_array_equal(result,accepted)
    np.testing.assert_array_equal(sim.q,accepted)
    assert not np.shares_memory(result,sim.q)
    sim._contact_geometry = lambda full: None
    np.testing.assert_array_equal(CoupledIndentation._initial_guess(sim),seen[0])
    def unrelated(full):
        raise CornerContactRefusal('nonfinite_corner','must not be hidden')
    sim._contact_geometry = unrelated
    with pytest.raises(CornerContactRefusal) as exc:
        CoupledIndentation._initial_guess(sim)
    assert exc.value.code == 'nonfinite_corner'
