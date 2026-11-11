"""Matched flat-web geometry and reciprocal work in a bounded native solve."""

import numpy as np
import pytest

from anyfesolver import stiffened_shell
from anyfesolver._stiffened_shell_l1_assembly_experimental import SPACE_A, SPACE_B
from scripts.b3_e4_f03_l1_matched_flat_web import build, evaluate
from scripts.b3_e4_f03_l1_uncoupled_flat_controls import evaluate as controls


def test_native_model_matches_declared_external_section_and_top_line():
    model, plate_tip, beam_tip, web = build(SPACE_A)
    points = np.asarray(model.shell.coordinates_m)
    assert points[:, 0].min() == 0.
    assert points[:, 0].max() == 1.
    assert points[:, 1].min() == -.1
    assert points[:, 1].max() == .1
    assert np.all(points[:, 2] == 0.)
    assert len(model.shell.q4_elements) == 16
    assert len(model.beams) == 2
    assert model.lines[0].beam_offset == (0., 0., -.1)
    assert len(model.lines[0].shell_nodes) == 5
    assert plate_tip == model.lines[0].shell_nodes[-1]
    assert beam_tip == 105
    assert web["area_m2"] == pytest.approx(.01*.2)
    assert web["Iy_m4"] == pytest.approx(.01*.2**3/12)
    assert web["Iz_m4"] == pytest.approx(.2*.01**3/12)
    assert web["J_rect_m4"] > 0.


def test_two_native_spaces_retain_equilibrium_and_reciprocal_work():
    result = evaluate()
    assert set(result["spaces"]) == {SPACE_A, SPACE_B}
    for value in result["spaces"].values():
        assert value["factorizations"] == 1
        assert all(max(abs(case["beam_root_uy_m"]), abs(case["beam_root_uz_m"])) <= 1e-12
                   for case in value["cases"].values())
        assert value["reciprocity_relative"] < 1e-5
        assert all(case["load_displacement_work_j"] > 0.
                       and case["free_equilibrium_ratio"] < 1e-8
                       and case["constraint_max"] < 1e-9
                       for case in value["cases"].values())
        assert all(case["shell_internal_energy_j"] >= 0.
                       and case["beam_internal_energy_j"] >= 0.
                       and case["shell_internal_energy_j"]+
                       case["beam_internal_energy_j"] == pytest.approx(
                           case["load_displacement_work_j"]/2, rel=1e-8)
                       for case in value["cases"].values())


def test_uncoupled_native_blocks_and_uniform_torsion_oracle():
    result = controls()
    plate = result["cases"]["plate_only"]
    web = result["cases"]["web_only"]
    assert plate["matrix_rows"] == 120
    assert web["matrix_rows"] == 72
    assert plate["load_displacement_work_j"] > 0.
    g = 210e9/(2*(1+.3))
    expected_web_work = 10.**2/(g*result["section"]["J_rect_m4"])
    assert web["load_displacement_work_j"] == pytest.approx(
        expected_web_work, rel=1e-10)
    assert plate["free_equilibrium_ratio"] < 1e-8
    assert web["free_equilibrium_ratio"] < 1e-8


def test_redundant_direct_and_p1_root_normal_support_is_rejected():
    model, *_ = build(SPACE_A)
    with pytest.raises(ValueError, match="independent supported attachment rows"):
        stiffened_shell.create_analysis(definition=model)
