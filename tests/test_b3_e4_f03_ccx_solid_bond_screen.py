"""Bounded controls for the exploratory conforming solid-bond deck."""

import importlib.util
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "b3_e4_f03_ccx_solid_bond_screen.py"
SPEC = importlib.util.spec_from_file_location("solid_bond_screen", MODULE)
SCREEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCREEN)


@pytest.mark.parametrize("load", ("tip_mx", "tip_fz"))
def test_full_bond_displacement_deck_changes_output_only(load):
    options = dict(boundary_policy="left_root", bond_policy="full",
                   element_family="C3D20R", plate_thickness_mm=35,
                   tip_load_map="quadratic_face")
    original, original_meta = SCREEN.build(4, load, **options)
    expanded, expanded_meta = SCREEN.build(
        4, load, full_displacement_output=True, **options)
    output_card = "*NODE FILE,OUTPUT=3D\nU\n"
    assert expanded.count(output_card) == 1
    assert expanded.replace(output_card, "") == original
    assert {key:value for key,value in expanded_meta.items()
            if key != "full_displacement_output"} == original_meta
    assert expanded_meta["full_displacement_output"] is True
    with pytest.raises(ValueError, match="bounded full-bond"):
        SCREEN.build(2, load, full_displacement_output=True, **options)


@pytest.mark.parametrize("factor", (1, 2, 4))
@pytest.mark.parametrize("shear", ("uy_x", "ux_y"))
def test_shared_interface_and_prescribed_plate_only(factor, shear):
    deck, meta = SCREEN.build(factor, shear)
    assert meta["shared_interface_nodes"] == (2 * factor + 1) * (4 * factor + 1)
    assert meta["plate_elements"] == 96 * factor**3
    assert meta["bar_elements"] == 16 * factor**3
    assert deck.count("*SOLID SECTION") == 2
    assert "*MPC" not in deck and "*TIE" not in deck
    assert len(meta["probe_node_ids"]) == len(set(meta["probe_node_ids"]))
    for station in meta["bar_station_nodes"].values():
        assert len(station) == (2 * factor + 1)**2


@pytest.mark.parametrize("factor", (1, 2, 4))
@pytest.mark.parametrize("load,expected", (("tip_mx", [0.,0.,0.,100.,0.,0.]),
                                            ("tip_fz", [0.,0.,1000.,0.,-1000.,0.])))
def test_terminal_bond_has_only_declared_shared_patch(factor, load, expected):
    deck, meta = SCREEN.build(factor, load, boundary_policy="left_root",
                              bond_policy="terminal")
    assert meta["bond_policy"] == "terminal"
    assert meta["bond_interval_m"] == [.9, 1.]
    assert meta["shared_interface_nodes"] == (factor+1)*(2*factor+1)
    assert meta["detached_interface_nodes"] == 4*factor*(2*factor+1)
    assert len(meta["shared_interface_node_ids"]) == meta["shared_interface_nodes"]
    assert len(meta["detached_stiffener_node_ids"]) == meta["detached_interface_nodes"]
    assert not set(meta["detached_stiffener_node_ids"]) & set(
        meta["shared_interface_node_ids"])
    assert not set(meta["beam_root_node_ids"]) & set(
        meta["plate_boundary_node_ids"])
    assert set(meta["boundary_node_ids"]) == (
        set(meta["beam_root_node_ids"]) | set(meta["plate_boundary_node_ids"]))
    assert "*MPC" not in deck and "*TIE" not in deck
    applied = {int(node): np.array(force) for node, force in
               meta["applied_nodal_forces_N"].items()}
    assert SCREEN._wrench(applied, meta["probe_coordinates_m"]) == pytest.approx(
        expected, abs=1e-10)
    displacement = {int(node): np.zeros(3) for node in meta["probe_node_ids"]}
    recovered = SCREEN.recover(meta, displacement)
    assert recovered["station_motion"]["500"]["interface_shared"] is False
    assert "shared_face_mean_translation_m" not in recovered["station_motion"]["500"]
    assert recovered["station_motion"]["1000"]["interface_shared"] is True
    assert "shared_face_mean_translation_m" in recovered["station_motion"]["1000"]
    assert recovered["unbonded_midspan_normal_separation_m"][
        "bar_minus_plate_z_m"] == [0., 0., 0.]
    bottom = next(index for index in meta["bar_station_nodes"]["500"]
                  if meta["probe_coordinates_m"][str(index)] == [.5, 0., .035])
    displacement[bottom] = np.array([0., 0., -1e-6])
    assert SCREEN.recover(meta, displacement)[
        "unbonded_midspan_normal_separation_m"]["bar_minus_plate_z_m"][1] == pytest.approx(-1e-6)


@pytest.mark.parametrize("factor",(1,2))
def test_half_thickness_quadratic_terminal_torque_preserves_bond_and_wrench(factor):
    deck,meta=SCREEN.build(
        factor,"tip_mx",boundary_policy="left_root",bond_policy="terminal",
        element_family="C3D20R",tip_mx_gap_audit=True,
        bar_clearance_um=2,plate_thickness_mm=35)
    assert meta["plate_thickness_m"]==.035
    assert meta["bar_tip_axis_z_m"]==.0675
    assert "x[-.0175,.0175]" in meta["model"]
    assert deck.count("*ELEMENT,TYPE=C3D20R")==2
    assert meta["shared_interface_nodes"]>0
    assert meta["detached_interface_nodes"]>0
    assert len(meta["unbonded_face_pairs"])>0
    assert min(meta["unbonded_initial_gaps_m"])>0
    assert max(meta["unbonded_initial_gaps_m"])==pytest.approx(2e-6)
    tip_z=[meta["probe_coordinates_m"][str(i)][2]
           for i in meta["bar_station_nodes"]["1000"]]
    root_z=[meta["boundary_coordinates_m"][str(i)][2]
            for i in meta["beam_root_node_ids"]]
    assert np.mean(tip_z)==pytest.approx(.0675)
    assert np.mean(root_z)==pytest.approx(.067502)
    applied={int(i):np.asarray(force) for i,force in
             meta["applied_nodal_forces_N"].items()}
    assert SCREEN._wrench(applied,meta["probe_coordinates_m"])==pytest.approx(
        [0.,0.,0.,100.,0.,0.],abs=1e-9)
    probe={i:np.zeros(3) for i in meta["probe_node_ids"]}
    recovered=SCREEN.recover(meta,probe)
    assert recovered["station_motion"]["1000"]["interface_shared"]
    assert recovered["unbonded_midspan_normal_separation_m"][
        "bar_minus_plate_z_m"]==pytest.approx([2e-6]*3,abs=1e-12)


@pytest.mark.parametrize("bond",("terminal","detached"))
def test_quadratic_tip_traction_has_exact_wrench_and_conjugate_work(bond):
    _,meta=SCREEN.build(
        2,"tip_mx",boundary_policy="left_root",bond_policy=bond,
        element_family="C3D20R",tip_mx_gap_audit=bond=="terminal",
        bar_clearance_um=2 if bond=="terminal" else 0,
        plate_thickness_mm=35,tip_load_map="quadratic_face")
    applied={int(i):np.asarray(force) for i,force in
             meta["applied_nodal_forces_N"].items()}
    assert len(applied)>len(meta["bar_station_nodes"]["1000"])
    assert set(applied).issubset(meta["probe_node_ids"])
    assert SCREEN._wrench(applied,meta["probe_coordinates_m"])==pytest.approx(
        [0.,0.,0.,100.,0.,0.],abs=1e-9)
    theta=1e-5
    u={int(i):np.array([0.,-(xyz[2]-.0675)*theta,xyz[1]*theta])
       for i,xyz in meta["probe_coordinates_m"].items()}
    assert SCREEN.recover(meta,u)["tip_work_conjugate_displacement"][
        "value"]==pytest.approx(theta,abs=1e-13)


def test_quadratic_reverse_force_has_exact_wrench_and_work():
    _,meta=SCREEN.build(
        2,"tip_fz",boundary_policy="left_root",bond_policy="terminal",
        element_family="C3D20R",reverse_tip_fz=True,
        plate_thickness_mm=35,tip_load_map="quadratic_face")
    applied={int(i):np.asarray(force) for i,force in
             meta["applied_nodal_forces_N"].items()}
    assert len(applied)>len(meta["bar_station_nodes"]["1000"])
    assert SCREEN._wrench(applied,meta["probe_coordinates_m"])==pytest.approx(
        [0.,0.,-1000.,0.,1000.,0.],abs=1e-9)
    u={int(i):np.array([0.,0.,-2e-5])
       for i in meta["probe_node_ids"]}
    assert SCREEN.recover(meta,u)["tip_work_conjugate_displacement"][
        "value"]==pytest.approx(2e-5,abs=1e-13)


@pytest.mark.parametrize("factor", (2, 4))
@pytest.mark.parametrize("load,expected", (
    ("tip_mx", [0., 0., 0., 100., 0., 0.]),
    ("tip_fz", [0., 0., 1000., 0., -1000., 0.])))
def test_quadratic_full_bond_is_conforming_and_has_exact_face_wrench(
        factor, load, expected):
    deck, meta = SCREEN.build(
        factor, load, boundary_policy="left_root", bond_policy="full",
        element_family="C3D20R", plate_thickness_mm=35,
        tip_load_map="quadratic_face")
    assert meta["bond_interval_m"] == [0., 1.]
    assert meta["plate_thickness_m"] == .035
    assert meta["bar_tip_axis_z_m"] == .0675
    assert meta["detached_interface_nodes"] == 0
    assert not meta["unbonded_face_pairs"]
    assert meta["shared_interface_nodes"] > 0
    assert set(meta["boundary_node_ids"]) == (
        set(meta["beam_root_node_ids"]) |
        set(meta["plate_boundary_node_ids"]))
    assert "*MPC" not in deck and "*TIE" not in deck
    applied = {int(i): np.asarray(force) for i, force in
               meta["applied_nodal_forces_N"].items()}
    assert SCREEN._wrench(applied, meta["probe_coordinates_m"]) == pytest.approx(
        expected, abs=1e-9)
    assert set(applied).issubset(meta["probe_node_ids"])
    assert len(applied) > len(meta["bar_station_nodes"]["1000"])


def test_thickness_screen_rejects_undeclared_geometry():
    with pytest.raises(ValueError,match="35/70 mm"):
        SCREEN.build(1,"tip_mx",plate_thickness_mm=50)


def test_quadratic_face_bernstein_check_distinguishes_hidden_dip():
    sites=((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.),
           (0.,-1.),(1.,0.),(0.,1.),(-1.,0.))
    positive=np.array([1e-6*(.25*x*x+.01) for x,_ in sites])
    bounded=SCREEN.certify_quad8_normal_gap(positive)
    assert bounded["root_bernstein_lower_m"]<0
    assert bounded["certified_nonnegative"]
    assert bounded["visited_subpatches"]>1
    negative=np.array([1e-6*(x*x+y*y-.1) for x,y in sites])
    penetrated=SCREEN.certify_quad8_normal_gap(negative)
    assert not penetrated["certified_nonnegative"]
    assert penetrated["negative_subpatches"]>0
    with pytest.raises(ValueError,match="bounded finite"):
        SCREEN.certify_quad8_normal_gap(np.ones(7))


@pytest.mark.parametrize("factor", (1, 2, 4))
def test_reverse_fz_audit_preserves_geometry_and_reverses_exact_tip_wrench(factor):
    forward, control = SCREEN.build(factor, "tip_fz", boundary_policy="left_root",
                                    bond_policy="terminal")
    deck, meta = SCREEN.build(factor, "tip_fz", boundary_policy="left_root",
                              bond_policy="terminal", reverse_tip_fz=True)
    assert meta["shared_interface_node_ids"] == control["shared_interface_node_ids"]
    assert meta["detached_stiffener_node_ids"] == control["detached_stiffener_node_ids"]
    assert len(meta["unbonded_interface_pairs"]) == meta["detached_interface_nodes"]
    assert "*NODE PRINT,NSET=UnbondedInterface\nU" in deck
    assert "*NODE PRINT,NSET=UnbondedInterface" not in forward
    applied = {int(node): np.array(force) for node, force in
               meta["applied_nodal_forces_N"].items()}
    assert SCREEN._wrench(applied, meta["probe_coordinates_m"]) == pytest.approx(
        [0., 0., -1000., 0., 1000., 0.], abs=1e-10)
    interface={node:np.zeros(3) for pair in meta["unbonded_interface_pairs"]
               for node in pair}
    for plate, bar in meta["unbonded_interface_pairs"]:
        interface[bar][2]=1e-6
    audit=SCREEN.audit_unbonded_open(meta,interface)
    assert audit["paired_nodes"] == meta["detached_interface_nodes"]
    assert audit["minimum_reference_normal_gap_m"] == pytest.approx(1e-6)
    assert audit["negative_nodal_gaps"] == 0
    interface[meta["unbonded_interface_pairs"][0][1]][2]=-1e-6
    assert SCREEN.audit_unbonded_open(meta,interface)["negative_nodal_gaps"] == 1


def test_reverse_fz_audit_rejects_wrong_scope_and_incomplete_inventory():
    with pytest.raises(ValueError, match="reverse Fz"):
        SCREEN.build(1,"tip_mx",boundary_policy="left_root",
                     bond_policy="terminal",reverse_tip_fz=True)
    _,meta=SCREEN.build(1,"tip_fz",boundary_policy="left_root",
                        bond_policy="terminal",reverse_tip_fz=True)
    with pytest.raises(ValueError,match="incomplete"):
        SCREEN.audit_unbonded_open(meta,{})


@pytest.mark.parametrize("factor", (1, 2, 4))
@pytest.mark.parametrize("load,expected", (("tip_mx", [0.,0.,0.,100.,0.,0.]),
                                            ("tip_fz", [0.,0.,1000.,0.,-1000.,0.])))
def test_detached_bar_control_preserves_mesh_loads_and_separate_roots(factor,load,expected):
    _, terminal=SCREEN.build(factor,load,boundary_policy="left_root",
                             bond_policy="terminal")
    deck,meta=SCREEN.build(factor,load,boundary_policy="left_root",
                            bond_policy="detached")
    assert meta["shared_interface_nodes"] == 0
    assert meta["bond_interval_m"] is None
    assert meta["detached_interface_nodes"] == (5*factor+1)*(2*factor+1)
    assert meta["plate_elements"] == terminal["plate_elements"]
    assert meta["bar_elements"] == terminal["bar_elements"]
    assert set(meta["beam_root_node_ids"]).isdisjoint(meta["plate_boundary_node_ids"])
    assert "*CONTACT PAIR" not in deck and "*TIE" not in deck
    applied={int(node):np.array(force) for node,force in
             meta["applied_nodal_forces_N"].items()}
    assert SCREEN._wrench(applied,meta["probe_coordinates_m"]) == pytest.approx(
        expected,abs=1e-10)
    displacement={int(node):np.zeros(3) for node in meta["probe_node_ids"]}
    assert not SCREEN.recover(meta,displacement)["station_motion"]["1000"]["interface_shared"]


@pytest.mark.parametrize("factor", (1, 2))
@pytest.mark.parametrize("bond", ("detached", "terminal"))
def test_quadratic_solid_preserves_corner_mesh_and_complete_opening_audit(factor,bond):
    linear, old=SCREEN.build(factor,"tip_fz",boundary_policy="left_root",
                            bond_policy=bond,reverse_tip_fz=bond=="terminal")
    deck,meta=SCREEN.build(factor,"tip_fz",boundary_policy="left_root",
                            bond_policy=bond,reverse_tip_fz=bond=="terminal",
                            element_family="C3D20R")
    assert meta["plate_elements"] == old["plate_elements"]
    assert meta["bar_elements"] == old["bar_elements"]
    assert meta["node_count"] > old["node_count"]
    assert deck.count("*ELEMENT,TYPE=C3D20R") == 2
    assert "*CONTACT PAIR" not in deck
    lines=deck.splitlines()
    coords={}
    for line in lines[lines.index("*NODE,NSET=AllNodes")+1:]:
        if line.startswith("*"):
            break
        fields=line.split(",")
        coords[int(fields[0])]=np.array([float(v) for v in fields[1:]])
    start=lines.index("*ELEMENT,TYPE=C3D20R,ELSET=Plate")
    assert len(lines[start+1].split(",")) == 16
    assert len(lines[start+2].split(",")) == 5
    cell=tuple(int(v) for v in
               (*lines[start+1].split(",")[1:], *lines[start+2].split(",")))
    assert len(cell)==20
    for midpoint,(a,b) in zip(cell[8:],SCREEN.HEX20_EDGES):
        assert coords[midpoint] == pytest.approx(
            (coords[cell[a]]+coords[cell[b]])/2,abs=1e-12)
    assert any(node>old["node_count"] for node in meta["beam_root_node_ids"])
    assert any(node>old["node_count"] for node in meta["plate_boundary_node_ids"])
    applied={int(node):np.array(force) for node,force in
             meta["applied_nodal_forces_N"].items()}
    expected=[0.,0.,-1000.,0.,1000.,0.] if bond=="terminal" else [0.,0.,1000.,0.,-1000.,0.]
    assert SCREEN._wrench(applied,meta["probe_coordinates_m"]) == pytest.approx(
        expected,abs=1e-10)
    if bond=="terminal":
        assert meta["detached_interface_nodes"] > old["detached_interface_nodes"]
        assert len(meta["unbonded_interface_pairs"]) == meta["detached_interface_nodes"]
        assert "*NODE PRINT,NSET=UnbondedInterface\nU" in deck
        assert meta["shared_interface_nodes"] > old["shared_interface_nodes"]


def test_quadratic_contact_and_unsupported_scope_are_rejected():
    with pytest.raises(ValueError,match="quadratic screen"):
        SCREEN.build(1,"tip_fz",element_family="C3D20R")
    with pytest.raises(ValueError,match="quadratic screen"):
        SCREEN.build(1,"tip_fz",boundary_policy="left_root",
                     bond_policy="terminal",contact_policy="mortar_hard",
                     element_family="C3D20R")


def test_quadratic_terminal_torque_gap_audit_keeps_the_physical_wrench():
    deck,meta=SCREEN.build(1,"tip_mx",boundary_policy="left_root",
                           bond_policy="terminal",element_family="C3D20R",
                           tip_mx_gap_audit=True)
    assert "*NODE PRINT,NSET=UnbondedInterface\nU" in deck
    assert len(meta["unbonded_interface_pairs"]) == meta["detached_interface_nodes"]
    applied={int(node):np.array(force) for node,force in
             meta["applied_nodal_forces_N"].items()}
    assert SCREEN._wrench(applied,meta["probe_coordinates_m"]) == pytest.approx(
        [0.,0.,0.,100.,0.,0.],abs=1e-10)
    motion={node:np.zeros(3) for pair in meta["unbonded_interface_pairs"]
            for node in pair}
    plate,bar=meta["unbonded_interface_pairs"][0]
    motion[bar][2]=-1e-6
    audit=SCREEN.audit_unbonded_open(meta,motion)
    assert audit["paired_nodes"] == meta["detached_interface_nodes"]
    assert audit["negative_nodal_gaps"] == 1
    assert audit["minimum_reference_normal_gap_m"] == pytest.approx(-1e-6)
    with pytest.raises(ValueError,match="torque gap audit"):
        SCREEN.build(1,"tip_fz",boundary_policy="left_root",
                     bond_policy="terminal",element_family="C3D20R",
                     tip_mx_gap_audit=True)


def test_c1_terminal_solid_sections_follow_native_tangent_without_changing_legacy_deck():
    options=dict(boundary_policy="left_root",bond_policy="terminal",
                 element_family="C3D20R",tip_mx_gap_audit=True,
                 bar_clearance_um=2,plate_thickness_mm=35,
                 tip_load_map="quadratic_face")
    legacy,old=SCREEN.build(4,"tip_mx",**options)
    assert hashlib.sha256(legacy.encode()).hexdigest() == (
        "85c7fd286fbf0363227e3209f9982c2a8beb32508d692f05946c5761b2e60344")
    deck,meta=SCREEN.build(4,"tip_mx",bar_reference_profile="c1_tangent",
                           **options)
    assert meta["bar_reference_profile"] == "C1_TWO_QUADRATIC_TANGENT_ALIGNED_V1"
    assert meta["shared_interface_node_ids"] == old["shared_interface_node_ids"]
    assert meta["applied_nodal_forces_N"] == old["applied_nodal_forces_N"]
    coords={}
    for line in deck.split("*NODE,NSET=AllNodes\n",1)[1].split("\n*",1)[0].splitlines():
        fields=line.split(",")
        coords[int(fields[0])]=np.array([float(v) for v in fields[1:]])

    def offset_and_slope(x):
        t=(x-.75)/.15
        if x<=.75:
            return 2e-6,0.
        if t<=.5:
            return 2e-6*(1-2*t*t),-4*2e-6*t/.15
        return 4e-6*(1-t)**2,-4*2e-6*(1-t)/.15

    checked_midpoint=False
    for plate,bar in meta["unbonded_interface_pairs"]:
        x,y,_=coords[plate]
        offset,slope=offset_and_slope(x)
        normalizer=np.sqrt(1+slope*slope)
        expected=(x+.05*slope/normalizer,y,
                  .0675+offset-.05/normalizer)
        np.testing.assert_allclose(coords[bar],expected,rtol=0.,atol=2e-12)
        if abs(x-.825)<1e-12 and abs(y)<1e-12:
            assert abs(coords[bar][0]-x)>1e-6
            checked_midpoint=True
    assert checked_midpoint
    for node in meta["shared_interface_node_ids"]:
        assert coords[node][2] == pytest.approx(.0175,abs=1e-12)
    with_u,with_u_meta=SCREEN.build(
        4,"tip_mx",bar_reference_profile="c1_tangent",
        full_displacement_output=True,**options)
    assert with_u.replace("*NODE FILE,OUTPUT=3D\nU\n","") == deck
    assert with_u_meta["full_displacement_output"] is True
    with pytest.raises(ValueError,match="C1 tangent profile"):
        SCREEN.build(4,"tip_mx",bar_reference_profile="c1_tangent",
                     **{**options,"bar_clearance_um":0})


@pytest.mark.parametrize("load,reverse,expected", (
    ("tip_mx",False,[0.,0.,0.,100.,0.,0.]),
    ("tip_fz",True,[0.,0.,-1000.,0.,1000.,0.])))
def test_c1_detached_section_control_keeps_geometry_load_and_independent_roots(
        load, reverse, expected):
    options=dict(boundary_policy="left_root",bond_policy="detached",
                 contact_policy="none",element_family="C3D20R",
                 bar_clearance_um=2,plate_thickness_mm=35,
                 tip_load_map="quadratic_face",full_displacement_output=True,
                 bar_reference_profile="c1_tangent",reverse_tip_fz=reverse)
    deck,meta=SCREEN.build(4,load,**options)
    assert meta["shared_interface_nodes"] == 0
    assert meta["bond_interval_m"] is None
    assert set(meta["beam_root_node_ids"]).isdisjoint(
        meta["plate_boundary_node_ids"])
    assert set(meta["boundary_node_ids"]) == (
        set(meta["beam_root_node_ids"]) |
        set(meta["plate_boundary_node_ids"]))
    assert "*CONTACT PAIR" not in deck and "*TIE" not in deck
    assert "*NODE FILE,OUTPUT=3D\nU\n" in deck
    applied={int(node):np.array(force) for node,force in
             meta["applied_nodal_forces_N"].items()}
    assert SCREEN._wrench(applied,meta["probe_coordinates_m"]) == pytest.approx(
        expected,abs=1e-9)
    with pytest.raises(ValueError,match="bounded full-bond"):
        SCREEN.build(2,load,**options)


def test_c1_penalty_contact_preserves_the_matched_torque_geometry_and_load():
    options=dict(boundary_policy="left_root", bond_policy="terminal",
                 element_family="C3D20R", tip_mx_gap_audit=True,
                 bar_clearance_um=2, plate_thickness_mm=35,
                 tip_load_map="quadratic_face",
                 full_displacement_output=True,
                 bar_reference_profile="c1_tangent")
    baseline, original = SCREEN.build(4, "tip_mx", **options)
    assert hashlib.sha256(baseline.replace("\n", "\r\n").encode("ascii")).hexdigest() == (
        "cc88870f0664cbadb39a114f4755dc996c226ab683d3304ea8f4a60f26b93da0")
    contact, candidate = SCREEN.build(
        4, "tip_mx", contact_policy="c1_penalty", **options)
    nodes = lambda deck: deck.split("*NODE,NSET=AllNodes\n", 1)[1].split(
        "\n*ELEMENT", 1)[0]
    assert nodes(contact) == nodes(baseline)
    for key in ("node_count", "element_family", "plate_elements",
                "bar_elements", "shared_interface_node_ids",
                "boundary_node_ids", "applied_nodal_forces_N",
                "unbonded_face_pairs", "bar_reference_profile"):
        assert candidate[key] == original[key]
    assert len(candidate["plate_contact_face_element_ids"]) == 128
    assert len(candidate["bar_contact_face_element_ids"]) == 128
    assert "*SURFACE BEHAVIOR,PRESSURE-OVERCLOSURE=LINEAR\n4.2e14,1" in contact
    assert "*CONTACT PAIR,INTERACTION=PenaltyUnbonded,TYPE=SURFACE TO SURFACE" in contact
    assert "*CONTACT PRINT,FREQUENCY=9999\nCDIS,CSTR,CNUM" in contact
    with pytest.raises(ValueError, match="bounded full-bond or C1 terminal"):
        SCREEN.build(4, "tip_fz", contact_policy="c1_penalty", **options)


def test_c1_opening_shear_keeps_torque_geometry_and_has_exact_reverse_wrench():
    common=dict(boundary_policy="left_root", bond_policy="terminal",
                element_family="C3D20R", bar_clearance_um=2,
                plate_thickness_mm=35, tip_load_map="quadratic_face",
                full_displacement_output=True,
                bar_reference_profile="c1_tangent")
    torque, torque_meta = SCREEN.build(
        4, "tip_mx", tip_mx_gap_audit=True, **common)
    shear, shear_meta = SCREEN.build(
        4, "tip_fz", reverse_tip_fz=True, **common)
    nodes_and_elements = lambda deck: deck.split("*NODE,NSET=AllNodes\n", 1)[1].split(
        "\n*NSET", 1)[0]
    assert nodes_and_elements(shear) == nodes_and_elements(torque)
    for key in ("shared_interface_node_ids", "unbonded_interface_pairs",
                "boundary_node_ids", "bar_reference_profile"):
        assert shear_meta[key] == torque_meta[key]
    applied={int(node):np.array(force) for node,force in
             shear_meta["applied_nodal_forces_N"].items()}
    assert SCREEN._wrench(applied, shear_meta["probe_coordinates_m"]) == pytest.approx(
        [0.,0.,-1000.,0.,1000.,0.], abs=1e-9)
    assert "*CONTACT PAIR" not in shear
    assert "*NODE FILE,OUTPUT=3D\nU" in shear
    assert hashlib.sha256(shear.replace("\n", "\r\n").encode("ascii")).hexdigest() == (
        "49b6e57dd3654048b7ba266365e9dd7be6898340589535fda6ca108c16efc76b")
    coarse, coarse_meta = SCREEN.build(
        2, "tip_fz", reverse_tip_fz=True, **common)
    assert len(coarse_meta["unbonded_face_pairs"]) == 32
    assert len(coarse_meta["shared_interface_node_ids"]) > 0
    assert "*NODE FILE,OUTPUT=3D\nU" in coarse
    coarse_applied = {int(node):np.array(force) for node,force in
                      coarse_meta["applied_nodal_forces_N"].items()}
    assert SCREEN._wrench(
        coarse_applied, coarse_meta["probe_coordinates_m"]) == pytest.approx(
            [0.,0.,-1000.,0.,1000.,0.], abs=1e-9)
    with pytest.raises(ValueError, match="bounded full-bond or C1 terminal"):
        SCREEN.build(1, "tip_fz", reverse_tip_fz=True, **common)
    with pytest.raises(ValueError, match="displacement output required"):
        SCREEN.build(4, "tip_mx", reverse_tip_fz=True, **common)


def test_c1_factor_two_torque_basis_matches_opening_structural_deck():
    common=dict(boundary_policy="left_root", bond_policy="terminal",
                element_family="C3D20R", bar_clearance_um=2,
                plate_thickness_mm=35, tip_load_map="quadratic_face",
                full_displacement_output=True,
                bar_reference_profile="c1_tangent")
    torque, tm = SCREEN.build(2, "tip_mx", tip_mx_gap_audit=True, **common)
    opening, om = SCREEN.build(2, "tip_fz", reverse_tip_fz=True, **common)
    structural = lambda deck: "\n".join(
        deck.split("*CLOAD", 1)[0].splitlines()[2:])
    assert structural(torque) == structural(opening)
    assert tm["unbonded_face_pairs"] == om["unbonded_face_pairs"]
    assert len(tm["unbonded_face_pairs"]) == 32
    assert tm["boundary_node_ids"] == om["boundary_node_ids"]
    assert "*NODE FILE,OUTPUT=3D\nU" in torque
    forces={int(node):np.array(force) for node,force in
            tm["applied_nodal_forces_N"].items()}
    assert SCREEN._wrench(forces, tm["probe_coordinates_m"]) == pytest.approx(
        [0.,0.,0.,100.,0.,0.], abs=1e-9)
    with pytest.raises(ValueError, match="bounded full-bond or C1 terminal"):
        SCREEN.build(1, "tip_mx", tip_mx_gap_audit=True, **common)
    with pytest.raises(ValueError, match="bounded full-bond or C1 terminal"):
        SCREEN.build(2, "tip_mx", tip_mx_gap_audit=True,
                     contact_policy="c1_penalty", **common)


@pytest.mark.parametrize("factor", (2, 4, 5))
def test_c1_combined_opening_torque_load_is_exact_sum_and_not_pure_tip_metric(factor):
    common=dict(boundary_policy="left_root", bond_policy="terminal",
                element_family="C3D20R", bar_clearance_um=2,
                plate_thickness_mm=35, tip_load_map="quadratic_face",
                full_displacement_output=True,
                bar_reference_profile="c1_tangent")
    torque, tm = SCREEN.build(factor, "tip_mx", tip_mx_gap_audit=True, **common)
    opening, om = SCREEN.build(factor, "tip_fz", reverse_tip_fz=True, **common)
    combined, cm = SCREEN.build(
        factor, "tip_mx", tip_mx_gap_audit=True,
        torque_with_opening_fz=True, **common)
    structural = lambda deck: "\n".join(
        deck.split("*CLOAD", 1)[0].splitlines()[2:])
    assert structural(torque) == structural(opening) == structural(combined)
    assert cm["torque_with_opening_fz"] is True
    assert cm["unbonded_face_pairs"] == tm["unbonded_face_pairs"]
    nodes = set(tm["applied_nodal_forces_N"])
    assert nodes == set(om["applied_nodal_forces_N"]) == set(cm["applied_nodal_forces_N"])
    for node in nodes:
        assert np.array(cm["applied_nodal_forces_N"][node]) == pytest.approx(
            np.array(tm["applied_nodal_forces_N"][node])+
            np.array(om["applied_nodal_forces_N"][node]), abs=1e-11)
    forces={int(node):np.array(force) for node,force in
            cm["applied_nodal_forces_N"].items()}
    assert SCREEN._wrench(forces, cm["probe_coordinates_m"]) == pytest.approx(
        [0.,0.,-1000.,100.,1000.,0.], abs=1e-9)
    zero = {int(node):np.zeros(3) for node in cm["probe_node_ids"]}
    response = SCREEN.recover(cm, zero)
    assert response["applied_load_work_j"] == 0.
    assert "tip_work_conjugate_displacement" not in response
    assert "mixed opening-force and torque work" in response["applied_load_work_meaning"]
    with pytest.raises(ValueError, match="combined C1 opening/torque"):
        SCREEN.build(factor, "tip_fz", reverse_tip_fz=True,
                     torque_with_opening_fz=True, **common)


def test_c1_factor_five_preserves_matched_model_and_transition_knot():
    common=dict(boundary_policy="left_root", bond_policy="terminal",
                element_family="C3D20R", bar_clearance_um=2,
                plate_thickness_mm=35, tip_load_map="quadratic_face",
                full_displacement_output=True,
                bar_reference_profile="c1_tangent",
                tip_mx_gap_audit=True)
    deck, meta = SCREEN.build(5, "tip_mx", torque_with_opening_fz=True,
                              **common)
    assert meta["factor"] == 5
    assert meta["bond_interval_m"] == [0.9, 1.0]
    assert meta["torque_with_opening_fz"] is True
    assert "450,0" in meta["surface_nodes"]
    assert "550,0" in meta["surface_nodes"]
    node_block = deck.split("*NODE,NSET=AllNodes\n", 1)[1].split("\n*", 1)[0]
    assert ",0.825," in node_block
    forces={int(node):np.array(force) for node,force in
            meta["applied_nodal_forces_N"].items()}
    assert SCREEN._wrench(forces, meta["probe_coordinates_m"]) == pytest.approx(
        [0.,0.,-1000.,100.,1000.,0.], abs=1e-9)
    assert "*NODE FILE,OUTPUT=3D\nU\n" in deck
    with pytest.raises(ValueError, match="factor-five solid mesh"):
        SCREEN.build(5, "tip_mx", tip_mx_gap_audit=True,
                     torque_with_opening_fz=False,
                     **{key: value for key, value in common.items()
                        if key not in ("full_displacement_output",
                                       "tip_mx_gap_audit")})


def test_c1_penalty_runner_allows_only_its_known_static_notice():
    notice = " *WARNING reading *STATIC: a nonlinear analysis is requested"
    other = " *WARNING in contact: face search failed"
    assert SCREEN.unexpected_solver_diagnostics(
        notice, c1_penalty=True) == []
    assert SCREEN.unexpected_solver_diagnostics(
        notice, c1_penalty=False) == [notice.strip()]
    assert SCREEN.unexpected_solver_diagnostics(
        notice+"\n"+other, c1_penalty=True) == [other.strip()]


def test_smooth_terminal_clearance_admits_shared_native_b3_frames():
    """The native Q2 frame guard admits the C1 reference used by the solid."""
    source=MODULE.parents[1]/"src"/"anyfesolver"/"ge_beam3_curved_reference.py"
    spec=importlib.util.spec_from_file_location("f03_curved_reference_source",source)
    curved=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=curved
    spec.loader.exec_module(curved)
    clearance=2e-6
    transition=.15

    def offset(x):
        t=(x-.75)/transition
        if t<=.5:
            return clearance*(1-2*t*t),-4*clearance*t/transition
        return 2*clearance*(1-t)**2,-4*clearance*(1-t)/transition

    def span(a,b):
        abscissae=np.linspace(a,b,3)
        coordinates=np.array([
            (x,0.,.0675+offset(x)[0]) for x in abscissae])
        frames=[]
        for x in abscissae:
            tangent=np.array((1.,0.,offset(x)[1]))
            tangent/=np.linalg.norm(tangent)
            transverse=np.array((0.,1.,0.))
            frames.append(np.column_stack((
                tangent,transverse,np.cross(tangent,transverse))))
        return curved.CurvedBeam3ReferenceGeometry(
            coordinates,np.array(frames))

    left,right=span(.75,.825),span(.825,.9)
    np.testing.assert_allclose(left.position(1.),right.position(-1.),
                               rtol=0.,atol=1e-14)
    np.testing.assert_allclose(left.nodal_triads[-1],right.nodal_triads[0],
                               rtol=0.,atol=1e-14)
    np.testing.assert_allclose(
        left.derivative(1.)/left.jacobian(1.),
        right.derivative(-1.)/right.jacobian(-1.),
        rtol=0.,atol=1e-13)
    np.testing.assert_allclose(left.nodal_triads[0],np.eye(3),
                               rtol=0.,atol=1e-13)
    np.testing.assert_allclose(right.nodal_triads[-1],np.eye(3),
                               rtol=0.,atol=1e-13)
    bonded_bottom=right.position(1.)+right.nodal_triads[-1]@(0.,0.,-.05)
    np.testing.assert_allclose(bonded_bottom,(.9,0.,.0175),
                               rtol=0.,atol=1e-13)
    bent_bottom=left.position(1.)+left.nodal_triads[-1]@(0.,0.,-.05)
    assert abs(bent_bottom[0]-.825)>1e-6


def test_c1_terminal_solid_bar_has_positive_quadratic_gauss_jacobians():
    deck,_=SCREEN.build(
        4,"tip_mx",boundary_policy="left_root",bond_policy="terminal",
        element_family="C3D20R",tip_mx_gap_audit=True,
        bar_clearance_um=2,plate_thickness_mm=35,
        tip_load_map="quadratic_face",bar_reference_profile="c1_tangent")
    lines=deck.splitlines()
    coords={}
    for line in lines[lines.index("*NODE,NSET=AllNodes")+1:]:
        if line.startswith("*"):
            break
        fields=line.split(",")
        coords[int(fields[0])]=np.array([float(v) for v in fields[1:]])
    start=lines.index("*ELEMENT,TYPE=C3D20R,ELSET=Stiffener")+1
    cells=[]
    for line in range(start,len(lines),2):
        if lines[line].startswith("*"):
            break
        first=[int(v) for v in lines[line].split(",")]
        second=[int(v) for v in lines[line+1].split(",")]
        cells.append(first[1:]+second)
    xyz=np.array([[coords[node] for node in cell] for cell in cells])
    assert len(cells)==1280

    corners=np.array(((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                      (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)))

    def shape(q):
        values=[np.prod(1+q*v)*(q@v-2)/8 for v in corners]
        for a,b in SCREEN.HEX20_EDGES:
            first,last=corners[a],corners[b]
            axis=int(np.flatnonzero(first!=last)[0])
            others=[index for index in range(3) if index!=axis]
            values.append((1-q[axis]**2)*
                          (1+q[others[0]]*first[others[0]])*
                          (1+q[others[1]]*first[others[1]])/4)
        return np.array(values)

    gauss=np.sqrt(3/5)
    gradients=[]
    for r in (-gauss,0.,gauss):
        for s in (-gauss,0.,gauss):
            for t in (-gauss,0.,gauss):
                point=np.array((r,s,t))
                assert abs(shape(point).sum()-1)<1e-13
                components=[]
                for axis in range(3):
                    perturbed=point.astype(complex)
                    perturbed[axis]+=1e-30j
                    components.append(np.imag(shape(perturbed))/1e-30)
                gradients.append(np.stack(components,axis=1))
    gradients=np.array(gradients)
    assert np.max(np.abs(gradients.sum(axis=1)))<1e-13
    jacobians=np.einsum("eik,gij->egkj",xyz,gradients)
    determinants=np.linalg.det(jacobians)
    assert np.isfinite(determinants).all()
    assert np.min(determinants)>0.


@pytest.mark.parametrize("clearance_um", (1, 2))
def test_declared_terminal_clearance_is_positive_and_tapers_to_shared_patch(clearance_um):
    deck,meta=SCREEN.build(2,"tip_mx",boundary_policy="left_root",
                           bond_policy="terminal",element_family="C3D20R",
                           tip_mx_gap_audit=True,bar_clearance_um=clearance_um)
    pairs=meta["unbonded_interface_pairs"]
    gaps=meta["unbonded_initial_gaps_m"]
    assert len(gaps)==len(pairs)
    assert min(gaps)>0
    assert max(gaps)==pytest.approx(clearance_um*1e-6,abs=1e-15)
    assert len(set(round(g,14) for g in gaps))>1
    assert meta["shared_interface_nodes"]>0
    assert "*CONTACT PAIR" not in deck
    applied={int(node):np.array(force) for node,force in
             meta["applied_nodal_forces_N"].items()}
    assert SCREEN._wrench(applied,meta["probe_coordinates_m"]) == pytest.approx(
        [0.,0.,0.,100.,0.,0.],abs=1e-10)
    motion={node:np.zeros(3) for pair in pairs for node in pair}
    audit=SCREEN.audit_unbonded_open(meta,motion)
    assert audit["negative_nodal_gaps"]==0
    assert audit["minimum_reference_normal_gap_m"]==pytest.approx(min(gaps))
    assert audit["sampled_face_count"]>0
    assert audit["sampled_face_points_per_face"]==81
    assert audit["sampled_face_negative_points"]==0
    assert audit["sampled_face_minimum_reference_normal_gap_m"]>0
    probes={node:np.zeros(3) for node in meta["probe_node_ids"]}
    recovered=SCREEN.recover(meta,probes)
    assert recovered["unbonded_midspan_normal_separation_m"][
        "bar_minus_plate_z_m"] == pytest.approx([clearance_um*1e-6]*3,abs=1e-15)
    with pytest.raises(ValueError,match="declared bar clearance"):
        SCREEN.build(1,"tip_mx",boundary_policy="left_root",
                     bond_policy="terminal",element_family="C3D8",
                     tip_mx_gap_audit=True,bar_clearance_um=clearance_um)


def test_quadratic_face_shape_has_partition_and_nodal_interpolation():
    nodes=((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.),
           (0.,-1.),(1.,0.),(0.,1.),(-1.,0.))
    for i,(xi,eta) in enumerate(nodes):
        expected=np.zeros(8)
        expected[i]=1.
        assert SCREEN._quad8_shape(xi,eta)==pytest.approx(expected,abs=1e-15)
    for xi,eta in ((0.,0.),(.4,-.6),(-.9,.8)):
        assert sum(SCREEN._quad8_shape(xi,eta))==pytest.approx(1.,abs=1e-15)


def test_quadratic_face_audit_finds_negative_interior_with_positive_nodes():
    pairs=[(i,i+8) for i in range(1,9)]
    positive=[1e-6]*4+[.1e-6]*4
    meta={"unbonded_interface_pairs":pairs,
          "unbonded_initial_gaps_m":positive,
          "unbonded_face_pairs":[pairs],
          "unbonded_face_initial_gaps_m":[positive]}
    motion={node:np.zeros(3) for pair in pairs for node in pair}
    audit=SCREEN.audit_unbonded_open(meta,motion)
    assert audit["negative_nodal_gaps"]==0
    assert audit["sampled_face_negative_points"]>0
    assert audit["sampled_face_minimum_reference_normal_gap_m"]<0


@pytest.mark.parametrize("factor", (1, 2, 4))
def test_terminal_hard_mortar_faces_exclude_bonded_patch(factor):
    deck, meta = SCREEN.build(factor, "tip_fz", boundary_policy="left_root",
                              bond_policy="terminal", contact_policy="mortar_hard")
    assert meta["contact_policy"] == "mortar_hard"
    assert len(meta["plate_contact_face_element_ids"]) == 8*factor**2
    assert len(meta["bar_contact_face_element_ids"]) == 8*factor**2
    assert max(meta["plate_contact_face_element_ids"]) <= meta["plate_elements"]
    assert min(meta["bar_contact_face_element_ids"]) > meta["plate_elements"]
    assert "PlateUnbondedFaces,S2" in deck
    assert "BarUnbondedFaces,S1" in deck
    assert "*CONTACT PAIR,INTERACTION=HardUnbonded,TYPE=MORTAR" in deck
    assert "*SURFACE INTERACTION,NAME=HardUnbonded\n*CONTACT PAIR" in deck
    assert "*SURFACE BEHAVIOR" not in deck and "ADJUST" not in deck
    assert "*CONTACT PRINT,FREQUENCY=9999\nCDIS,CSTR,CNUM" in deck
    without, _ = SCREEN.build(factor, "tip_fz", boundary_policy="left_root",
                              bond_policy="terminal")
    assert "*CONTACT PAIR" not in without


def test_contact_requires_separate_terminal_bond():
    with pytest.raises(ValueError, match="requires terminal bond"):
        SCREEN.build(1, "tip_fz", boundary_policy="left_root",
                     contact_policy="mortar_hard")


def test_recovery_rigid_motion_and_orthogonal_shear_measures():
    _, meta = SCREEN.build(1, "uy_x")
    coordinates = meta["probe_coordinates_m"]
    for sign in (1., -1.):
        motion = {int(node): np.array([-.4e-4 * xyz[1],
                                       .4e-4 * xyz[0], 0.]) * sign
                  for node, xyz in coordinates.items()}
        result = SCREEN.recover(meta, motion)
        assert result["centerline_tangent_yaw_rad"] == pytest.approx(sign * .4e-4)
        assert result["surface_square_patch_yaw_rad"] == pytest.approx(sign * .4e-4)
        for station in result["station_motion"].values():
            assert station["section_rotation_rad"][2] == pytest.approx(sign * .4e-4)
            assert station["max_warping_residual_m"] < 1e-17


def test_probe_parser_requires_complete_unique_finite_inventory():
    dat = "displacements (vx,vy,vz) for set PROBE and time 0.1000000E+01\n\n"
    dat += "  1  0.0  1.0E-5  0.0\n  2  0.0  2.0E-5  0.0\n\n"
    values = SCREEN.parse_probe_displacements(dat, {1, 2})
    assert values[2][1] == pytest.approx(2e-5)
    with pytest.raises(ValueError, match="incomplete"):
        SCREEN.parse_probe_displacements(dat, {1, 2, 3})


@pytest.mark.parametrize("factor", (1, 2, 4))
@pytest.mark.parametrize("load,expected", (("tip_mx", [0.,0.,0.,100.,0.,0.]),
                                            ("tip_fz", [0.,0.,1000.,0.,-1000.,0.])))
def test_end_face_load_has_exact_declared_wrench(factor, load, expected):
    deck,meta=SCREEN.build(factor,load)
    assert "*CLOAD" in deck
    applied={int(node):np.array(force) for node,force in
             meta["applied_nodal_forces_N"].items()}
    assert len(applied)==(2*factor+1)**2
    assert SCREEN._wrench(applied,meta["probe_coordinates_m"]) == pytest.approx(expected,abs=1e-10)


@pytest.mark.parametrize("factor", (1, 4))
@pytest.mark.parametrize("load", ("tip_mx", "tip_fz"))
def test_left_edge_reference_preserves_end_wrench_and_free_stiffener(factor, load):
    _, meta = SCREEN.build(factor, load, boundary_policy="left")
    boundary = meta["boundary_coordinates_m"]
    assert meta["boundary_policy"] == "left"
    assert len(boundary) == (6 * factor + 1) * (2 * factor + 1)
    assert all(xyz[0] == -.5 for xyz in boundary.values())
    assert set(meta["boundary_node_ids"]).isdisjoint(
        meta["bar_station_nodes"]["0"])
    applied = {int(node): np.array(force) for node, force in
               meta["applied_nodal_forces_N"].items()}
    expected = ([0., 0., 0., 100., 0., 0.] if load == "tip_mx" else
                [0., 0., 1000., 0., -1000., 0.])
    assert SCREEN._wrench(applied, meta["probe_coordinates_m"]) == pytest.approx(
        expected, abs=1e-10)


@pytest.mark.parametrize("factor", (1, 4))
def test_left_edge_and_root_face_support_are_explicit(factor):
    deck, meta = SCREEN.build(factor, "tip_fz", boundary_policy="left_root")
    plate = set(meta["plate_boundary_node_ids"])
    root = set(meta["beam_root_node_ids"])
    support = set(meta["boundary_node_ids"])
    assert len(root) == (2 * factor + 1) ** 2
    assert root == set(meta["bar_station_nodes"]["0"])
    assert support == plate | root
    assert root - plate
    assert meta["support_set"] == "Supports"
    assert "*NODE PRINT,NSET=Supports" in deck
    assert "*MPC" not in deck and "*TIE" not in deck


@pytest.mark.parametrize("load", ("tip_mx", "tip_fz"))
def test_end_face_recovery_is_conjugate_to_actual_discrete_load(load):
    _, meta = SCREEN.build(1, load, boundary_policy="left_root")
    amplitude = 2.5e-5
    displacement = {}
    for node, xyz in meta["probe_coordinates_m"].items():
        _, y, z = xyz
        displacement[int(node)] = (
            np.array([0., -(z-.085)*amplitude, y*amplitude])
            if load == "tip_mx" else np.array([0., 0., amplitude]))
    result = SCREEN.recover(meta, displacement)
    measure = result["tip_work_conjugate_displacement"]
    assert measure["value"] == pytest.approx(amplitude, abs=1e-15)
    assert measure["unit"] == ("rad" if load == "tip_mx" else "m")


def test_shared_face_trace_distinguishes_rigid_twist_from_section_warping():
    _, meta = SCREEN.build(4, "tip_mx", boundary_policy="left_root")
    twist = 1e-4
    rigid = {int(node): np.array([0., -(xyz[2]-.085)*twist,
                                  xyz[1]*twist])
             for node, xyz in meta["probe_coordinates_m"].items()}
    station = SCREEN.recover(meta, rigid)["station_motion"]["500"]
    assert np.linalg.norm(station["shared_face_minus_fitted_bottom_m"]) < 1e-15
    assert abs(station["shared_face_slope_minus_section_rx_rad"]) < 1e-14
    warped = {node: value + np.array([0., 0., .01 *
                                     meta["probe_coordinates_m"][str(node)][1] ** 3])
              for node, value in rigid.items()}
    station = SCREEN.recover(meta, warped)["station_motion"]["500"]
    assert abs(station["shared_face_slope_minus_section_rx_rad"]) > 1e-6
    assert station["max_warping_residual_m"] > 1e-8


def test_midspan_transverse_slope_preserves_independent_shear_field():
    _, meta = SCREEN.build(4, "tip_fz", boundary_policy="left_root")
    rotation = 1e-4
    rigid = {int(node): np.array([(xyz[2]-.085)*rotation, 0.,
                                  -xyz[0]*rotation])
             for node, xyz in meta["probe_coordinates_m"].items()}
    assert abs(SCREEN.recover(meta, rigid)[
        "midspan_transverse_kinematics"]["slope_plus_section_ry_rad"]) < 1e-14
    shear = 2e-5
    sheared = {node: value + np.array([0., 0., shear *
                                      meta["probe_coordinates_m"][str(node)][0]])
               for node, value in rigid.items()}
    measured = SCREEN.recover(meta, sheared)["midspan_transverse_kinematics"]
    assert measured["slope_plus_section_ry_rad"] == pytest.approx(shear, abs=1e-13)


def test_left_root_support_split_uses_common_origin_and_complete_inventory():
    meta = {"boundary_policy": "left_root",
            "plate_boundary_node_ids": [1], "beam_root_node_ids": [2],
            "boundary_coordinates_m": {"1": [-.5, .2, 0.],
                                      "2": [0., 0., .0675]}}
    reaction = {1: np.array([0., 0., -10.]),
                2: np.array([0., 4., 0.])}
    split = SCREEN.split_left_root_support_wrenches(meta, reaction)
    assert split["plate_support_wrench_N_Nm"] == pytest.approx(
        [0., 0., -10., -2., -5., 0.])
    assert split["bar_root_support_wrench_N_Nm"] == pytest.approx(
        [0., 4., 0., -.27, 0., 0.])
    assert split["total_support_wrench_N_Nm"] == pytest.approx(
        [0., 4., -10., -2.27, -5., 0.])
    with pytest.raises(ValueError, match="disjoint complete"):
        SCREEN.split_left_root_support_wrenches(meta, {1: reaction[1]})
