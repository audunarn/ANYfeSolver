"""Bounded 3D solid plate/stiffener bond discriminator, not G0 acceptance.

The plate and finite square stiffener share the declared bonded interface
nodes. Contact is absent by default; a bounded external-reference option
adds finite-stiffness penalty contact on the unbonded C1 torque faces.
The two imposed plate-edge shears distinguish a line-tangent rotation from
the rotation of a finite bonded region at a free stiffener termination.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np


WORKTREE = Path(__file__).resolve().parents[1]
EXE = Path("C:/Github/ANYfeSolver/.qualification/b3-ge-e4-pl/"
           "calculix-223/CalculiX-2.23.0-win-x64/bin/ccx.exe")
EXE_SHA = "27a9426a1ee2f72c38fbe892e0ffaf38e14867628ffd0235770261b6f3ec9c5f"
TIMEOUT_SECONDS = 120
GAMMA = 1e-4
PLATE_X_MM = (-500, -250, 0, 250, 500, 750, 1000, 1250, 1500)
PLATE_Y_MM = (-500, -250, -50, 0, 50, 250, 500)
PLATE_Z_MM = (-35, 0, 35)
BAR_X_MM = (0, 250, 500, 750, 1000)
BAR_Y_MM = (-50, 0, 50)
BAR_Z_MM = (35, 85, 135)
STATIONS_MM = (0, 250, 500, 750, 1000)
HEX20_EDGES = ((0,1),(1,2),(2,3),(3,0),
               (4,5),(5,6),(6,7),(7,4),
               (0,4),(1,5),(2,6),(3,7))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unexpected_solver_diagnostics(text: str, *, c1_penalty: bool) -> list[str]:
    """Keep every solver warning except the known contact *STATIC notice."""
    static_notice = "*WARNING reading *STATIC: a nonlinear analysis is requested"
    return [line.strip() for line in text.splitlines()
            if re.search(r"(?i)\b(warning|error|fatal)\b", line) and
            not (c1_penalty and line.strip() == static_notice)]


def _cuts(values: tuple[int | Fraction, ...], factor: int) -> tuple[Fraction, ...]:
    if type(factor) is not int or factor not in (1, 2, 4, 5):
        raise ValueError("bounded solid mesh factor must be 1, 2, 4, or 5")
    return tuple(Fraction(a)+(Fraction(b-a)*k)/factor
                 for a,b in zip(values[:-1],values[1:])
                 for k in range(factor))+(Fraction(values[-1]),)


def _number(value: float) -> str:
    # CalculiX input fields are bounded; these coordinates are rational mm cuts.
    return f"{value:.12g}"


def _set_lines(indices: set[int] | list[int]) -> list[str]:
    ordered=sorted(indices)
    return [",".join(str(v) for v in ordered[i:i+12])
            for i in range(0,len(ordered),12)]


def build(factor: int, shear: str, *, boundary_policy: str = "perimeter",
          bond_policy: str = "full", contact_policy: str = "none",
          reverse_tip_fz: bool = False, element_family: str = "C3D8",
          tip_mx_gap_audit: bool = False, bar_clearance_um: int = 0,
          plate_thickness_mm: int = 70, tip_load_map: str = "legacy",
          full_displacement_output: bool = False,
          bar_reference_profile: str = "legacy_linear",
          torque_with_opening_fz: bool = False,
          sided_port_cut: bool = False,
          ) -> tuple[str, dict]:
    """Return an exact-coordinate brick deck and explicit bond identity."""
    if shear not in ("uy_x", "ux_y", "tip_mx", "tip_fz"):
        raise ValueError("undeclared solid-bond loading")
    c1_open_shear = (shear == "tip_fz" and reverse_tip_fz and
                     bar_reference_profile == "c1_tangent" and
                     contact_policy == "none")
    c1_torque_basis = (shear == "tip_mx" and tip_mx_gap_audit and
                       bar_reference_profile == "c1_tangent" and
                       contact_policy == "none")
    if type(sided_port_cut) is not bool or (sided_port_cut and not (
            factor == 5 and c1_torque_basis and torque_with_opening_fz and
            boundary_policy == "left_root" and bond_policy == "terminal" and
            element_family == "C3D20R" and bar_clearance_um == 2 and
            plate_thickness_mm == 35 and tip_load_map == "quadratic_face" and
            full_displacement_output)):
        raise ValueError("sided port cut requires the bounded F5 C1 combined fixture")
    c1_detached = (factor == 4 and shear in ("tip_mx", "tip_fz") and
                   boundary_policy == "left_root" and
                   bond_policy == "detached" and contact_policy == "none" and
                   element_family == "C3D20R" and
                   bar_reference_profile == "c1_tangent" and
                   bar_clearance_um == 2 and plate_thickness_mm == 35 and
                   tip_load_map == "quadratic_face" and
                   full_displacement_output)
    if type(torque_with_opening_fz) is not bool or (
            torque_with_opening_fz and not (
                c1_torque_basis and factor in (2, 4, 5) and
                boundary_policy == "left_root" and bond_policy == "terminal" and
                element_family == "C3D20R" and bar_clearance_um == 2 and
                plate_thickness_mm == 35 and
                tip_load_map == "quadratic_face" and
                full_displacement_output)):
        raise ValueError("combined C1 opening/torque requires bounded no-contact full-field case")
    if type(full_displacement_output) is not bool or (
            full_displacement_output and not (
                (factor in (4, 5) or
                 (factor == 2 and (c1_open_shear or c1_torque_basis))) and
                shear in ("tip_mx", "tip_fz") and
                boundary_policy == "left_root" and
                contact_policy in ("none", "c1_penalty") and
                element_family == "C3D20R" and
                plate_thickness_mm == 35 and
                tip_load_map == "quadratic_face" and
                (bond_policy == "full" or
                 (bond_policy == "terminal" and
                  ((shear == "tip_mx" and tip_mx_gap_audit) or
                   c1_open_shear) and
                  bar_clearance_um == 2 and
                  bar_reference_profile == "c1_tangent") or
                 c1_detached))):
        raise ValueError("bounded full-bond or C1 terminal displacement output required")
    if type(plate_thickness_mm) is not int or plate_thickness_mm not in (35,70):
        raise ValueError("bounded 35/70 mm plate thickness required")
    plate_top=Fraction(plate_thickness_mm,2)
    plate_z=(-plate_top,Fraction(0),plate_top)
    bar_z=(plate_top,plate_top+50,plate_top+100)
    bar_axis_z_m=float(plate_top+50)/1000
    if boundary_policy not in ("perimeter", "left", "left_root") or (
            boundary_policy != "perimeter" and shear not in ("tip_mx", "tip_fz")):
        raise ValueError("unsupported plate boundary/loading combination")
    if bond_policy not in ("full", "terminal", "detached") or (
            bond_policy in ("terminal", "detached") and
            (boundary_policy != "left_root" or
             shear not in ("tip_mx", "tip_fz"))):
        raise ValueError("terminal/detached bond requires left/root tip-load case")
    if contact_policy not in ("none", "mortar_hard", "c1_penalty") or (
            contact_policy == "mortar_hard" and bond_policy != "terminal"):
        raise ValueError("hard mortar contact requires terminal bond")
    if contact_policy == "c1_penalty" and not (
            factor == 4 and shear == "tip_mx" and
            boundary_policy == "left_root" and bond_policy == "terminal" and
            element_family == "C3D20R" and tip_mx_gap_audit and
            bar_clearance_um == 2 and plate_thickness_mm == 35 and
            tip_load_map == "quadratic_face" and full_displacement_output and
            bar_reference_profile == "c1_tangent"):
        raise ValueError("C1 penalty contact requires the matched bounded torque case")
    if reverse_tip_fz and (shear != "tip_fz" or
                           (bond_policy != "terminal" and not c1_detached) or
                           boundary_policy != "left_root" or
                           contact_policy != "none"):
        raise ValueError("reverse Fz opening audit requires no-contact terminal tip load")
    if tip_mx_gap_audit and (shear != "tip_mx" or bond_policy != "terminal" or
                             boundary_policy != "left_root" or
                             contact_policy not in ("none", "c1_penalty") or
                             reverse_tip_fz):
        raise ValueError("torque gap audit requires no-contact terminal torque")
    if bar_clearance_um not in (0, 1, 2) or (bar_clearance_um and
            (not (tip_mx_gap_audit or c1_open_shear or c1_detached) or
             element_family != "C3D20R")):
        raise ValueError("declared bar clearance requires quadratic terminal torque or C1 opening-shear audit")
    if bar_reference_profile not in ("legacy_linear", "c1_tangent") or (
            bar_reference_profile == "c1_tangent" and
            (bar_clearance_um != 2 or
             not (factor in (4, 5) or
                  (factor == 2 and (c1_open_shear or c1_torque_basis))) or
             (bond_policy != "terminal" and not c1_detached) or
             tip_load_map != "quadratic_face")):
        raise ValueError("C1 tangent profile requires the bounded quadratic terminal reference")
    if factor == 5 and not (
            (c1_open_shear or c1_torque_basis) and
            boundary_policy == "left_root" and bond_policy == "terminal" and
            element_family == "C3D20R" and bar_clearance_um == 2 and
            plate_thickness_mm == 35 and tip_load_map == "quadratic_face" and
            full_displacement_output):
        raise ValueError("factor-five solid mesh requires bounded C1 influence or combined load")
    if element_family not in ("C3D8", "C3D20R") or (
            element_family == "C3D20R" and
            (bond_policy not in ("detached", "terminal", "full") or
             boundary_policy != "left_root" or
             shear not in ("tip_mx", "tip_fz") or
             contact_policy not in ("none", "c1_penalty"))):
        raise ValueError("quadratic screen requires no-contact left/root tip load")
    if tip_load_map not in ("legacy", "quadratic_face") or (
            tip_load_map == "quadratic_face" and
            (element_family != "C3D20R" or
             shear not in ("tip_mx", "tip_fz") or
             (shear == "tip_fz" and bond_policy != "full" and
              not reverse_tip_fz) or
             bond_policy not in ("detached", "terminal", "full") or
             boundary_policy != "left_root" or
             contact_policy not in ("none", "c1_penalty"))):
        raise ValueError("consistent quadratic face traction requires admitted no-contact C3D20R tip loading")
    plate_x = (PLATE_X_MM if bond_policy == "full" else
               tuple(sorted(set((*PLATE_X_MM, 900)))))
    bar_x = (BAR_X_MM if bond_policy == "full" else
             tuple(sorted(set((*BAR_X_MM, 900)))))
    if factor == 5:
        plate_x = tuple(sorted(set((*plate_x, 825))))
        bar_x = tuple(sorted(set((*bar_x, 825))))
    if sided_port_cut:
        # The saved F5 has x=.95 as a midside node in a .94-.96 element.
        # Split both shared domains there so element-side weak work is
        # recoverable without extrapolating a stress cut.
        plate_x = tuple(sorted(set((*plate_x, 950))))
        bar_x = tuple(sorted(set((*bar_x, 950))))
    px,py,pz=(_cuts(v,factor) for v in
              (plate_x,PLATE_Y_MM,plate_z))
    bx,by,bz=(_cuts(v,factor) for v in
              (bar_x,BAR_Y_MM,bar_z))
    if bar_reference_profile == "c1_tangent" and Fraction(825) not in bx:
        raise AssertionError("C1 transition midpoint must split the solid grid")
    coordinates: dict[tuple[Fraction,Fraction,Fraction],int]={}
    detached: dict[tuple[Fraction,Fraction,Fraction],int]={}
    plate_nodes=set()
    bar_nodes=set()
    next_id=1

    def node(key: tuple[Fraction,Fraction,Fraction]) -> int:
        nonlocal next_id
        if key not in coordinates:
            coordinates[key]=next_id
            next_id+=1
        return coordinates[key]

    def bar_node(key: tuple[Fraction,Fraction,Fraction]) -> int:
        nonlocal next_id
        if bond_policy in ("terminal", "detached") and key[2] == plate_top and (
                bond_policy == "detached" or key[0] < 900):
            if key not in detached:
                detached[key]=next_id
                next_id+=1
            return detached[key]
        return node(key)

    for z in pz:
        for y in py:
            for x in px:
                plate_nodes.add(node((x,y,z)))
    for z in bz:
        for y in by:
            for x in bx:
                bar_nodes.add(bar_node((x,y,z)))

    def cells(xs,ys,zs,node_at):
        for iz in range(len(zs)-1):
            for iy in range(len(ys)-1):
                for ix in range(len(xs)-1):
                    x0,x1=xs[ix:ix+2]
                    y0,y1=ys[iy:iy+2]
                    z0,z1=zs[iz:iz+2]
                    yield tuple(node_at(p) for p in (
                        (x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
                        (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)))

    plate_cells=tuple(cells(px,py,pz,node))
    bar_cells=tuple(cells(bx,by,bz,bar_node))
    inverse={index:key for key,index in
             ((*coordinates.items(), *detached.items()))}
    if element_family == "C3D20R":
        midsides: dict[tuple[int,int],int] = {}
        def elevate(cells8, domain_nodes):
            nonlocal next_id
            raised=[]
            for cell in cells8:
                edge_nodes=[]
                for ia, ib in HEX20_EDGES:
                    a, b=cell[ia], cell[ib]
                    edge=tuple(sorted((a,b)))
                    if edge not in midsides:
                        inverse[next_id]=tuple((u+v)/2 for u,v in
                                               zip(inverse[a],inverse[b]))
                        midsides[edge]=next_id
                        next_id+=1
                    edge_nodes.append(midsides[edge])
                    domain_nodes.add(midsides[edge])
                raised.append(cell+tuple(edge_nodes))
            return tuple(raised)
        plate_cells=elevate(plate_cells,plate_nodes)
        bar_cells=elevate(bar_cells,bar_nodes)
    if len(set((*coordinates.values(), *detached.values()))) != len(coordinates) + len(detached):
        raise AssertionError("duplicate solid corner-node identity")
    if set(inverse) != set(range(1,next_id)):
        raise AssertionError("solid node identity changed")
    reference_inverse=inverse.copy()
    if bar_clearance_um:
        clearance_mm=Fraction(bar_clearance_um,1000)
        for index in bar_nodes-plate_nodes:
            x,y,z=inverse[index]
            if x < 900:
                if bar_reference_profile == "legacy_linear":
                    taper=min(Fraction(1), (900-x)/150)
                    inverse[index]=(x,y,z+clearance_mm*taper)
                else:
                    if x <= 750:
                        offset,slope=clearance_mm,Fraction(0)
                    else:
                        t=(x-750)/150
                        if t <= Fraction(1,2):
                            offset=clearance_mm*(1-2*t*t)
                            slope=-4*clearance_mm*t/150
                        else:
                            offset=2*clearance_mm*(1-t)*(1-t)
                            slope=-4*clearance_mm*(1-t)/150
                    normalizer=(1+float(slope)**2)**.5
                    local_z=float(z-(plate_top+50))
                    inverse[index]=(
                        float(x)-local_z*float(slope)/normalizer,
                        y,
                        float(plate_top+50+offset)+local_z/normalizer)
    interface={(x,y,plate_top) for x in bx for y in by
               if bond_policy == "full" or
               (bond_policy == "terminal" and x >= 900)}
    if not all(coordinates[p] in plate_nodes & bar_nodes for p in interface):
        raise AssertionError("plate/bar interface is not conforming")
    if bond_policy in ("terminal", "detached") and (
            not detached or
            any(coordinates[p] in bar_nodes for p in detached)):
        raise AssertionError("unbonded plate/stiffener nodes were merged")
    plate_boundary={coordinates[(x,y,z)] for z in pz for y in py for x in px
                    if (x == px[0] if boundary_policy != "perimeter" else
                        x in (px[0],px[-1]) or y in (py[0],py[-1]))}
    beam_root={bar_node((Fraction(0),y,z)) for y in by for z in bz}
    if element_family == "C3D20R":
        plate_boundary.update(index for index in plate_nodes
                              if inverse[index][0] == px[0])
        beam_root.update(index for index in bar_nodes
                         if inverse[index][0] == 0)
    boundary=(plate_boundary | beam_root if boundary_policy == "left_root"
              else plate_boundary)
    if boundary & (bar_nodes-plate_nodes) and boundary_policy != "left_root":
        raise AssertionError("bar has a directly prescribed node")
    if boundary_policy == "left_root" and boundary & (bar_nodes-plate_nodes) != (
            beam_root-plate_nodes):
        raise AssertionError("unexpected stiffener support")
    station_nodes={str(s): sorted(bar_node((Fraction(s),y,z))
                                  for y in by for z in bz)
                   for s in STATIONS_MM}
    surface_x=((250,500,750) if factor==1 else
               (375,500,625) if factor==2 else
               (450,500,550) if factor==5 else
               (437.5,500,562.5))
    surface={(x,y):coordinates[(Fraction(x),Fraction(y),plate_top)]
             for x in surface_x for y in (-50,0,50)}
    probe=set().union(*station_nodes.values(),surface.values())
    if tip_load_map == "quadratic_face":
        probe.update(index for index in bar_nodes if inverse[index][0] == 1000)
    lines=[("** BOUNDED C3D8 PLATE AND CONFORMING SOLID STIFFENER; DEVELOPMENT ONLY"
            if element_family == "C3D8" else
            "** BOUNDED C3D20R PLATE AND SOLID STIFFENER; DEVELOPMENT ONLY"),
           f"** factor={factor}; shear={shear}; gamma={_number(GAMMA)}",
           "*NODE,NSET=AllNodes"]
    plate_contact=[]
    bar_contact=[]
    if contact_policy in ("mortar_hard", "c1_penalty"):
        def unbonded_face(cell: tuple[int, ...], face: tuple[int, ...]) -> bool:
            # C1 bar nodes are displaced from their topology coordinates.
            xyz=[reference_inverse[cell[index]] for index in face]
            return (all(z == plate_top for _,_,z in xyz) and
                    min(x for x,_,_ in xyz) >= 0 and
                    max(x for x,_,_ in xyz) <= 900 and
                    min(y for _,y,_ in xyz) >= -50 and
                    max(y for _,y,_ in xyz) <= 50)
        plate_contact=[eid for eid,cell in enumerate(plate_cells,1)
                       if unbonded_face(cell,(4,5,6,7))]
        bar_contact=[eid for eid,cell in
                     enumerate(bar_cells,len(plate_cells)+1)
                     if unbonded_face(cell,(0,1,2,3))]
        expected=4*factor*2*factor
        if len(plate_contact)!=expected or len(bar_contact)!=expected:
            raise AssertionError("unbonded contact face inventory changed")
    for index in range(1,len(inverse)+1):
        xyz=inverse[index]
        lines.append(f"{index},"+",".join(_number(float(v)/1000) for v in xyz))
    lines.append(f"*ELEMENT,TYPE={element_family},ELSET=Plate")
    for eid,cell in enumerate(plate_cells,1):
        if element_family == "C3D20R":
            lines.append(f"{eid},"+",".join(map(str,cell[:15])))
            lines.append(",".join(map(str,cell[15:])))
        else:
            lines.append(f"{eid},"+",".join(map(str,cell)))
    lines.append(f"*ELEMENT,TYPE={element_family},ELSET=Stiffener")
    for eid,cell in enumerate(bar_cells,len(plate_cells)+1):
        if element_family == "C3D20R":
            lines.append(f"{eid},"+",".join(map(str,cell[:15])))
            lines.append(",".join(map(str,cell[15:])))
        else:
            lines.append(f"{eid},"+",".join(map(str,cell)))
    if contact_policy in ("mortar_hard", "c1_penalty"):
        lines.append("*ELSET,ELSET=PlateUnbondedFaces")
        lines.extend(_set_lines(plate_contact))
        lines.append("*ELSET,ELSET=BarUnbondedFaces")
        lines.extend(_set_lines(bar_contact))
        lines.extend(("*SURFACE,NAME=PlateTopUnbonded,TYPE=ELEMENT",
                      "PlateUnbondedFaces,S2",
                      "*SURFACE,NAME=BarBottomUnbonded,TYPE=ELEMENT",
                      "BarUnbondedFaces,S1"))
        if contact_policy == "mortar_hard":
            lines.extend(("*SURFACE INTERACTION,NAME=HardUnbonded",
                          "*CONTACT PAIR,INTERACTION=HardUnbonded,TYPE=MORTAR",
                          "BarBottomUnbonded,PlateTopUnbonded"))
        else:
            lines.extend(("*SURFACE INTERACTION,NAME=PenaltyUnbonded",
                          "*SURFACE BEHAVIOR,PRESSURE-OVERCLOSURE=LINEAR",
                          "4.2e14,1",
                          "*CONTACT PAIR,INTERACTION=PenaltyUnbonded,TYPE=SURFACE TO SURFACE",
                          "BarBottomUnbonded,PlateTopUnbonded"))
    if element_family == "C3D20R" and bond_policy == "full":
        plate_surface={reference_inverse[i]:i for i in plate_nodes
                       if reference_inverse[i][2] == plate_top and
                       0 <= reference_inverse[i][0] <= 1000 and
                       -50 <= reference_inverse[i][1] <= 50}
        bar_surface={reference_inverse[i]:i for i in bar_nodes
                     if reference_inverse[i][2] == plate_top and
                     0 <= reference_inverse[i][0] <= 1000 and
                     -50 <= reference_inverse[i][1] <= 50}
        if plate_surface != bar_surface:
            raise AssertionError("quadratic full-bond interface is not shared")
        unbonded_pairs=[]
        unbonded_face_pairs=[]
    elif element_family == "C3D20R":
        plate_surface={reference_inverse[i]:i for i in plate_nodes
                       if reference_inverse[i][2] == plate_top and
                       0 <= reference_inverse[i][0] < 900 and
                       -50 <= reference_inverse[i][1] <= 50}
        bar_surface={reference_inverse[i]:i for i in bar_nodes
                     if reference_inverse[i][2] == plate_top and
                     0 <= reference_inverse[i][0] < 900 and
                     -50 <= reference_inverse[i][1] <= 50}
        if set(plate_surface) != set(bar_surface):
            raise AssertionError("quadratic unbonded interface is nonmatching")
        unbonded_pairs=[(plate_surface[key],bar_surface[key])
                        for key in sorted(plate_surface)]
        if any(a == b for a,b in unbonded_pairs):
            raise AssertionError("quadratic unbonded interface was merged")
        def interface_faces(cells, face_indices):
            faces={}
            for cell in cells:
                face=tuple(cell[i] for i in face_indices)
                points=[reference_inverse[i] for i in face]
                if (all(z==plate_top for _,_,z in points) and
                    min(x for x,_,_ in points)>=0 and
                    max(x for x,_,_ in points)<=900 and
                    min(y for _,y,_ in points)>=-50 and
                    max(y for _,y,_ in points)<=50):
                    key=tuple((x,y) for x,y,_ in points[:4])
                    if key in faces:
                        raise AssertionError("repeated quadratic interface face")
                    faces[key]=face
            return faces
        plate_faces=interface_faces(plate_cells,(4,5,6,7,12,13,14,15))
        bar_faces=interface_faces(bar_cells,(0,1,2,3,8,9,10,11))
        if set(plate_faces)!=set(bar_faces):
            raise AssertionError("quadratic interface faces do not match")
        unbonded_face_pairs=[]
        for key in sorted(plate_faces):
            paired=list(zip(plate_faces[key],bar_faces[key]))
            if any(reference_inverse[a]!=reference_inverse[b] for a,b in paired):
                raise AssertionError("quadratic interface face nodes do not match")
            unbonded_face_pairs.append(paired)
    else:
        unbonded_pairs = [(coordinates[key], bar_index)
                          for key, bar_index in sorted(detached.items())]
        unbonded_face_pairs=[]
    if reverse_tip_fz or tip_mx_gap_audit:
        lines.append("*NSET,NSET=UnbondedInterface")
        lines.extend(_set_lines([node for pair in unbonded_pairs for node in pair]))
    lines.extend(("*MATERIAL,NAME=Isotropic", "*ELASTIC", "210000000000,0.3",
                  "*SOLID SECTION,ELSET=Plate,MATERIAL=Isotropic",
                  "*SOLID SECTION,ELSET=Stiffener,MATERIAL=Isotropic",
                  "*NSET,NSET=Probe"))
    lines.extend(_set_lines(probe))
    support_set="Supports" if boundary_policy == "left_root" else "PlateEdge"
    lines.append(f"*NSET,NSET={support_set}")
    lines.extend(_set_lines(boundary))
    lines.extend(("*STEP", "*STATIC", "*BOUNDARY"))
    for index in sorted(boundary):
        x,y,_=inverse[index]
        values=((0.,GAMMA*float(x)/1000,0.) if shear=="uy_x" else
                (GAMMA*float(y)/1000,0.) if shear=="ux_y" else
                (0.,0.,0.))
        for dof,value in enumerate(values,1):
            lines.append(f"{index},{dof},{dof},{_number(value)}")
    applied={}
    if shear in ("tip_mx","tip_fz"):
        tip=tuple(bar_node((Fraction(1000),y,z)) for y in by for z in bz)
        # Trapezoidal face integration of a distributed force/couple; each
        # discrete load has the exact requested resultant about the bar axis.
        weights={index:(.5 if j in (0,len(by)-1) else 1.)*
                        (.5 if k in (0,len(bz)-1) else 1.)
                 for j,y in enumerate(by) for k,z in enumerate(bz)
                 for index in (bar_node((Fraction(1000),y,z)),)}
        if tip_load_map == "quadratic_face":
            # Integrate the declared end-face traction against each
            # quadratic serendipity face shape; the legacy corner-weighted
            # load remains byte-identical for every existing deck.
            end_nodes={inverse[index][1:]:index for index in bar_nodes
                       if inverse[index][0] == 1000}
            roots,gauss_weights=np.polynomial.legendre.leggauss(3)
            face_data=[]
            polar=0.
            for z0,z1 in zip(bz[:-1],bz[1:]):
                for y0,y1 in zip(by[:-1],by[1:]):
                    ym=(y0+y1)/2
                    zm=(z0+z1)/2
                    positions=((y0,z0),(y1,z0),(y1,z1),(y0,z1),
                               (ym,z0),(y1,zm),(ym,z1),(y0,zm))
                    ids=tuple(end_nodes[position] for position in positions)
                    yz=np.array([[float(y)/1000,float(z)/1000]
                                 for y,z in positions])
                    jacobian=float((y1-y0)*(z1-z0))/4e6
                    for xi,wx in zip(roots,gauss_weights):
                        for eta,wy in zip(roots,gauss_weights):
                            shape=_quad8_shape(float(xi),float(eta))
                            y,z=shape@yz
                            weight=float(wx*wy*jacobian)
                            polar+=weight*(y*y+(z-bar_axis_z_m)**2)
                            face_data.append((ids,shape,weight,y,z))
            if polar<=0. or not np.isfinite(polar):
                raise AssertionError("invalid quadratic end-face polar integral")
            vectors={index:np.zeros(3) for index in end_nodes.values()}
            for ids,shape,weight,y,z in face_data:
                traction=(np.array((0.,-100.*(z-bar_axis_z_m)/polar,
                                    100.*y/polar)) if shear=="tip_mx" else
                          np.array((0.,0.,(-1000. if reverse_tip_fz else
                                           1000.)/.01)))
                if torque_with_opening_fz:
                    traction[2] -= 1000./.01
                for index,n in zip(ids,shape):
                    vectors[index]+=weight*n*traction
            applied={index:tuple(force) for index,force in vectors.items()}
        elif shear=="tip_fz":
            normalizer=sum(weights.values())
            applied={index:(0.,0.,(-1000. if reverse_tip_fz else 1000.)*weight/normalizer)
                     for index,weight in weights.items()}
        else:
            polar=sum(weight*(float(inverse[index][1]/1000)**2+
                              (float(inverse[index][2]/1000)-bar_axis_z_m)**2)
                      for index,weight in weights.items())
            applied={index:(0.,-100.*weight*(float(inverse[index][2]/1000)-bar_axis_z_m)/polar,
                            100.*weight*float(inverse[index][1]/1000)/polar)
                     for index,weight in weights.items()}
        if set(applied)!=(set(end_nodes.values()) if tip_load_map == "quadratic_face"
                         else set(tip)):
            raise AssertionError("incomplete loaded end face")
        lines.append("*CLOAD")
        for index,force in sorted(applied.items()):
            for dof,value in enumerate(force,1):
                if value:
                    lines.append(f"{index},{dof},{_number(value)}")
    if contact_policy in ("mortar_hard", "c1_penalty"):
        lines.extend(("*CONTACT PRINT,FREQUENCY=9999", "CDIS,CSTR,CNUM"))
    lines.extend(("*NODE PRINT,NSET=Probe", "U",
                  f"*NODE PRINT,NSET={support_set}", "RF",
                  "*END STEP"))
    if full_displacement_output:
        lines[-1:-1] = ("*NODE FILE,OUTPUT=3D", "U")
    if reverse_tip_fz or tip_mx_gap_audit:
        lines[-1:-1] = ("*NODE PRINT,NSET=UnbondedInterface", "U")
    shared_ids=(sorted(i for i in plate_nodes & bar_nodes if inverse[i][2] == plate_top)
                if element_family == "C3D20R" else
                sorted(coordinates[p] for p in interface))
    meta={"factor":factor,"shear":shear,"gamma":GAMMA,
          "plate_thickness_m":plate_thickness_mm/1000,
          "bar_tip_axis_z_m":bar_axis_z_m,
          "element_family":element_family,
          "node_count":next_id-1,"plate_elements":len(plate_cells),
          "bar_elements":len(bar_cells),"shared_interface_nodes":len(shared_ids),
          "detached_interface_nodes":len(unbonded_pairs),
          "shared_interface_node_ids":shared_ids,
          "detached_stiffener_node_ids":sorted(b for _,b in unbonded_pairs),
          "bond_policy":bond_policy,"bond_interval_m":(
              [0.,1.] if bond_policy=="full" else
              [.9,1.] if bond_policy=="terminal" else None),
          "contact_policy":contact_policy,
          "bar_clearance_um":bar_clearance_um,
          "reverse_tip_fz":reverse_tip_fz,
          "tip_mx_gap_audit":tip_mx_gap_audit,
          "tip_load_map":tip_load_map,
          "unbonded_interface_pairs":(unbonded_pairs if reverse_tip_fz or tip_mx_gap_audit
                                        else []),
          "unbonded_initial_gaps_m":([float((inverse[bar][2]-inverse[plate][2])/1000)
                                        for plate,bar in unbonded_pairs]
                                       if reverse_tip_fz or tip_mx_gap_audit else []),
          "unbonded_face_pairs":(unbonded_face_pairs if reverse_tip_fz or tip_mx_gap_audit
                                  else []),
          "unbonded_face_initial_gaps_m":([
              [float((inverse[bar][2]-inverse[plate][2])/1000)
               for plate,bar in face]
              for face in unbonded_face_pairs]
              if reverse_tip_fz or tip_mx_gap_audit else []),
          "plate_contact_face_element_ids":plate_contact,
          "bar_contact_face_element_ids":bar_contact,
          "boundary_nodes":len(boundary),"boundary_node_ids":sorted(boundary),
          "plate_boundary_node_ids":sorted(plate_boundary),
          "beam_root_node_ids":(sorted(beam_root) if boundary_policy == "left_root"
                                else []),
          "support_set":support_set,
          "probe_node_ids":sorted(probe),
          "bar_station_nodes":station_nodes,
          "probe_coordinates_m":{str(index):[float(v)/1000 for v in inverse[index]]
                                 for index in probe},
          "reference_probe_coordinates_m":{
              str(index):[float(v)/1000 for v in reference_inverse[index]]
              for index in probe},
          "boundary_coordinates_m":{str(index):[float(v)/1000 for v in inverse[index]]
                                    for index in boundary},
          "applied_nodal_forces_N":{str(index):list(force)
                                    for index,force in applied.items()},
          "surface_nodes":{f"{x},{y}":index for (x,y),index in surface.items()},
          "model":(f"3D {element_family} plate/bar with shared nodes " +
                   ("along the full interface" if bond_policy=="full" else
                    "only on the last 0.1 m terminal interface" if
                    bond_policy=="terminal" else "nowhere (free-bar control)") +
                   ("; plate [-.5,1.5]x[-.5,.5]x[-.035,.035] m; bar [0,1]x[-.05,.05]x[.035,.135] m; E=210 GPa, nu=.3"
                    if plate_thickness_mm==70 else
                    "; plate [-.5,1.5]x[-.5,.5]x[-.0175,.0175] m; bar [0,1]x[-.05,.05]x[.0175,.1175] m; E=210 GPa, nu=.3")),
          "boundary_policy":boundary_policy,
          "boundary":("all three translations on plate left edge and stiffener root face fixed"
                      if boundary_policy == "left_root" else
                      "all three translations on plate left edge fixed; bar has no direct supports"
                      if boundary_policy == "left" else
                      "all three translations on plate perimeter follow declared affine shear; bar has no direct supports or loads")}
    if full_displacement_output:
        meta["full_displacement_output"] = True
    if torque_with_opening_fz:
        meta["torque_with_opening_fz"] = True
    if sided_port_cut:
        meta["sided_port_cut_at_095_m"] = True
    if bar_reference_profile == "c1_tangent":
        meta["bar_reference_profile"] = "C1_TWO_QUADRATIC_TANGENT_ALIGNED_V1"
        meta["bar_reference_meaning"] = (
            "2 micrometre clearance through x=0.75-0.90 m; each solid "
            "bar node follows a tangent-aligned native B3 section map; "
            "unbonded paired-node normal gaps are not a contact certificate")
    return "\n".join(lines)+"\n",meta


def parse_nodal_vectors(dat: str, expected: set[int],
                        quantity: str, node_set: str) -> dict[int,np.ndarray]:
    lines=dat.splitlines()
    starts=[i for i,line in enumerate(lines)
            if re.search(rf"(?i)^\s*{re.escape(quantity)}\s*\(",line) and
            re.search(rf"(?i)for set {re.escape(node_set)}\b",line)]
    if len(starts)!=1:
        raise ValueError(f"one {node_set} {quantity} block required")
    result={}
    for line in lines[starts[0]+1:]:
        parts=line.split()
        if not parts:
            if result:
                break
            continue
        if len(parts)!=4 or not parts[0].isdigit():
            if result:
                break
            continue
        node=int(parts[0])
        value=np.array([float(v) for v in parts[1:]])
        if node in result or not np.isfinite(value).all():
            raise ValueError("invalid or repeated nodal vector")
        result[node]=value
    if set(result)!=expected:
        raise ValueError("incomplete nodal vector inventory")
    return result


def parse_probe_displacements(dat: str, expected: set[int]) -> dict[int,np.ndarray]:
    return parse_nodal_vectors(dat,expected,"displacements","PROBE")


def _quad8_shape(xi: float, eta: float) -> np.ndarray:
    corners=((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.))
    return np.array([
        .25*(1.+sx*xi)*(1.+sy*eta)*(sx*xi+sy*eta-1.)
        for sx,sy in corners]+[
        .5*(1.-xi*xi)*(1.-eta),
        .5*(1.+xi)*(1.-eta*eta),
        .5*(1.-xi*xi)*(1.+eta),
        .5*(1.-xi)*(1.-eta*eta)])


def certify_quad8_normal_gap(nodal_gap_m: np.ndarray, *,
                             max_depth: int = 14,
                             max_patches: int = 100000) -> dict:
    """Bound one paired-face normal gap using quadratic Bernstein nets.

    The certificate concerns corresponding material coordinates on matching
    reference faces, not deformed closest-point contact or surface topology.
    """
    nodal=np.asarray(nodal_gap_m,dtype=float)
    if (nodal.shape!=(8,) or not np.isfinite(nodal).all() or
            type(max_depth) is not int or not 0<=max_depth<=20 or
            type(max_patches) is not int or max_patches<1):
        raise ValueError("bounded finite quadratic face gap required")
    sites=(-1.,0.,1.)
    values=np.array([[_quad8_shape(xi,eta)@nodal for eta in sites]
                     for xi in sites])
    transform=np.array(((1.,0.,0.),(-.5,2.,-.5),(0.,0.,1.)))
    net=transform@values@transform.T
    def split_axis0(controls):
        a=(controls[0]+controls[1])/2
        b=(controls[1]+controls[2])/2
        c=(a+b)/2
        return np.stack((controls[0],a,c)),np.stack((c,b,controls[2]))
    def split_both(controls):
        left,right=split_axis0(controls)
        pieces=[]
        for half in (left,right):
            low,high=split_axis0(half.T)
            pieces.extend((low.T,high.T))
        return pieces
    stack=[(net,0)]
    visited=0
    undecided=0
    negative=0
    while stack:
        controls,depth=stack.pop()
        visited+=1
        lower=float(np.min(controls))
        upper=float(np.max(controls))
        if lower>=0.:
            continue
        if upper<0. or float(_quad8_shape(0.,0.)@nodal)<0. and depth==0:
            negative+=1
            continue
        if depth==max_depth or visited+len(stack)+4>max_patches:
            undecided+=1
            continue
        stack.extend((piece,depth+1) for piece in split_both(controls))
    return {"certified_nonnegative":negative==0 and undecided==0,
            "negative_subpatches":negative,"undecided_subpatches":undecided,
            "visited_subpatches":visited,"max_depth":max_depth,
            "max_patches":max_patches,
            "root_bernstein_lower_m":float(np.min(net)),
            "root_bernstein_upper_m":float(np.max(net)),
            "meaning":"floating-point Bernstein control-net bound for reported matching-reference-coordinate quadratic nodal gaps; not an outward-rounded interval proof or deformed closest-point surface-intersection check"}


def certify_unbonded_quadratic_faces(meta: dict,
                                    displacement: dict[int,np.ndarray], *,
                                    max_depth: int = 14) -> dict:
    faces=meta.get("unbonded_face_pairs",())
    initial=meta.get("unbonded_face_initial_gaps_m",())
    if not faces or len(faces)!=len(initial):
        raise ValueError("complete quadratic face inventory required")
    expected={node for face in faces for plate,bar in face
              if plate!=bar for node in (plate,bar)}
    if set(displacement)!=expected:
        raise ValueError("complete quadratic face displacement inventory required")
    records=[]
    for face,initial_face in zip(faces,initial):
        if len(face)!=8 or len(initial_face)!=8:
            raise ValueError("eight-node paired face required")
        nodal=np.array([initial_face[k]+(displacement[bar][2]-
                        displacement[plate][2] if plate!=bar else 0.)
                        for k,(plate,bar) in enumerate(face)])
        records.append(certify_quad8_normal_gap(nodal,max_depth=max_depth))
    return {"face_count":len(records),
            "certified_nonnegative_faces":sum(r["certified_nonnegative"]
                                               for r in records),
            "negative_faces":sum(r["negative_subpatches"]>0 for r in records),
            "undecided_faces":sum(r["undecided_subpatches"]>0 for r in records),
            "max_visited_subpatches":max(r["visited_subpatches"] for r in records),
            "minimum_root_bernstein_lower_m":min(
                r["root_bernstein_lower_m"] for r in records),
            "meaning":"floating-point matching-coordinate reference-normal control-net check only; no outward-rounded or deformed closest-point/contact proof"}


def audit_unbonded_open(meta: dict, displacement: dict[int,np.ndarray]) -> dict:
    """Reference-normal gap at every paired node of the matching free faces."""
    pairs=meta["unbonded_interface_pairs"]
    if not pairs or set(displacement) != {node for pair in pairs for node in pair}:
        raise ValueError("incomplete unbonded interface displacement inventory")
    initial=meta.get("unbonded_initial_gaps_m", [0.]*len(pairs))
    if len(initial) != len(pairs):
        raise ValueError("incomplete initial interface gap inventory")
    gap=np.array([initial[i]+displacement[bar][2]-displacement[plate][2]
                  for i,(plate,bar) in enumerate(pairs)])
    if not np.isfinite(gap).all():
        raise ValueError("nonfinite unbonded interface gap")
    result={"paired_nodes":len(pairs),
            "minimum_reference_normal_gap_m":float(gap.min()),
            "maximum_reference_normal_gap_m":float(gap.max()),
            "negative_nodal_gaps":int(np.count_nonzero(gap < -1e-12)),
            "meaning":"reference-normal paired-node check on matching element faces; contact and deformed-surface closest-point gap were not solved"}
    faces=meta.get("unbonded_face_pairs",[])
    face_initial=meta.get("unbonded_face_initial_gaps_m",[])
    if faces:
        if len(faces)!=len(face_initial) or any(len(face)!=8 for face in faces):
            raise ValueError("invalid quadratic interface face inventory")
        roots=np.polynomial.legendre.leggauss(9)[0]
        shapes=np.array([_quad8_shape(float(xi),float(eta))
                         for xi in roots for eta in roots])
        sample_min=float("inf")
        negatives=0
        for face,initial_face in zip(faces,face_initial):
            if len(initial_face)!=8:
                raise ValueError("invalid quadratic initial face gap inventory")
            nodal=np.array([initial_face[i]+(
                displacement[bar][2]-displacement[plate][2]
                if plate!=bar else 0.)
                for i,(plate,bar) in enumerate(face)])
            sampled=shapes@nodal
            if not np.isfinite(sampled).all():
                raise ValueError("nonfinite quadratic interface face gap")
            sample_min=min(sample_min,float(sampled.min()))
            negatives+=int(np.count_nonzero(sampled < -1e-12))
        result.update({"sampled_face_count":len(faces),
                       "sampled_face_points_per_face":len(shapes),
                       "sampled_face_minimum_reference_normal_gap_m":sample_min,
                       "sampled_face_negative_points":negatives,
                       "sampled_face_meaning":"nine-point Gauss grid per direction on paired quadratic faces, excluding the zero-gap bonded seam; not an exact minimum or deformed closest-point contact check"})
    return result


def _wrench(forces: dict[int,np.ndarray], xyz: dict[str,list[float]]) -> list[float]:
    resultant=np.sum(tuple(forces.values()),axis=0)
    moment=np.sum(tuple(np.cross(np.array(xyz[str(index)]),force)
                        for index,force in forces.items()),axis=0)
    return [*resultant.tolist(),*moment.tolist()]


def split_left_root_support_wrenches(
        meta: dict, reactions: dict[int,np.ndarray]) -> dict[str,list[float]]:
    """Recover disjoint plate and bar-root support wrenches from saved RFs.

    This applies only to the declared left/root solid model. Moments use
    the common global origin, so eccentric reaction forces are included.
    """
    if meta.get("boundary_policy") != "left_root":
        raise ValueError("left/root support model required")
    plate = set(map(int, meta["plate_boundary_node_ids"]))
    bar = set(map(int, meta["beam_root_node_ids"]))
    if not plate or not bar or plate & bar or plate | bar != set(reactions):
        raise ValueError("disjoint complete plate/bar reaction inventory required")
    xyz = meta["boundary_coordinates_m"]
    plate_wrench = np.asarray(_wrench(
        {node: reactions[node] for node in sorted(plate)}, xyz))
    bar_wrench = np.asarray(_wrench(
        {node: reactions[node] for node in sorted(bar)}, xyz))
    total_wrench = np.asarray(_wrench(reactions, xyz))
    if not np.allclose(plate_wrench+bar_wrench, total_wrench,
                       rtol=1e-12, atol=1e-9):
        raise ArithmeticError("support partition does not reconstruct total")
    return {"plate_support_wrench_N_Nm": plate_wrench.tolist(),
            "bar_root_support_wrench_N_Nm": bar_wrench.tolist(),
            "total_support_wrench_N_Nm": total_wrench.tolist()}


def recover(meta: dict, displacement: dict[int,np.ndarray],
            reactions: dict[int,np.ndarray] | None = None) -> dict:
    surface=meta["surface_nodes"]
    plate_top_z=meta.get("plate_thickness_m",.07)/2
    def value(x,y):
        return displacement[surface[f"{x},{y}"]]
    left,right=((250,750) if meta["factor"]==1 else
                (375,625) if meta["factor"]==2 else
                (450,550) if meta["factor"]==5 else
                (437.5,562.5))
    def local(x,y):
        a,b=((left,500) if x<500 else (500,right))
        return value(a,y)+(value(b,y)-value(a,y))*((x-a)/(b-a))
    tangent=(local(550,0)[1]-local(450,0)[1])/.1
    patch=np.array([(x/1000,y/1000) for x in (450,550)
                     for y in (-50,50)],dtype=float)
    pu=np.array([local(x,y)[:2] for x in (450,550)
                  for y in (-50,50)])
    centered=patch-patch.mean(axis=0)
    patch_yaw=float(np.sum(centered[:,0]*pu[:,1]-centered[:,1]*pu[:,0])/
                    np.sum(centered**2))
    stations={}
    for station,indices in meta["bar_station_nodes"].items():
        xyz=np.array([meta["probe_coordinates_m"][str(node)] for node in indices])
        reference_xyz=np.array([meta["reference_probe_coordinates_m"][str(node)]
                                for node in indices])
        center=np.array([float(station)/1000,0.,
                         (xyz[:,2].min()+xyz[:,2].max())/2])
        design=np.zeros((3*len(indices),6))
        response=np.array([displacement[node] for node in indices]).reshape(-1)
        for i,relative in enumerate(xyz-center):
            x,y,z=relative
            design[3*i:3*i+3,:3]=np.eye(3)
            design[3*i:3*i+3,3:]=((0.,z,-y),(-z,0.,x),(y,-x,0.))
        fitted=np.linalg.lstsq(design,response,rcond=None)[0]
        residual=(response-design@fitted).reshape(-1,3)
        top=np.flatnonzero(np.abs(reference_xyz[:,2]-plate_top_z)<1e-12)
        if len(top)!=2*meta["factor"]+1:
            raise ValueError("incomplete stiffener bottom-face trace")
        top=top[np.argsort(xyz[top,1])]
        if (abs(xyz[top[0],1]+.05)>1e-12 or
                abs(xyz[top[-1],1]-.05)>1e-12):
            raise ValueError("stiffener bottom-face extent changed")
        weights=np.ones(len(top))
        weights[[0,-1]]=.5
        top_mean=np.average(response.reshape(-1,3)[top],axis=0,weights=weights)
        fitted_bottom=fitted[:3]+np.cross(fitted[3:],(0.,0.,-.05))
        top_slope_z_y=float((displacement[indices[top[-1]]][2]-
                             displacement[indices[top[0]]][2])/.1)
        shared=(meta.get("bond_policy", "full")=="full" or
                (meta.get("bond_policy")=="terminal" and float(station)>=900.))
        entry={"translation_m":fitted[:3].tolist(),
               "section_rotation_rad":fitted[3:].tolist(),
               "max_warping_residual_m":float(np.linalg.norm(residual,axis=1).max()),
               "fitted_section_bottom_translation_m":fitted_bottom.tolist(),
               "node_count":len(indices),"interface_shared":shared}
        if shared:
            entry.update({"shared_face_mean_translation_m":top_mean.tolist(),
                          "shared_face_minus_fitted_bottom_m":(
                              top_mean-fitted_bottom).tolist(),
                          "shared_face_slope_z_y_rad":top_slope_z_y,
                          "shared_face_slope_minus_section_rx_rad":(
                              top_slope_z_y-float(fitted[3]))})
        else:
            entry.update({"detached_bottom_mean_translation_m":top_mean.tolist(),
                          "detached_bottom_slope_z_y_rad":top_slope_z_y})
        stations[station]=entry
    if meta.get("bond_policy", "full")=="full":
        result={"centerline_tangent_yaw_rad":float(tangent),
                "surface_square_patch_yaw_rad":patch_yaw,
                "surface_square_patch_half_width_m":.05,
                "station_motion":stations}
        surface_slope_z_x=float((value(right,0)[2]-value(left,0)[2])/
                                ((right-left)/1000))
        beam_ry=float(stations["500"]["section_rotation_rad"][1])
        result["midspan_transverse_kinematics"]={
            "plate_top_slope_z_x_rad":surface_slope_z_x,
            "beam_section_ry_rad":beam_ry,
            "slope_plus_section_ry_rad":surface_slope_z_x+beam_ry,
            "meaning":"kinematic slope/section comparison; not a measured interface gap or shell director shear"}
    else:
        midspan=[]
        xyz=meta["probe_coordinates_m"]
        reference_xyz=meta["reference_probe_coordinates_m"]
        for y in (-50,0,50):
            plate=surface[f"500,{y}"]
            bottom=next((index for index in meta["bar_station_nodes"]["500"]
                         if abs(reference_xyz[str(index)][1]-y/1000)<1e-12 and
                            abs(reference_xyz[str(index)][2]-plate_top_z)<1e-12),None)
            if bottom is None:
                raise ValueError("incomplete unbonded midspan bottom probe")
            midspan.append(float(xyz[str(bottom)][2]-xyz[str(plate)][2]+
                                 displacement[bottom][2]-displacement[plate][2]))
        result={"plate_midspan_centerline_tangent_yaw_rad":float(tangent),
                "plate_midspan_square_patch_yaw_rad":patch_yaw,
                "station_motion":stations,
                "unbonded_midspan_normal_separation_m":{
                    "y_m":[-.05,0.,.05],"bar_minus_plate_z_m":midspan,
                    "meaning":"negative value is sampled geometric overlap of unbonded surfaces; not a mortar integration-point contact gap or traction"},
                "meaning":("midspan plate and bar have declared clearance outside the bonded patch"
                           if meta.get("bar_clearance_um") else
                           "midspan plate and bar are coincident but intentionally unbonded")}
    applied={int(index):np.array(force) for index,force in
             meta["applied_nodal_forces_N"].items()}
    if applied:
        work=float(sum(force@displacement[index]
                       for index,force in applied.items()))
        magnitude=(100. if meta["shear"]=="tip_mx" else 1000.)
        result["applied_load_work_j"]=work
        if meta.get("torque_with_opening_fz"):
            result["applied_load_work_meaning"] = (
                "mixed opening-force and torque work; no pure-load "
                "tip displacement follows by dividing this value")
        else:
            result["tip_work_conjugate_displacement"]={
                "value":work/magnitude,
                "unit":"rad" if meta["shear"]=="tip_mx" else "m",
                "definition":"sum(applied nodal force dot nodal displacement) / declared load magnitude"}
    if reactions is not None:
        applied_wrench=(_wrench(applied,meta["probe_coordinates_m"])
                        if applied else [0.]*6)
        support_wrench=_wrench(reactions,meta["boundary_coordinates_m"])
        result["applied_wrench_N_Nm"]=applied_wrench
        result["support_wrench_N_Nm"]=support_wrench
        result["wrench_imbalance_N_Nm"]=(np.array(applied_wrench)+
                                          np.array(support_wrench)).tolist()
    return result


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--suite",choices=("shear","loads"),default="shear")
    parser.add_argument("--factor",type=int,choices=(1,2,4,5),nargs="+",default=(1,2))
    parser.add_argument("--boundary",choices=("perimeter","left","left_root"),default="perimeter")
    parser.add_argument("--bond",choices=("full","terminal","detached"),default="full")
    parser.add_argument("--contact",choices=("none","mortar_hard","c1_penalty"),
                        default="none")
    parser.add_argument("--element-family",choices=("C3D8","C3D20R"),
                        default="C3D8")
    parser.add_argument("--reverse-fz-open-audit",action="store_true")
    parser.add_argument("--tip-mx-gap-audit",action="store_true")
    parser.add_argument("--bar-clearance-um",type=int,choices=(0,1,2),default=0)
    parser.add_argument("--bar-reference-profile",
                        choices=("legacy_linear","c1_tangent"),
                        default="legacy_linear")
    parser.add_argument("--half-plate-thickness",action="store_true")
    parser.add_argument("--quadratic-face-traction",action="store_true")
    parser.add_argument("--full-displacement-output",action="store_true")
    parser.add_argument("--torque-with-opening-fz",action="store_true")
    args=parser.parse_args()
    c1_open_shear = (args.bar_reference_profile == "c1_tangent" and
                     args.reverse_fz_open_audit and
                     args.contact == "none")
    c1_torque_basis = (args.bar_reference_profile == "c1_tangent" and
                       args.tip_mx_gap_audit and args.contact == "none")
    if 5 in args.factor and not (list(args.factor) == [5] and
            (c1_open_shear or c1_torque_basis) and
            args.suite == "loads" and args.boundary == "left_root" and
            args.bond == "terminal" and args.element_family == "C3D20R" and
            args.bar_clearance_um == 2 and args.half_plate_thickness and
            args.quadratic_face_traction and args.full_displacement_output):
        parser.error("factor five requires the bounded C1 influence or combined reference")
    if args.boundary != "perimeter" and args.suite != "loads":
        parser.error("left-edge support is admitted only for the tip-load suite")
    if args.bond == "terminal" and (args.boundary != "left_root" or
                                    args.suite != "loads"):
        parser.error("terminal bond requires left/root tip-load suite")
    if args.bond == "detached" and (args.boundary != "left_root" or
                                    args.suite != "loads" or args.contact != "none"):
        parser.error("detached control requires no-contact left/root tip-load suite")
    if args.contact == "mortar_hard" and args.bond != "terminal":
        parser.error("hard mortar contact requires terminal bond")
    if args.element_family == "C3D20R" and (
            args.boundary != "left_root" or args.bond not in ("detached","terminal","full") or
            args.suite != "loads" or
            args.contact not in ("none","c1_penalty")):
        parser.error("quadratic screen requires no-contact left/root tip loads")
    if args.reverse_fz_open_audit and (args.suite!="loads" or
            args.boundary!="left_root" or args.bond!="terminal" or
            args.contact!="none"):
        parser.error("reverse Fz opening audit requires no-contact terminal load suite")
    if args.tip_mx_gap_audit and (args.suite!="loads" or
            args.boundary!="left_root" or args.bond!="terminal" or
            args.contact not in ("none","c1_penalty") or
            args.reverse_fz_open_audit):
        parser.error("torque gap audit requires no-contact terminal tip-load suite")
    if args.bar_clearance_um and (not (args.tip_mx_gap_audit or c1_open_shear) or
                                  args.element_family!="C3D20R"):
        parser.error("declared bar clearance requires quadratic terminal torque or C1 opening-shear audit")
    if args.bar_reference_profile=="c1_tangent" and not (
            (list(args.factor) in ([4], [5]) or
             (list(args.factor)==[2] and
              (c1_open_shear or c1_torque_basis))) and
            args.suite=="loads" and
            args.boundary=="left_root" and args.bond=="terminal" and
            args.contact in ("none","c1_penalty") and
            args.element_family=="C3D20R" and
            (args.tip_mx_gap_audit or c1_open_shear) and
            args.bar_clearance_um==2 and
            args.half_plate_thickness and args.quadratic_face_traction):
        parser.error("C1 tangent reference requires the bounded terminal screen")
    if args.half_plate_thickness and (
            args.element_family!="C3D20R" or
            not (((args.tip_mx_gap_audit or c1_open_shear) and
                  args.bar_clearance_um==2 and
                  args.bond=="terminal") or
                 (args.reverse_fz_open_audit and args.bar_clearance_um==0 and
                  args.bond=="terminal" and args.quadratic_face_traction) or
                 (args.quadratic_face_traction and args.bar_clearance_um==0 and
                  args.bond in ("detached","full"))) or
            args.boundary!="left_root" or any(f not in (2,4,5) for f in args.factor)):
        parser.error("half-thickness diagnostic requires admitted quadratic terminal, detached, or full-bond control")
    if args.quadratic_face_traction and (
            args.element_family!="C3D20R" or args.suite!="loads" or
            args.boundary!="left_root" or args.bond not in ("terminal","detached","full") or
            args.contact not in ("none","c1_penalty") or
            (args.reverse_fz_open_audit and args.bond!="terminal")):
        parser.error("quadratic face traction requires admitted no-contact C3D20R tip loading")
    if args.full_displacement_output and not (
            (list(args.factor) in ([4], [5]) or
             (list(args.factor)==[2] and
              (c1_open_shear or c1_torque_basis))) and
            args.suite=="loads" and
            args.boundary=="left_root" and
            args.contact in ("none","c1_penalty") and
            args.element_family=="C3D20R" and
            args.half_plate_thickness and args.quadratic_face_traction and
            (args.bond=="full" or
             (args.bond=="terminal" and
              (args.tip_mx_gap_audit or c1_open_shear) and
              args.bar_clearance_um==2 and
              args.bar_reference_profile=="c1_tangent"))):
        parser.error("full-field reference requires the bounded full bond or C1 terminal case")
    if args.contact == "c1_penalty" and not (
            list(args.factor)==[4] and args.suite=="loads" and
            args.boundary=="left_root" and args.bond=="terminal" and
            args.element_family=="C3D20R" and args.tip_mx_gap_audit and
            args.bar_clearance_um==2 and args.bar_reference_profile=="c1_tangent" and
            args.half_plate_thickness and args.quadratic_face_traction and
            args.full_displacement_output):
        parser.error("C1 penalty contact requires the matched factor-4 torque case")
    if args.torque_with_opening_fz and not (
            c1_torque_basis and list(args.factor) in ([2], [4], [5]) and
            args.suite == "loads" and args.boundary == "left_root" and
            args.bond == "terminal" and args.element_family == "C3D20R" and
            args.bar_clearance_um == 2 and args.half_plate_thickness and
            args.quadratic_face_traction and args.full_displacement_output):
        parser.error("combined C1 opening/torque requires bounded no-contact full-field case")
    if digest(EXE)!=EXE_SHA:
        raise ValueError("CalculiX executable identity changed")
    head=subprocess.run(("git","rev-parse","HEAD"),cwd=WORKTREE,
                        check=True,capture_output=True,text=True).stdout.strip()
    status=subprocess.run(("git","status","--porcelain"),cwd=WORKTREE,
                          check=True,capture_output=True,text=True).stdout
    # A protected, unrelated pytest scratch root predates this experiment.
    # Keep it intact while requiring every tracked or other untracked edit
    # to be committed before recording source-bound external evidence.
    dirty=[line for line in status.splitlines()
           if line != "?? .pytest-b3-g0-interior/"]
    if dirty:
        raise ValueError("solid bond screen requires clean source")
    root=args.root.resolve()
    root.mkdir(parents=True,exist_ok=False)
    manifest={"status":"EXPLORATORY_NOT_G0_ACCEPTANCE",
              "question":("Do the inferred finite-area shell director pose and solid beam section fit agree on a mesh-aligned interior patch under the already matched full-bond torque and shear loads?"
                          if args.full_displacement_output else
                          "Does a fully shared 35 mm quadratic solid bond under matched left/root supports and consistent tip torque/shear loading approach a stable physical transfer response?"
                          if args.quadratic_face_traction and args.bond=="full" else
                          "Does reverse transverse loading with consistent quadratic end-face force leave the 35 mm unbonded solid interface open and resolve its work?"
                          if args.quadratic_face_traction and args.reverse_fz_open_audit else
                          "Does replacing corner-weighted tip torque with consistent quadratic end-face traction change the 35 mm terminal bond torque partition or detached free-bar work?"
                          if args.quadratic_face_traction else
                          "At the plate-only characterized 35 mm thickness, does a quadratic terminal solid bond retain open unbonded faces and a resolved torque partition/tip work?"
                          if args.half_plate_thickness else
                          "Does a declared micrometre clearance keep the unbonded terminal-bar faces open under torque while leaving work insensitive to clearance and mesh?"
                          if args.bar_clearance_um else
                          "Does terminal torque on the free, coincident no-contact faces produce a negative physical normal gap, invalidating this torque comparator?"
                          if args.tip_mx_gap_audit else
                          "Does replacing bending-stiff C3D8 with quadratic C3D20R on the same free-bar mesh reduce shear/torsion resolution error before advancing a terminal reference?"
                          if args.element_family=="C3D20R" and args.bond=="detached" else
                          "Can a C3D20R terminal model retain an open unbonded interface under reverse Fz while reducing the solid section error?"
                          if args.element_family=="C3D20R" and args.reverse_fz_open_audit else
                          "Do the same C3D8 bar meshes resolve free bending and torsion before their terminal-bond responses are used to judge a connection?"
                          if args.bond=="detached" else
                          "Does reversed transverse loading leave the whole matching unbonded solid interface open, making the no-contact solution an admissible directional comparator?"
                          if args.reverse_fz_open_audit else
                          "finite physical bond yaw under two affine shears"
                          if args.suite=="shear" else
                          "Can hard mortar contact remove unbonded penetration while retaining the declared terminal solid bond and load work?"
                          if args.contact=="mortar_hard" else
                          "terminal last-0.1 m solid bond versus declared native terminal footprint under matched supports and loads"
                          if args.bond=="terminal" else
                          "effect of plate support on physical solid-bond end torque and transverse-force transfer"),
              "decision":("Small interior pose gaps in both loads support the finite-area pose geometry for further native line testing; substantial gaps keep the shell-side pose idealization open. This one solid level cannot select a multiplier or qualify accuracy."
                          if args.full_displacement_output else
                          "If torque/shear work and support partition settle at factors 2/4, compare matched physical observables with the native full-line and reduced-channel screens while recording solid warping and shell idealization limits; drift blocks accuracy scoring. No line law is selected from this reference alone."
                          if args.quadratic_face_traction and args.bond=="full" else
                          "If the reverse-Fz faces remain open and the force work settles across the two bounded factors, compare the physical shear work and support split with the existing sign-reversed native linear cases; overlap or drift blocks external shear accuracy scoring. Neither outcome selects a connection law."
                          if args.quadratic_face_traction and args.reverse_fz_open_audit else
                          "If the consistent face load materially shifts torque partition, revise the external load idealization before judging the native footprint; if it does not, focus the next discriminator on connection pose and local tip work. Neither outcome selects a production law."
                          if args.quadratic_face_traction else
                          "If face gaps and torque work remain admissible and settle, compare the section-axis torque partition and local tip compliance with the same-thickness graded native model. A persistent gap points to connection transfer rather than the already smaller plate-only discrepancy; overlap or drift blocks accuracy scoring. No footprint cutoff is inferred."
                          if args.half_plate_thickness else
                          "If paired-node gaps remain positive and torque work is clearance- and mesh-insensitive, retain only a declared-gap external reference candidate pending face-interior checks; otherwise require contact. No native footprint is selected."
                          if args.bar_clearance_um else
                          "If any paired node overlaps under torque, reject no-contact torque as a physical comparator; if all remain open, assess face interiors and deformation before using it. Neither outcome selects a footprint."
                          if args.tip_mx_gap_audit else
                          "If quadratic free-bar torque/shear approach the declared section values and change little over the bounded levels, advance the open-terminal shear check; otherwise stop this solid reference route. No footprint is selected."
                          if args.element_family=="C3D20R" and args.bond=="detached" else
                          "If the quadratic reverse-Fz case has a fully open matching interface and section-resolved response, retain it as a directional shear comparator; otherwise stop. No torque or footprint accuracy is inferred."
                          if args.element_family=="C3D20R" and args.reverse_fz_open_audit else
                          "If the free bar itself remains mesh-sensitive, do not use current terminal-solid responses to rank connection laws; if it stabilizes, investigate plate and terminal-bond discretization next. Neither outcome selects a native footprint."
                          if args.bond=="detached" else
                          "If every reference-normal paired-node gap is nonnegative, retain only the reverse-Fz open-interface result as a directional physical comparator, subject to mesh convergence and deformed-surface checks; otherwise reject this no-contact reference for shear. Neither outcome selects a native footprint."
                          if args.reverse_fz_open_audit else
                          "If the hard-mortar case executes with admissible contact output, no sampled penetration, work and reaction balance, advance it to mesh/contact assessment; otherwise preserve the failed reference and revise its external contact formulation. No native connection is selected by an unresolved contact result."
                          if args.contact=="mortar_hard" else
                          "If the terminal solid work responses settle under refinement, compare them separately with the rigid and distributing native footprint idealizations; if they do not settle, retain a feasibility reference only. A mismatch prompts connection/section/warping diagnosis, never an automatic line-space selection."
                          if args.bond=="terminal" else
                          "Retain as a bounded physical-bond diagnostic under its declared support and connection."),
              "source_head":head,"source_sha256":digest(Path(__file__)),
              "executable":str(EXE),"executable_sha256":EXE_SHA,
              "cases":[(factor,shear) for factor in args.factor
                   for shear in (("tip_mx","tip_fz") if args.quadratic_face_traction and args.bond=="full" else
                                 ("tip_fz",) if args.reverse_fz_open_audit else
                                 ("tip_mx",) if args.quadratic_face_traction else
                                 ("tip_mx",) if args.tip_mx_gap_audit else
                                 ("uy_x","ux_y") if args.suite=="shear"
                                 else ("tip_mx","tip_fz"))],
              "boundary_policy":args.boundary,
              "bond_policy":args.bond,
              "contact_policy":args.contact,
              "element_family":args.element_family,
              "element_reference":"https://www.dhondt.de/ccx_2.22.pdf, element types: C3D20R preferred for bending; no quadratic contact is used",
              "reverse_fz_open_audit":args.reverse_fz_open_audit,
              "tip_mx_gap_audit":args.tip_mx_gap_audit,
              "bar_clearance_um":args.bar_clearance_um,
              "plate_thickness_m":.035 if args.half_plate_thickness else .07,
              "tip_load_map":("quadratic_face" if args.quadratic_face_traction
                              else "legacy"),
              "contact_reference":"https://www.dhondt.de/ccx_2.22.pdf, sections on MORTAR contact and hard surface behavior (external-reference diagnostic only)",
              "limit":{"omp_threads":1,"timeout_seconds_each":TIMEOUT_SECONDS},
              "limits_of_claim":["3D solid idealization, not E4 or B3",
                                 f"{len(args.factor)} development mesh levels, not convergence qualification",
                                 f"prescribed plate {args.boundary} and bar root as declared",
                                 "quadratic screen uses consistent integrated end-face torque or transverse traction"
                                 if args.quadratic_face_traction else
                                 "quadratic screen retains corner-node tip loads to isolate element interpolation; not a consistent quadratic face traction"
                                 if args.element_family=="C3D20R" else
                                 "linear-brick tip loads use corner-node face weights",
                                 "bar axis and section are lifted by the declared clearance outside the terminal bond and taper to the original axis over x=0.75-0.9 m; this differs from the straight native beam"
                                 if args.bar_clearance_um else
                                 "fully shared plate/bar interface; no unbonded plate/bar face"
                                 if args.bond=="full" else
                                 "no initial clearance outside the shared patch",
                                 ("detached control has no physical plate/bar connection"
                                  if args.bond=="detached" else
                                  "terminal mode uses external hard-mortar contact only outside the shared patch"
                                  if args.contact=="mortar_hard" else
                                 "terminal mode omits contact outside the shared patch"),
                                 "connection and section still differ from native weak-line B3/E4"]}
    if args.full_displacement_output:
        manifest["full_displacement_output"] = True
        manifest["limits_of_claim"].append(
            "output-only replay of factor-4 cases; full solid displacement field for read-only interior pose projection, not a new convergence level")
    if args.bar_reference_profile=="c1_tangent":
        manifest["bar_reference_profile"]="C1_TWO_QUADRATIC_TANGENT_ALIGNED_V1"
        manifest["question"]=(
            "On the unchanged C1 tangent-aligned terminal geometry, does "
            "the separate opening -1000 N Fz load leave the unbonded faces "
            "open with a balanced physical response?" if c1_open_shear else
            "On one tangent-matched terminal solid geometry, does the saved "
            "100 N m torque response preserve an admissible unbonded-face gap "
            "before comparing any native connection response?")
        manifest["decision"]=(
            "An open, balanced completed shear case advances to matching "
            "native curved-model shear and refinement; overlap or failure "
            "stops this directional reference. No torque or multiplier "
            "accuracy follows." if c1_open_shear else
            "A negative deformed paired-face gap rejects this no-contact "
            "reference. Nonnegative paired-face gaps only advance to a "
            "continuous deformed-surface intersection check and matching "
            "native curved-model construction; neither outcome selects a "
            "footprint or multiplier space.")
        manifest["limits_of_claim"]=[
            ("bar nodes follow the two-quadratic C1 tangent-aligned section "
             "map over x=0.75-0.9 m; native coupled mechanics is not solved"
             if entry.startswith("bar axis and section are lifted") else entry)
            for entry in manifest["limits_of_claim"]]
        manifest["limits_of_claim"].append(
            "paired reference-normal gaps are not a continuous deformed-contact certificate")
        if (list(args.factor)==[2] and c1_torque_basis and
                not args.torque_with_opening_fz):
            manifest["question"] = (
                "On the identical factor-two C1 tangent terminal model, "
                "does the no-contact torque field supply a linear basis "
                "whose superposition with the executed opening-shear field "
                "has an open saved interface and stable incremental torque "
                "transfer relative to factor four?")
            manifest["decision"] = (
                "Only a positive combined-field face-gap bound and stable "
                "physical observables retain a pre-opened incremental torque "
                "reference for further contact and mesh checks. Pure torque "
                "penetration cannot be scored, and no native line law follows.")
            manifest["limits_of_claim"].append(
                "pure no-contact torque is a linear basis only; the physical "
                "pre-opened state is derived with the separately executed "
                "matching opening-shear solve")
        if args.full_displacement_output:
            if c1_open_shear:
                full_u_meaning = (
                    "full displacement output on the C1 opening-shear model "
                    "for deformed seam and surface projection")
            elif args.torque_with_opening_fz:
                full_u_meaning = (
                    "full displacement output on the executed combined "
                    "C1 opening/torque model for contact-gap assessment")
            elif list(args.factor)==[2] and c1_torque_basis:
                full_u_meaning = (
                    "factor-two C1 torque linear-basis field for the "
                    "pre-opened incremental reference")
            else:
                full_u_meaning = (
                    "output-only replay of the unchanged C1 terminal torque "
                    "model for deformed seam and surface projection")
            manifest["limits_of_claim"]=[
                (full_u_meaning if
                 entry.startswith("output-only replay of factor-4 cases")
                 else entry)
                for entry in manifest["limits_of_claim"]]
    if args.contact == "c1_penalty":
        if "?? .pytest-b3-g0-interior/" in status.splitlines():
            manifest["untracked_scratch_exclusion"] = ".pytest-b3-g0-interior/"
        paired_deck, _ = build(
            4, "tip_mx", boundary_policy="left_root", bond_policy="terminal",
            element_family="C3D20R", tip_mx_gap_audit=True,
            bar_clearance_um=2, plate_thickness_mm=35,
            tip_load_map="quadratic_face", full_displacement_output=True,
            bar_reference_profile="c1_tangent")
        # The preserved Windows input was written with CRLF newlines.
        paired_hash = hashlib.sha256(
            paired_deck.replace("\n", "\r\n").encode("ascii")).hexdigest()
        if paired_hash != "cc88870f0664cbadb39a114f4755dc996c226ab683d3304ea8f4a60f26b93da0":
            raise ValueError("paired C1 no-contact deck identity changed")
        manifest["paired_no_contact_deck_sha256"] = paired_hash
        manifest["question"] = (
            "On the same C1 tangent-aligned quadratic terminal torque model, "
            "does complete unbonded-face penalty contact execute with an "
            "admissible deformed gap and balanced wrench?")
        manifest["decision"] = (
            "Failure, unsupported contact, or sampled penetration retains "
            "the physical torque reference as unresolved. An admissible "
            "single run advances only to matched shear and mesh/contact "
            "convergence; it does not score native accuracy or select a line law.")
        manifest["element_reference"] = (
            "C3D20R quadratic solid on the existing C1 factor-4 geometry; "
            "contact execution is an exploratory extension")
        manifest["contact_reference"] = (
            "https://www.dhondt.de/ccx_2.22.pdf, linear penalty "
            "surface-to-surface contact; finite penetration is possible")
        manifest["limits_of_claim"] = [
            entry for entry in manifest["limits_of_claim"]
            if not entry.startswith("terminal mode omits contact") and
            not entry.startswith("paired reference-normal gaps") and
            not entry.startswith("output-only replay")]
        manifest["limits_of_claim"].extend((
            "complete unbonded faces use finite-stiffness penalty contact, "
            "not an exact hard-contact law",
            "one torque mesh and one contact stiffness do not establish "
            "nonpenetration, work convergence, or native accuracy"))
    if args.torque_with_opening_fz:
        manifest["torque_with_opening_fz"] = True
        manifest["question"] = (
            "Does an executed combined -1000 N Fz opening and +100 N m Mx "
            "torque load on the unchanged C1 terminal model match the "
            "superposed separately executed linear fields, remain open on "
            "the saved opposed faces, and preserve wrench balance?")
        manifest["decision"] = (
            "Agreement and a positive continuous saved-face gap retain "
            "the pre-opened incremental torque as a directional reference "
            "for further contact and mesh checks; discrepancy or overlap "
            "stops this route. Neither outcome selects a native line law.")
        manifest["limits_of_claim"].append(
            "combined loading is a pre-opened state, not pure torque from "
            "the virgin reference; side-surface contact remains unchecked")
    elif list(args.factor) == [5]:
        manifest["question"] = (
            "What are the separate factor-five linear torque or opening-force "
            "influences on the same C1 geometry, bond, supports and load map "
            "as the saved combined factor-five field?")
        manifest["decision"] = (
            "If each influence and its sum settle and reproduce the saved "
            "combined work, compare torque and shear separately with the "
            "native idealization. Material drift or failed balance returns "
            "the solid comparator for revision; no line space is selected.")
        manifest["limits_of_claim"].append(
            "pure torque is a linear influence field and may close the "
            "unbonded gap; only the separately saved pre-opened combined "
            "field has a positive-gap assessment")
    (root/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    records=[]
    for factor,shear in manifest["cases"]:
        name=f"solid-{factor}-{shear}"
        folder=root/name
        folder.mkdir()
        deck,meta=build(factor,shear,boundary_policy=args.boundary,
                        bond_policy=args.bond,contact_policy=args.contact,
                        reverse_tip_fz=args.reverse_fz_open_audit,
                        element_family=args.element_family,
                        tip_mx_gap_audit=args.tip_mx_gap_audit and shear=="tip_mx",
                        bar_clearance_um=args.bar_clearance_um,
                        plate_thickness_mm=35 if args.half_plate_thickness else 70,
                        tip_load_map=("quadratic_face" if args.quadratic_face_traction
                                      else "legacy"),
                        full_displacement_output=args.full_displacement_output,
                        bar_reference_profile=args.bar_reference_profile,
                        torque_with_opening_fz=args.torque_with_opening_fz)
        (folder/f"{name}.inp").write_text(deck,encoding="ascii")
        (folder/"model.json").write_text(json.dumps(meta,sort_keys=True,indent=2)+"\n")
        environment=os.environ.copy()
        environment["OMP_NUM_THREADS"]="1"
        start=time.perf_counter()
        failure=None
        try:
            run=subprocess.run((str(EXE),"-i",name),cwd=folder,env=environment,
                               capture_output=True,timeout=TIMEOUT_SECONDS,check=False)
            code,stdout,stderr=run.returncode,run.stdout,run.stderr
        except subprocess.TimeoutExpired as exc:
            code,stdout,stderr=None,exc.stdout or b"",exc.stderr or b""
            failure="TIMEOUT_120_SECONDS"
        (folder/"stdout.log").write_bytes(stdout)
        (folder/"stderr.log").write_bytes(stderr)
        parsed=None
        if failure is None and code!=0:
            failure=f"CALCULIX_EXIT_{code}"
        if failure is None:
            try:
                out=stdout.decode(errors="replace")
                sta=(folder/f"{name}.sta").read_text(errors="replace")
                dat=(folder/f"{name}.dat").read_text(errors="replace")
                status_rows=[re.match(r"^\s*1\s+1\s+1\s+\d+\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s*$",line)
                             for line in sta.splitlines()]
                complete=[m for m in status_rows if m is not None]
                diagnostics=unexpected_solver_diagnostics(
                    out+sta+dat+stderr.decode(errors="replace"),
                    c1_penalty=args.contact=="c1_penalty")
                if (out.count("Job finished")!=1 or len(complete)!=1 or
                    abs(float(complete[0].group(1))-1.)>1e-10 or
                    diagnostics):
                    raise ValueError("solver diagnostic or incomplete job")
                probes=parse_probe_displacements(dat,set(meta["probe_node_ids"]))
                reactions=parse_nodal_vectors(dat,set(meta["boundary_node_ids"]),
                                              "forces",meta["support_set"])
                parsed=recover(meta,probes,reactions)
                if args.reverse_fz_open_audit or (args.tip_mx_gap_audit and shear=="tip_mx"):
                    expected={node for pair in meta["unbonded_interface_pairs"]
                              for node in pair}
                    interface=parse_nodal_vectors(dat,expected,"displacements",
                                                  "UNBONDEDINTERFACE")
                    parsed["unbonded_open_audit"]=audit_unbonded_open(meta,interface)
            except (OSError,ValueError) as exc:
                failure=f"PARSE_{type(exc).__name__}: {exc}"
        record={"case":name,"status":"EXPLORATORY_COMPLETE" if failure is None
                else "EXPLORATORY_FAILURE","failure":failure,"exit_code":code,
                "seconds":time.perf_counter()-start,"model":meta,
                "response":parsed,
                "files_sha256":{p.name:digest(p) for p in sorted(folder.iterdir())
                                if p.is_file()}}
        (folder/"result.json").write_text(json.dumps(record,sort_keys=True,
                                      indent=2,allow_nan=False)+"\n")
        records.append(record)
        print(json.dumps({"case":name,"status":record["status"],
                          "failure":failure,"seconds":record["seconds"]}),flush=True)
        if failure:
            break
    (root/"result.json").write_text(json.dumps(records,sort_keys=True,
                                     indent=2,allow_nan=False)+"\n")
    return 0 if len(records)==len(manifest["cases"]) and all(
        r["failure"] is None for r in records) else 1


if __name__=="__main__":
    raise SystemExit(main())
