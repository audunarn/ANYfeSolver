"""Experimental flat strip with explicit beam-root ground supports.

The beam nodal root translations and x/y rotations are restrained; normal
spin remains free. This is a separate support condition from a shell-only
clamp or a physical retained-section endpoint attachment. BL5 is unqualified.
Run through an installed ANYfeSolver package; no private operators are used.
"""
import argparse
import json

import numpy as np
import anyfesolver
from anyfesolver import b3_ge, stiffened_shell as ss


LOADS={"fy":(0.,100.,0.,0.,0.,0.),"mx":(0.,0.,0.,10.,0.,0.)}


def _beam_reference_points(intervals):
    xs=np.linspace(0.,1.,intervals+1)
    return np.column_stack((xs,np.zeros_like(xs),np.full_like(xs,.0675)))


def make_definition(intervals=16):
    if type(intervals) is not int or intervals<2 or intervals%2:
        raise ValueError("an even positive shell/half-cell count of at least two is required")
    e,nu=210e9,.3;g=e/(2*(1+nu))
    section=b3_ge.EllipsoidalGeneralizedSection(np.diag((
        e*.01,g*.008,g*.008,g*1.6666666666666667e-5,
        e*8.333333333333333e-6,e*8.333333333333333e-6)),np.eye(6),1e20,1e9)
    xs=np.linspace(0.,1.,intervals+1)
    points=_beam_reference_points(intervals)
    beams=tuple(b3_ge.define_beam("b3-ge",i+1,(101+2*i,102+2*i,103+2*i),
        points[2*i:2*i+3],np.tile(np.eye(3),(3,1,1)),section,np.eye(6))
        for i in range(intervals//2))
    row=intervals+1
    shell=ss.ShellPanel(tuple(range(1,2*row+1)),
        tuple((float(x),y,0.) for y in (0.,.2) for x in xs),
        tuple((i,i+1,i+row+1,i+row) for i in range(1,row)),.035,e,nu,7850.)
    line=ss.ExperimentalBL5Line(tuple(range(1,row+1)),
        tuple((i,half) for i in range(len(beams)) for half in (0,1)),multiplier_span=1)
    fixed=tuple(ss.FixedCoordinate("shell",n,j) for n in (1,row+1) for j in range(6))
    fixed+=tuple(ss.FixedCoordinate("beam",101,j) for j in range(5))
    return ss.LinearDefinition(ss.B3_GE_E4_PL_STIFFENED_SHELL_V1,shell,beams,(),fixed,(line,))


def _ground_wrench(definition,result):
    """Sum actual support reactions about the reference origin, by domain."""
    coordinates=definition.shell.coordinates_m
    ns=6*len(coordinates)
    # The example's explicit sequential IDs/order and geometry are shared with
    # its definition builder; native definition bytes are never decoded here.
    beam_points=_beam_reference_points(2*len(definition.beams))
    outputs={}
    for name,xyzs,start in (("shell",coordinates,0),
                            ("beam",beam_points,ns)):
        force=np.zeros(3);moment=np.zeros(3)
        for row,xyz in enumerate(xyzs):
            reaction=result.reactions[start+6*row:start+6*row+6]
            force+=reaction[:3];moment+=reaction[3:]+np.cross(xyz,reaction[:3])
        outputs[name]={"force_n":force.tolist(),"moment_about_origin_nm":moment.tolist()}
    return outputs


def run(intervals=16):
    definition=make_definition(intervals)
    owner=ss.create_analysis(definition=definition)
    try:
        reports={}
        for name,wrench in LOADS.items():
            result=owner.solve_linear((ss.NodalAction("beam",101+intervals,wrench),))
            raw=owner.checkpoint()
            restored=ss.create_analysis(definition=definition)
            try:
                recovered=restored.restore(raw)
                if raw!=restored.checkpoint(): raise AssertionError("reference snapshot bytes changed")
                for field in ("shell_displacement","beam_coordinates","multipliers","reactions"):
                    np.testing.assert_array_equal(getattr(result,field),getattr(recovered,field))
            finally:restored.close()
            ground=_ground_wrench(definition,result)
            tip=result.beam_coordinates[6*intervals:6*intervals+6]
            external_force=np.asarray(wrench[:3]);external_moment=np.asarray(wrench[3:])
            external_moment+=np.cross((1.,0.,.0675),external_force)
            force=external_force+sum((np.asarray(x["force_n"]) for x in ground.values()),np.zeros(3))
            moment=external_moment+sum((np.asarray(x["moment_about_origin_nm"]) for x in ground.values()),np.zeros(3))
            scale=max(np.linalg.norm(external_force),np.linalg.norm(external_moment),1.)
            balance=max(np.linalg.norm(force),np.linalg.norm(moment))/scale
            if balance>1e-8: raise AssertionError("physical ground-reaction balance failed")
            reports[name]={"status":result.status,"applied_work_j":float(np.asarray(wrench)@tip),
                "tip_translation_m":tip[:3].tolist(),"tip_rotation_rad":tip[3:].tolist(),
                "beam_nodal_root_spin_rad":float(result.beam_coordinates[5]),
                "ground_reactions":ground,"scaled_ground_balance":float(balance),
                "equilibrium_ratio":result.free_equilibrium_ratio,"scaled_weak_gap":result.constraint_max,
                "restart_bytes_equal":True}
        preflight=owner.preflight()
        if preflight["reference_linear_factorizations"]!=1:
            raise AssertionError("eligible static operator refactorized")
        return {"source":anyfesolver.__file__,"qualification":owner.capabilities["status"],
            "shell_elements":intervals,"beam_elements":intervals//2,
            "support_semantics":"shell-root clamp plus beam nodal root ground components 0..4; normal spin free",
            "reaction_semantics":"ground reactions, not concentrated beam-to-shell endpoint transfer",
            "line_semantics":"integrated BL5 top-surface/director bond; no additional endpoint ties",
            "factorizations":preflight["reference_linear_factorizations"],"cases":reports}
    finally:owner.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals",type=int,default=16)
    args=parser.parse_args()
    print(json.dumps(run(args.intervals),sort_keys=True))
