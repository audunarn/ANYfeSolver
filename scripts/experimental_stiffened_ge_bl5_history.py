"""Installed experimental nonmatching BL5 load/unload/restart example.

Position/top-surface director bond, normal spin released. Elastic response
only; this example neither selects a multiplier basis nor qualifies BL5.
"""
import argparse
import json
import numpy as np
import anyfesolver
from anyfesolver import b3_ge, stiffened_shell as ss


def make_definition(basis="p0"):
    e,nu=210e9,.3; g=e/(2*(1+nu))
    section=b3_ge.EllipsoidalGeneralizedSection(np.diag((
        e*.01,g*.008,g*.008,g*1.6666666666666667e-5,
        e*8.333333333333333e-6,e*8.333333333333333e-6)),np.eye(6),1e20,1e9)
    beam_points=np.array([(x,0.,.0675) for x in (0.,.25,.5,.75,1.)])
    beams=tuple(b3_ge.define_beam("b3-ge",i+1,(101+2*i,102+2*i,103+2*i),
        beam_points[2*i:2*i+3],np.tile(np.eye(3),(3,1,1)),section,np.eye(6)) for i in range(2))
    xs=(0.,1/3,2/3,1.)
    shell=ss.ShellPanel(tuple(range(1,9)),
        tuple((x,y,0.) for y in (0.,.2) for x in xs),
        ((1,2,6,5),(2,3,7,6),(3,4,8,7)),.035,e,nu,7850.)
    line=ss.ExperimentalBL5Line((1,2,3,4),((0,0),(0,1),(1,0),(1,1)),basis=basis)
    fixed=tuple(ss.FixedCoordinate("shell",n,j) for n in (1,5) for j in range(6))
    return ss.LinearDefinition(ss.B3_GE_E4_PL_STIFFENED_SHELL_V1,shell,beams,(),fixed,(line,))


def run(basis="p0"):
    definition=make_definition(basis)
    actions=(ss.NodalAction("beam",105,(0.,0.,100.,0.,0.,0.)),)
    owner=ss.create_analysis(definition=definition)
    restored=ss.create_analysis(definition=definition)
    try:
        first=owner.solve(actions,targets=(.5,1.,.75))
        saved=owner.checkpoint(); restored.restore(saved)
        assert saved==restored.checkpoint()
        a=owner.solve(actions,targets=(.9,)); b=restored.solve(actions,targets=(.9,))
        assert owner.checkpoint()==restored.checkpoint()
        np.testing.assert_array_equal(a.beam_positions,b.beam_positions)
        return {"status":a.status,"qualification":owner.capabilities["status"],
            "source":anyfesolver.__file__,"basis":basis,
            "shell_elements":3,"beam_elements":2,"halfcell_count":4,
            "reference_segments":owner.preflight()["experimental_line_samples"]//2,
            "accepted_factors":a.accepted_factors,"restart_bytes_equal":True,
            "equilibrium_ratio":a.free_equilibrium_ratio,"scaled_weak_gap":a.constraint_max,
            "station_records":len(a.beam_shell_line_work),
            "tip_position_m":a.beam_positions[-1].tolist(),
            "checkpoint_bytes":len(saved)}
    finally: owner.close(); restored.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis",choices=("p0","continuous_p1_coarse"),default="p0")
    args=parser.parse_args()
    print(json.dumps(run(args.basis),sort_keys=True))
