"""Experimental nonmatching BL5 line with signed translation-force stages.

Three Q4 shells and two native B3 elements share a one-metre attachment.
BL5 constrains integrated position/director gaps and releases normal spin;
it is not a six-channel welded bond or a qualified physical reference.
The objective Q4 policy is explicit. Only elastic static use is demonstrated;
the generalized section's inertia metadata is not a qualified spectral model.
Run this file with an installed ANYfeSolver or the repository source environment.
"""
from contextlib import ExitStack
import json

import numpy as np

from anyfesolver import b3_ge, stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled


def definition():
    e, nu = 210e9, .3
    g = e/(2*(1+nu))
    section = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((e*.01,g*.008,g*.008,g*1.6666666666666667e-5,
                 e*8.333333333333333e-6,e*8.333333333333333e-6)),
        np.eye(6),1e20,1e9)
    axis = np.array([(x,0.,.0675) for x in (0.,.25,.5,.75,1.)])
    beams = tuple(b3_ge.define_beam('b3-ge',i+1,
        (101+2*i,102+2*i,103+2*i),axis[2*i:2*i+3],
        np.tile(np.eye(3),(3,1,1)),section,np.eye(6)) for i in range(2))
    xs = (0.,1/3,2/3,1.)
    shell = ss.ShellPanel(tuple(range(1,9)),
        tuple((x,y,0.) for y in (0.,.2) for x in xs),
        tuple((i,i+1,i+5,i+4) for i in range(1,4)),.035,e,nu,7850.)
    line = ss.ExperimentalBL5Line((1,2,3,4),((0,0),(0,1),(1,0),(1,1)),
                                  multiplier_span=1)
    fixed = tuple(ss.FixedCoordinate('shell',n,d) for n in (1,5) for d in range(6))
    return ss.LinearDefinition(ss.B3_GE_E4_PL_STIFFENED_SHELL_V1,
        shell,beams,(),fixed,(line,),q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)


def workflow():
    with ExitStack() as stack:
        model = definition()
        direct = ss.create_analysis(definition=model); stack.callback(direct.close)
        restored = ss.create_analysis(definition=model); stack.callback(restored.close)
        programme = ss.StagedDeadLoadProgramme(direct.preflight()['identity'],tuple(
            (ss.NodalAction('beam',105,(0.,0.,f,0.,0.,0.)),) for f in (100.,-100.)))
        targets = (.5,1.,1.25,1.75,2.)
        control = ss.AdaptiveIncrementControl(2,16)
        token = CancellationToken()
        def observe(result):
            if result.load_position == 1.: token.cancel('accepted line stage')
        try:
            direct.solve(programme,targets,increment_control=control,
                         cancellation_token=token,on_accepted=observe)
        except SolveCancelled:
            pass
        else:
            raise AssertionError('accepted-prefix cancellation required')
        prefix = direct.recover()
        assert prefix.load_position == 1.
        direct.rebase_rotations(prefix.generation)
        restored.restore(direct.checkpoint())
        result = direct.solve(programme,targets,increment_control=control)
        repeated = restored.solve(programme,targets,increment_control=control)
        assert result.load_programme_complete and repeated.load_programme_complete
        assert direct.checkpoint() == restored.checkpoint()
        assert result.free_equilibrium_ratio <= 1e-8 and result.constraint_max <= 1e-9
        return dict(status=result.status,experimental=True,production_qualified=False,
            model_identity=direct.preflight()['identity'],programme_identity=programme.identity,
            shell_elements=3,beam_elements=2,accepted_positions=result.accepted_factors,
            final_position=result.load_position,line_work_samples=len(result.beam_shell_line_work),
            tip_position_m=result.beam_positions[-1].tolist(),
            equilibrium=result.free_equilibrium_ratio,compatibility=result.constraint_max,
            restart_identical=True)


if __name__ == '__main__':
    print(json.dumps(workflow(),indent=2))
