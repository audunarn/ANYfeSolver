"""Experimental mixed native point panel: plastic loading, zero unload, restart.

The low-yield single fibre is a lifecycle control, not a capacity material law.
This example does not qualify point welds, line coupling, spectra or capacity.
"""
from pathlib import Path
from math import fsum
import argparse
import json
import numpy as np
from anyfesolver import b3_ge, stiffened_shell as ss


def definition():
    e,nu = 210e9,.3
    g = e/(2*(1+nu))
    generalized = b3_ge.EllipsoidalGeneralizedSection(np.diag((e*.01,g*.008,
        g*.008,g*1.6666666666666667e-5,e*8.333333333333333e-6,
        e*8.333333333333333e-6)),np.eye(6),1e20,1e9)
    arm = np.sqrt(8.333333333333333e-6/.01)
    fibres = tuple(b3_ge.Fibre(str(i),y*arm,z*arm,.0025,e,
        b3_ge.FlowCurve((0.,),(1e5 if i == 0 else 1e20,),1e9))
        for i,(y,z) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))))
    background = np.diag(np.sqrt((0.,g*.008,g*.008,g*1.6666666666666667e-5,0.,0.)))
    physical = b3_ge.PhysicalFibreSection(fibres,background)
    coordinates = np.array([(x,-.5,.1) for x in (0.,.5,1.,1.5,2.)])
    beams = tuple(b3_ge.define_beam('b3-ge',i+1,(101+2*i,102+2*i,103+2*i),
        coordinates[2*i:2*i+3],np.tile(np.eye(3),(3,1,1)),section,np.eye(6))
        for i,section in enumerate((generalized,physical)))
    shell = ss.ShellPanel((1,2,3,4,5,6),
        ((0.,-.5,0.),(1.,-.5,0.),(2.,-.5,0.),(0.,.5,0.),(1.,.5,0.),(2.,.5,0.)),
        ((1,2,5,4),(2,3,6,5)),.07,e,nu,7850.)
    return ss.LinearDefinition(ss.B3_GE_E4_PL_STIFFENED_SHELL_V1,shell,beams,
        (ss.PointConnection(1,101),ss.PointConnection(2,103),ss.PointConnection(3,105)),
        tuple(ss.FixedCoordinate('shell',node,axis) for node in (1,4) for axis in range(6)),
        q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)


def workflow(output):
    output = Path(output); output.mkdir(exist_ok=False)
    model = definition()
    owner = ss.create_analysis(definition=model)
    restarted = ss.create_analysis(definition=model)
    try:
        actions = (ss.NodalAction('beam',105,(0.,0.,1000.,0.,0.,0.)),)
        loads = ss.StagedDeadLoadProgramme(owner.preflight()['identity'],
            (actions,(),actions),zero_load_reference_endpoint=1)
        options = dict(max_iterations=20,max_backtracks=12)
        loaded = owner.solve(loads,(.5,1.),**options)
        assert any(fsum(row.incremental_dissipation_j) > 0. for row in loaded.beam_fibre_energy)
        zero = owner.solve(loads,(1.5,2.),**options)
        assert not np.any(zero.applied_dead_load) and zero.load_equilibrium_reference_used
        assert zero.free_equilibrium_ratio <= 1e-8 and zero.constraint_max <= 1e-9
        raw = owner.checkpoint(); (output/'zero.checkpoint').write_bytes(raw)
        (output/'load-programme.json').write_bytes(loads.serialize())
        restarted.restore(raw)
        assert restarted.checkpoint() == raw
        a = owner.solve(loads,(2.5,3.),**options)
        b = restarted.solve(loads,(2.5,3.),**options)
        assert owner.checkpoint() == restarted.checkpoint()
        assert a.beam_fibre_energy == b.beam_fibre_energy
        result = dict(status=a.status,positions=a.accepted_factors,
            zero_equilibrium_ratio=zero.free_equilibrium_ratio,zero_gap=zero.constraint_max,
            zero_reference_endpoint=zero.load_equilibrium_reference_endpoint,
            zero_reference_policy=zero.load_equilibrium_reference_policy,
            restart_exact=True,production_qualified=False)
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        return result
    finally:
        owner.close(); restarted.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(workflow(parser.parse_args().output)))
