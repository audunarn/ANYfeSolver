"""Public connected native TransientConfig workflow; source UNVERIFIED."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from anyfesolver.dynamics import TransientConfig
from anyfesolver.ge_beam3_native_connected_dynamics import solve_transient
from examples.native_connected_dynamic_dev import build,velocity


def run(steps=4,dt=.00025, *, checkpoint=None):
    owner=build();config=TransientConfig(dt=dt,t_end=steps*dt,
        initial_velocity=velocity(owner),save_every=1)
    result=solve_transient(owner,config,checkpoint=checkpoint,
        expected_sha256=sha256(checkpoint).hexdigest() if checkpoint is not None else None)
    return owner,result


def run_balance_case(directory, *, clamped=False):
    """Independent fixed-case development assessment through the public API.

    Every trajectory starts with a fresh owner; only the authentic prefix is
    supplied to the resume route. Raw completed trajectories persist before
    independent predicates. No accepted state from an earlier run seeds a solve.
    """
    from examples.native_connected_balance_reference import (
        BASE_DT,BASE_STEPS,case_velocity,require_case_definition,assess_trajectory)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    records={};summaries={}

    def execute(name,steps,dt,checkpoint=None):
        owner=build(clamped=clamped);require_case_definition(owner,clamped)
        config=TransientConfig(dt=dt,t_end=steps*dt,initial_velocity=case_velocity(clamped),save_every=1)
        result=solve_transient(owner,config,checkpoint=checkpoint,
            expected_sha256=sha256(checkpoint).hexdigest()if checkpoint is not None else None)
        data=result.to_dict()
        if not result.success or data['accepted_steps']!=steps or data['accepted_time']!=steps*dt:raise AssertionError('complete registered public trajectory/horizon required')
        # An accepted result is evidence, even if the next independent gate fails.
        (directory/(name+'.json')).write_text(json.dumps(data,sort_keys=True,allow_nan=False)+'\n')
        (directory/(name+'.checkpoint')).write_bytes(result.checkpoint)
        require_case_definition(owner,clamped)
        summary=assess_trajectory(data,clamped)
        (directory/(name+'.balance.json')).write_text(json.dumps(summary,sort_keys=True,allow_nan=False)+'\n')
        records[name]=(result,data);summaries[name]=summary
        return result

    full=execute('full',BASE_STEPS,BASE_DT)
    prefix=execute('prefix',BASE_STEPS//2,BASE_DT)
    resumed=execute('resumed',BASE_STEPS,BASE_DT,prefix.checkpoint)
    if resumed.checkpoint!=full.checkpoint or resumed.to_dict()!=full.to_dict():raise AssertionError('exact fresh-owner full/split public result and checkpoint required')
    execute('refined',2*BASE_STEPS,BASE_DT/2.)
    refinement=[]
    for coarse,fine in zip(summaries['full']['steps'],summaries['refined']['steps'][1::2]):
        refinement.append(dict(coarse_index=coarse['index'],fine_index=fine['index'],
            coarse_time=coarse['end_time'],fine_time=fine['end_time'],
            physical_energy_difference_j=fine['physical_energy']-coarse['physical_energy'],
            linear_momentum_difference_ns=(np.asarray(fine['linear_momentum'])-coarse['linear_momentum']).tolist(),
            angular_momentum_difference_nms=(np.asarray(fine['angular_momentum'])-coarse['angular_momentum']).tolist()))
    manifest={p.name:sha256(p.read_bytes()).hexdigest()for p in sorted(directory.iterdir())if p.is_file()}
    summary=dict(status='PASS_SCOPED_CONNECTED_PHYSICAL_BALANCE_DEVELOPMENT_ONLY',
        clamped=clamped,base_dt=BASE_DT,base_steps=BASE_STEPS,refinement_dt=BASE_DT/2.,
        exact_fresh_split=True,assessments=summaries,artifacts=manifest,engineering_accepted=False,
        common_time_refinement=refinement,refinement_ratio_gate=None,
        conservation_is_separate_from_oracle_parity=True,historical_single_element_gates_closed_here=False)
    (directory/'ASSESSMENT.json').write_text(json.dumps(summary,sort_keys=True,allow_nan=False)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--steps',type=int,default=4)
    parser.add_argument('--balance-output',type=Path)
    parser.add_argument('--clamped',action='store_true')
    args=parser.parse_args()
    if args.balance_output is not None:
        if args.steps!=4:parser.error('balance case retains exactly four base steps')
        summary=run_balance_case(args.balance_output,clamped=args.clamped)
        print(json.dumps(dict(status=summary['status'],clamped=args.clamped,exact_fresh_split=True,
            assessment=str(args.balance_output/'ASSESSMENT.json'),engineering_accepted=False)))
        return
    if args.clamped:parser.error('--clamped belongs to the explicit balance case')
    owner,result=run(args.steps)
    print(json.dumps(dict(status=result.to_dict()['status'],accepted_steps=result.to_dict()['accepted_steps'],
        profile=result.to_dict()['profile'],checkpoint_bytes=len(result.checkpoint),engineering_accepted=False)))


if __name__=='__main__':main()
