"""Original registered16Q4/32S3 force/displacement path, NEW enriched profile.

No historical accepted field/checkpoint is a starting state. Historical JSON
supplies only the exact registered builder/control/load descriptor.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from anyfesolver.e4_pl_enriched import EnrichedProfileError
from anyfesolver.general_contact.enriched_overlap import QuadraturePolicy
from anyfesolver.general_contact.enriched_path import EnrichedPathOwner

INPUT=Path(__file__).resolve().parents[1]/'docs/reference_cases/enriched_d1_registered_inputs_20261019.json'


def build(mode='force',*,input_path=INPUT):
    from scripts.general_contact_d1_interior_press_dev import build_saved_cycle
    from scripts.general_contact_d1_displacement_cycle_dev import build_displacement_cycle,REGISTERED_INPUTS,NCP_SCALE_N
    if mode not in ('force','displacement'): raise EnrichedProfileError('registered path mode required')
    saved=json.loads(Path(input_path).read_bytes())
    if any(key not in saved or saved[key]!=value for key,value in REGISTERED_INPUTS.items()):
        raise EnrichedProfileError('exact original full16Q4/32S3 registered inputs required')
    model,definition,settings,program=(build_saved_cycle(saved) if mode=='force' else build_displacement_cycle(saved))
    owner=EnrichedPathOwner(model,definition,program,quadrature=QuadraturePolicy(1e-10,1e-8,1e-5,14144),
        force_scale_n=.018 if mode=='force' else NCP_SCALE_N,physical_force_scale_n=.018)
    if (owner.template.physical_dofs!=300 or owner.template.offset.size!=364
        or len(owner.template.cells)!=16 or len({p.definition.target_element for p in owner.template.pairs})!=32
        or owner.template.pressure_ids!=tuple(range(1,26)) or program.total_steps!=8 or owner._state is not None):
        raise EnrichedProfileError('full300+64/all25/force8-displacement8/fresh-state inventory required')
    return model,definition,owner,dict(mode=mode,input_sha256=hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        original_nodal_coordinates=300,added_physical_coordinates=64,total_coordinates=364,
        free_coordinates=owner.template.transform.shape[1],pressure_ids=list(owner.template.pressure_ids),
        program=program.contract(),physical_definition=owner.template.definition,
        load_scale_n=owner.criteria.force_scale_n,physical_force_moment_scale=.018,
        historical_state_reused=False,scientific_acceptance=False)


def run(output,*,mode='force',through=8,prefix=None,prefix_sha256=None,cancel=lambda:False,on_candidate=None):
    model,definition,owner,descriptor=build(mode)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if prefix is not None:
        if prefix_sha256 is None: raise EnrichedProfileError('trusted actual prefix SHA required')
        owner.restore(Path(prefix).read_bytes(),expected_sha256=prefix_sha256,cancel=cancel)
    def persist_accepted():
        actual=owner.checkpoint();step=owner.accepted_step
        staging=output/f'checkpoint-{step:02d}.pending'
        with staging.open('xb') as stream:
            stream.write(actual);stream.flush();os.fsync(stream.fileno())
        os.replace(staging,output/'checkpoint.json')
        return hashlib.sha256(actual).hexdigest()
    def publish(candidate):
        step=json.loads(candidate)['accepted_step']
        envelope=dict(status='CANDIDATE_NOT_COMMITTED',record=json.loads(candidate)['records'][-1],
            proposed_checkpoint_sha256=hashlib.sha256(candidate).hexdigest())
        with (output/f'candidate-{step:02d}.json').open('x',encoding='utf-8') as stream:
            json.dump(envelope,stream,sort_keys=True,allow_nan=False);stream.flush()
        if on_candidate is not None:on_candidate(owner,candidate)
    try:
        if owner._state is not None:persist_accepted()
        if type(through) is not int or not owner.accepted_step<=through<=owner.program.total_steps:
            raise EnrichedProfileError('invalid registered path through-step')
        while owner.accepted_step<through:
            record=owner.advance(cancel=cancel,before_publish=publish)
            checkpoint_sha=persist_accepted()
            candidate_path=output/f"candidate-{record['step_index']:02d}.json"
            receipt=dict(status='ACCEPTED_PHYSICAL_STEP',step_index=record['step_index'],
                candidate_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                checkpoint_sha256=checkpoint_sha)
            (output/f"commit-{record['step_index']:02d}.json").write_text(json.dumps(receipt,sort_keys=True)+'\n')
    except BaseException as primary:
        try:
            checkpoint=output/'checkpoint.json'
            (output/'failure.json').write_text(json.dumps(dict(status='NONPASS',primary_type=type(primary).__name__,
                message=str(primary),owner_accepted_step=owner.accepted_step,
                durable_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.is_file() else None,
                uncommitted_candidates_are_not_accepted=True),sort_keys=True)+'\n')
        except BaseException as secondary:primary.add_note('secondary public failure publication: '+repr(secondary))
        raise
    result=json.loads(owner.checkpoint())
    result.update(descriptor=descriptor,qualification_status='DEVELOPMENT_ONLY',
        status='completed' if owner.accepted_step==owner.program.total_steps else 'accepted_prefix',
        full_D1_acceptance=False,full_coordinate_work_reference='REQUIRED_SEPARATE_ASSESSMENT',
        pressure_stability='UNPROVEN',smooth_branch_tangent='REQUIRED_SEPARATE_ASSESSMENT')
    (output/'result.json').write_text(json.dumps(result,sort_keys=True,allow_nan=False)+'\n')
    return model,definition,owner,result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path);parser.add_argument('--mode',choices=('force','displacement'),default='force')
    parser.add_argument('--through',type=int,default=8)
    parser.add_argument('--prefix',type=Path);parser.add_argument('--prefix-sha256')
    args=parser.parse_args()
    run(args.output,mode=args.mode,through=args.through,prefix=args.prefix,prefix_sha256=args.prefix_sha256)
