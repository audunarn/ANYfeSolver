"""Run one output-only C1 F5 CalculiX stress replay under the frozen cap.

This development diagnostic reuses the archived pre-opened C3D20R model.
Only an integration-point stiffener stress print request is added to the
deck. It does not qualify a native beam/shell attachment or change a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from scripts.b3_e4_f03_ccx_solid_bond_screen import (
    parse_nodal_vectors, split_left_root_support_wrenches)


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'.qualification/b3-ge-e4-pl'
SOURCE=BASE/'f03-ccx-c1-preopen-combined-factor5-dev-20260926-v1'
GAP=BASE/'f03-c1-factor5-admissible-torque-direction-dev-20260926-v1/result.json'
EXE=Path('C:/Github/ANYfeSolver/.qualification/b3-ge-e4-pl/calculix-223/CalculiX-2.23.0-win-x64/bin/ccx.exe')
EXPECTED_EXE='27a9426a1ee2f72c38fbe892e0ffaf38e14867628ffd0235770261b6f3ec9c5f'
TARGET=BASE/'f03-c1-factor5-stiffener-ip-stress-dev-20260926-v1'
STEM='solid-5-tip_mx'
INSERT='*EL PRINT,ELSET=Stiffener\nS\n'
LIMIT_SECONDS=120


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved() -> tuple[Path,dict]:
    folders=list(SOURCE.rglob('model.json'))
    if len(folders)!=1:
        raise ValueError('one archived C1 model required')
    model_path=folders[0]
    model=json.loads(model_path.read_text())
    required={'factor':5,'element_family':'C3D20R',
              'bond_policy':'terminal','contact_policy':'none',
              'boundary_policy':'left_root','tip_load_map':'quadratic_face',
              'bar_clearance_um':2}
    if any(model.get(k)!=v for k,v in required.items()):
        raise ValueError('archived model outside frozen diagnostic')
    gap=json.loads(GAP.read_text())['fields']['base_100_Nm']
    if (gap['opening_force_N']!=-1000 or gap['torque_Nm']!=100 or
            gap['continuous_face_gap_bound']['status']!='POSITIVE_SAVED_FIELD_FACE_GAP_BOUND' or
            gap['continuous_face_gap_bound']['minimum_nonseam_gap_bound_m']<=0):
        raise ValueError('archived pre-opened field lacks positive gap bound')
    return model_path.parent,model


def prepare() -> None:
    source_folder,model=saved()
    source_deck=source_folder/(STEM+'.inp')
    if not source_deck.is_file() or sha(EXE)!=EXPECTED_EXE:
        raise ValueError('archived deck or pinned CalculiX executable unavailable')
    deck=source_deck.read_text()
    if deck.count('*END STEP\n')!=1 or '*EL PRINT' in deck:
        raise ValueError('cannot make unique output-only deck change')
    edited=deck.replace('*END STEP\n',INSERT+'*END STEP\n')
    if edited.replace(INSERT,'')!=deck:
        raise AssertionError('non-output deck change')
    if TARGET.exists():
        raise FileExistsError('versioned stress replay root already exists')
    TARGET.mkdir(parents=True)
    (TARGET/(STEM+'.inp')).write_text(edited)
    manifest={
        'classification':'bounded development diagnostic; not G0B acceptance',
        'question':'Can saved-model integration-point stress resolve the physical bond-side transfer obscured by FRD displacement rounding?',
        'outcome_rule':'First require identical physical deck, positive saved-field gap, solved state/reaction parity and reconstructed support wrench. Passing local stress transfer can guide end-section versus native-pose diagnosis; failure is inconclusive, not formulation failure.',
        'source_git':subprocess.run(('git','rev-parse','HEAD'),cwd=ROOT,
                                    capture_output=True,text=True,check=True).stdout.strip(),
        'runner_sha256':sha(Path(__file__)),
        'original_deck_sha256':sha(source_deck),
        'stress_deck_sha256':sha(TARGET/(STEM+'.inp')),
        'original_model_sha256':sha(source_folder/'model.json'),
        'gap_evidence_sha256':sha(GAP),
        'executable_sha256':sha(EXE),
        'output_only_deck_diff':INSERT,
        'omp_threads':1,'timeout_seconds':LIMIT_SECONDS,
        'source_folder':str(source_folder.relative_to(ROOT)),
        'limitations':['3D solid reference with no contact outside the shared patch',
                       'one stress-output replay; native line and 3D section idealizations differ',
                       'integration-point stresses are not automatically face tractions']}
    (TARGET/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('prepared',TARGET,'deck',manifest['stress_deck_sha256'])


def execute() -> None:
    source_folder,model=saved()
    manifest=json.loads((TARGET/'manifest.json').read_text())
    deck=TARGET/(STEM+'.inp')
    if (sha(deck)!=manifest['stress_deck_sha256'] or
            sha(source_folder/(STEM+'.inp'))!=manifest['original_deck_sha256'] or
            sha(EXE)!=manifest['executable_sha256'] or
            (TARGET/'result.json').exists()):
        raise ValueError('prepared identities changed or replay already executed')
    environment=os.environ.copy()
    environment['OMP_NUM_THREADS']='1'
    start=time.monotonic()
    timed_out=False
    code=None
    try:
        process=subprocess.run((str(EXE),'-i',STEM),cwd=TARGET,
                               env=environment,capture_output=True,
                               timeout=LIMIT_SECONDS,check=False)
        code=process.returncode
        stdout,stderr=process.stdout,process.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out=True
        stdout=exc.stdout or b''
        stderr=exc.stderr or b''
    elapsed=time.monotonic()-start
    (TARGET/'stdout.log').write_bytes(stdout)
    (TARGET/'stderr.log').write_bytes(stderr)
    result={'elapsed_s':elapsed,'timed_out':timed_out,'exit_code':code,
            'status':'FAILED_OR_INCOMPLETE','source_git':manifest['source_git']}
    dat=TARGET/(STEM+'.dat')
    frd=TARGET/(STEM+'.frd')
    if code==0 and dat.is_file() and frd.is_file():
        old_dat=(source_folder/(STEM+'.dat')).read_text(errors='replace')
        new_dat=dat.read_text(errors='replace')
        supports=set(model['boundary_node_ids'])
        old_rf=parse_nodal_vectors(old_dat,supports,'forces','SUPPORTS')
        new_rf=parse_nodal_vectors(new_dat,supports,'forces','SUPPORTS')
        rf_error=max(float(np.max(abs(new_rf[node]-old_rf[node])))
                     for node in supports)
        old_w=split_left_root_support_wrenches(model,old_rf)
        new_w=split_left_root_support_wrenches(model,new_rf)
        result.update({'maximum_support_RF_difference_N':rf_error,
                       'original_support_wrenches_N_Nm':old_w,
                       'replay_support_wrenches_N_Nm':new_w,
                       'stress_header_present':('stresses' in new_dat.lower()),
                       'dat_sha256':sha(dat),'frd_sha256':sha(frd)})
        if rf_error<1e-4 and result['stress_header_present']:
            result['status']='COMPLETE_OUTPUT_ONLY_REPLAY'
    (TARGET/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(result['status'],f'{elapsed:.3f} s',
          'support RF max difference',result.get('maximum_support_RF_difference_N'))
    if result['status']!='COMPLETE_OUTPUT_ONLY_REPLAY':
        raise RuntimeError('stress replay failed or did not preserve archived support reactions')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','execute'))
    args=parser.parse_args()
    (prepare if args.mode=='prepare' else execute)()


if __name__=='__main__':
    main()
