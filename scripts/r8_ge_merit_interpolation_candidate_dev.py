"""Stdlib-only preparation of an unapplied connected candidate; never imports it."""
from pathlib import Path
from hashlib import sha256
import ast
import difflib
import json

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/agent_plans/development/r8-ge-paired-adaptive-20261010/execution-v4/fixed-correction-diagnostic-v1/saved-merit-remainder-v1/interpolation-candidate-v1'
POLICY='PAIRED_FIXED_WORK_QUADRATIC_V1'

def replace(source,old,new):
    if source.count(old)!=1:raise ValueError(('nonunique source seam',old,source.count(old)))
    return source.replace(old,new)

def prepare():
    DOC.mkdir(parents=True,exist_ok=True)
    changes={};records=[]
    for name in ('_stiffened_shell_displacement_corrector.py','_stiffened_shell_displacement_control.py',
                 '_stiffened_shell_displacement_adaptive.py','_stiffened_shell_ge_experimental.py','stiffened_shell.py'):
        path=ROOT/'src/anyfesolver'/name;original=path.read_text();source=original
        if name.endswith('corrector.py'):
            helper='''
PAIRED_FIXED_WORK_QUADRATIC_V1="PAIRED_FIXED_WORK_QUADRATIC_V1"


def _quadratic_merit_amount(old,trial,gradient,amount):
    """Proposal only: gradient is d(||W R||^2)/dt divided by old^2.

    Normalize both merits by their maximum. Invalid/non-descent quadratics
    retain geometric reduction; no norm floor or predicted acceptance.
    """
    from math import isfinite
    fallback=.5*amount
    if not all(isfinite(v) for v in (old,trial,gradient,amount)) or old<=0. or trial<0. or gradient>=0. or amount<=0.:
        return fallback
    scale=max(old,trial);r=old/scale;s=trial/scale
    slope=(gradient*r)*r
    denominator=s*s-r*r-slope*amount
    if not isfinite(denominator) or denominator<=0.:return fallback
    proposal=amount*((-slope*amount)/(2.*denominator))
    return proposal if isfinite(proposal) and 0.<proposal<amount else fallback

'''
            source=replace(source,'def correct(driver,',helper+'def correct(driver,')
            source=replace(source,'trial_policy=None, event_records=None, force_seed_sha256=None):',
                           'trial_policy=None, event_records=None, force_seed_sha256=None, merit_globalization=None):')
            source=replace(source,'    owner=driver.owner\n','''    owner=driver.owner
    if merit_globalization is not None:
        from ._stiffened_shell_affine_endpoint import is_paired_displacement
        if (merit_globalization!=PAIRED_FIXED_WORK_QUADRATIC_V1 or
                not is_paired_displacement(owner) or trial_policy is not None):
            raise ValueError("paired fixed-work interpolation policy required")
''')
            source=replace(source,'        old=merit(state,evaluation)\n','''        old=merit(state,evaluation)
        interpolating=merit_globalization is not None
        next_amount=1.
        gradient=float('nan')
        if interpolating and np.isfinite(old) and old>0.:
            # Complete fixed free-row work metric and actual bordered action.
            action=evaluation[1]@step-direction*delta
            with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                gradient=float(2.*np.dot(evaluation[0][free]*weights[free]/old,
                                        action[free]*weights[free]/old))
''')
            source=replace(source,'            amount=initial_amount*.5**backtrack\n',
                           '            amount=next_amount if interpolating else initial_amount*.5**backtrack\n')
            source=replace(source,'            try:\n                candidate=driver._advance(state,step,amount)\n',
                           '            if interpolating and (not np.isfinite(amount) or amount<=0.):\n                raise failure("nonrepresentable interpolated trial amount",iteration)\n            try:\n                candidate=driver._advance(state,step,amount)\n')
            source=replace(source,'            if native_cutback is not None and native_cutback<.5 and backtrack:',
                           '            if not interpolating and native_cutback is not None and native_cutback<.5 and backtrack:')
            source=replace(source,'                rejected+=1;continue\n',
                           '                rejected+=1\n                if interpolating:next_amount=.5*amount\n                continue\n')
            source=replace(source,'            rejected+=1\n        else:\n',
                           '            rejected+=1\n            if interpolating:next_amount=_quadratic_merit_amount(old,merit(candidate,changed),gradient,amount)\n        else:\n')
        elif name.endswith('displacement_control.py'):
            source=replace(source,'from ._stiffened_shell_displacement_corrector import correct',
                           'from ._stiffened_shell_displacement_corrector import correct,PAIRED_FIXED_WORK_QUADRATIC_V1')
            source=replace(source,'    force_seed_sha256: str | None = field(default=None,kw_only=True)\n',
                           '    force_seed_sha256: str | None = field(default=None,kw_only=True)\n    merit_globalization: str | None = field(default=None,kw_only=True)\n')
            source=replace(source,'    def __post_init__(self):\n','''    def __post_init__(self):
        if self.merit_globalization is not None and (
                self.merit_globalization!=PAIRED_FIXED_WORK_QUADRATIC_V1 or
                self.force_seed_sha256 is None or self.trial_policy is not None):
            raise ValueError("explicit paired fixed-work interpolation policy and seed required")
''')
            source=replace(source,'    paired=is_paired_displacement(owner)\n','''    paired=is_paired_displacement(owner)
    if getattr(control,'merit_globalization',None) is not None and (
            not paired or _adaptive is None or
            _adaptive.merit_globalization!=control.merit_globalization):
        raise ValueError("fixed-work interpolation requires matching paired adaptive context")
''')
            source=replace(source,'trial_policy=control.trial_policy,force_seed_sha256=control.force_seed_sha256)',
                           'trial_policy=control.trial_policy,force_seed_sha256=control.force_seed_sha256,\n            **({} if control.merit_globalization is None else {"merit_globalization":control.merit_globalization}))')
        elif name.endswith('displacement_adaptive.py'):
            source=replace(source,'from ._stiffened_shell_displacement_corrector import DisplacementCorrectorFailure',
                           'from ._stiffened_shell_displacement_corrector import DisplacementCorrectorFailure,PAIRED_FIXED_WORK_QUADRATIC_V1')
            source=replace(source,'MAGICS=(MAGIC,EVENT_MAGIC,NATIVE_MAGIC,PAIRED_MAGIC)',
                           'PAIRED_INTERPOLATION_MAGIC=b"ANYSSDCP2"\nMAGICS=(MAGIC,EVENT_MAGIC,NATIVE_MAGIC,PAIRED_MAGIC,PAIRED_INTERPOLATION_MAGIC)')
            source=replace(source,'    force_seed_sha256: str | None = None\n','    force_seed_sha256: str | None = None\n    merit_globalization: str | None = None\n')
            source=replace(source,'    def __post_init__(self):\n','''    def __post_init__(self):
        if self.merit_globalization is not None and (
                self.merit_globalization!=PAIRED_FIXED_WORK_QUADRATIC_V1 or
                self.force_seed_sha256 is None or self.trial_policy is not None):
            raise ValueError("paired fixed-work interpolation context required")
''')
            source=replace(source,"    if paired:value['force_seed_sha256']=context.force_seed_sha256\n",
                           "    if paired:value['force_seed_sha256']=context.force_seed_sha256\n    if context.merit_globalization is not None:value['merit_globalization']=context.merit_globalization\n")
            source=replace(source,'    header=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("ascii")\n',
                           '    if context.merit_globalization is not None:magic=PAIRED_INTERPOLATION_MAGIC\n    header=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("ascii")\n')
            source=replace(source,'    if raw.startswith(PAIRED_MAGIC)!=paired or (not paired and raw.startswith(NATIVE_MAGIC)!=native):',
                           '    interpolating=raw.startswith(PAIRED_INTERPOLATION_MAGIC)\n    if raw.startswith((PAIRED_MAGIC,PAIRED_INTERPOLATION_MAGIC))!=paired or (not paired and raw.startswith(NATIVE_MAGIC)!=native):')
            source=replace(source,"        if paired:keys.add('force_seed_sha256')\n",
                           "        if paired:keys.add('force_seed_sha256')\n        if interpolating:keys.add('merit_globalization')\n")
            source=replace(source,'        state=DisplacementIncrementState.from_payload(v["state"])\n','''        if interpolating and v['merit_globalization']!=PAIRED_FIXED_WORK_QUADRATIC_V1:
            raise ValueError("explicit paired interpolation wrapper policy required")
        state=DisplacementIncrementState.from_payload(v["state"])
''')
            source=replace(source,"                        force_seed_sha256=v.get('force_seed_sha256'))",
                           "                        force_seed_sha256=v.get('force_seed_sha256'),merit_globalization=v.get('merit_globalization'))")
            source=replace(source,'                context.force_seed_sha256!=control.force_seed_sha256):',
                           '                context.force_seed_sha256!=control.force_seed_sha256 or\n                context.merit_globalization!=control.merit_globalization):')
            source=replace(source,'                        control.force_seed_sha256)\n',
                           '                        control.force_seed_sha256,control.merit_globalization)\n')
            source=replace(source,'    paired=is_paired_displacement(owner)\n    if native:\n','''    paired=is_paired_displacement(owner)
    if control.merit_globalization is not None and not paired:
        raise ValueError("fixed-work interpolation requires paired adaptive owner")
    if native:
''')
        elif name.endswith('ge_experimental.py'):
            source=replace(source,'from ._stiffened_shell_displacement_adaptive import PAIRED_MAGIC as PAIRED_ADAPTIVE_MAGIC',
                           'from ._stiffened_shell_displacement_adaptive import PAIRED_MAGIC as PAIRED_ADAPTIVE_MAGIC,PAIRED_INTERPOLATION_MAGIC')
            source=replace(source,'if raw.startswith(PAIRED_ADAPTIVE_MAGIC):','if raw.startswith((PAIRED_ADAPTIVE_MAGIC,PAIRED_INTERPOLATION_MAGIC)):')
            source=replace(source,"                capabilities['arc_length']='not admitted for paired native material'",
                           "                capabilities['merit_globalization']='experimental opt-in paired fixed-work quadratic candidate; unqualified; actual original merit and physical acceptance unchanged'\n                capabilities['arc_length']='not admitted for paired native material'")
        else:
            source=replace(source,'from ._stiffened_shell_displacement_control import GENERALIZED_BRANCH_EVENT_TRIAL',
                           'from ._stiffened_shell_displacement_control import GENERALIZED_BRANCH_EVENT_TRIAL,PAIRED_FIXED_WORK_QUADRATIC_V1')
            source=replace(source,'__all__ += ["GENERALIZED_BRANCH_EVENT_TRIAL"]',
                           '__all__ += ["GENERALIZED_BRANCH_EVENT_TRIAL","PAIRED_FIXED_WORK_QUADRATIC_V1"]')
        ast.parse(source)
        target=DOC/('candidate_'+name)
        if target.exists() and target.read_text()!=source:
            prior=DOC/'superseded-preparation-v1';prior.mkdir(exist_ok=True)
            saved=prior/target.name
            if saved.exists():raise FileExistsError(saved)
            saved.write_bytes(target.read_bytes())
        target.write_text(source,encoding='utf-8',newline='\n')
        changes[name]=(original,source)
        records.append(dict(product=str(path),product_sha256=sha256(path.read_bytes()).hexdigest(),
                            candidate=str(target),candidate_sha256=sha256(target.read_bytes()).hexdigest()))
    patch=''.join(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                       fromfile='a/src/anyfesolver/'+name,tofile='b/src/anyfesolver/'+name))
                  for name,(old,new) in changes.items())
    for name in ('CONNECTED_CANDIDATE.patch','CANDIDATE_SOURCE_BINDINGS.json'):
        path=DOC/name
        if path.exists():
            prior=DOC/'superseded-preparation-v1';prior.mkdir(exist_ok=True)
            saved=prior/name
            if saved.exists():raise FileExistsError(saved)
            saved.write_bytes(path.read_bytes())
    (DOC/'CONNECTED_CANDIDATE.patch').write_text(patch,encoding='utf-8',newline='\n')
    (DOC/'CANDIDATE_SOURCE_BINDINGS.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(dict(status='UNAPPLIED_AST_ONLY',files=len(records),namespace=str(DOC))))

def freeze():
    """Static/hash checks only; this neither imports product nor runs controls."""
    records=json.loads((DOC/'CANDIDATE_SOURCE_BINDINGS.json').read_text())
    spans={}
    for row in records:
        for kind in ('product','candidate'):
            p=Path(row[kind]);assert sha256(p.read_bytes()).hexdigest()==row[kind+'_sha256']
            tree=ast.parse(p.read_text())
            if kind=='candidate':spans[p.name]={n.name:[n.lineno,n.end_lineno] for n in tree.body
                if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in
                ('correct','_quadratic_merit_amount','solve','Context','DisplacementControl','encode','decode','restore_paired','rebase')}
    controls=ROOT/'tests/test_r8_ge_merit_interpolation_candidate_dev.py'
    names=[n.name for n in ast.parse(controls.read_text()).body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]
    assert len(names)==8
    old=ROOT/'tests/test_r8_ge_paired_adaptive_dev.py'
    old_names=('test_source_paired_identity_and_legacy_header','test_source_public_solve_adaptive_dispatch',
               'test_source_restore_atomic_cancel_stale_and_progress','test_source_typed_cutback_and_durable_limit')
    found={n.name for n in ast.parse(old.read_text()).body if isinstance(n,ast.FunctionDef)}
    assert set(old_names)<=found
    frozen=json.loads((DOC.parent/'MERIT_BINDINGS.json').read_text())
    for relative,digest in frozen.items():assert sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
    for name,count in (('MERIT_SELECTED.json',7),('../saved-compliance-v1/COMPLIANCE_TERMINAL_SELECTED.json',110)):
        p=DOC.parent/name;selection=json.loads(p.read_text());assert selection['file_count']==count
        for relative,digest in selection['files'].items():assert sha256((p.parent/relative).read_bytes()).hexdigest()==digest,relative
    inventory=[str(controls.relative_to(ROOT)).replace('\\','/')+'::'+n for n in names]
    inventory += [str(old.relative_to(ROOT)).replace('\\','/')+'::'+n for n in old_names]
    readiness=dict(status='CONNECTED_CANDIDATE_UNAPPLIED_UNVERIFIED',product_files_unchanged=5,
        independent_review='MISTRAL_PENDING_AUTHOR_NOT_REVIEWER',source_controls=inventory,expected_pass=12,expected_skip=0,
        checks_actually_run=['stdlib AST parsing','exact hash comparison; original MERIT18 and selections7/110 unchanged'],
        no_numeric_imports_tests_or_saved_arithmetic=True,policy=POLICY,magic_selected='ANYSSDCP2',
        legacy_magics_unchanged=['ANYSSDCP1','ANYSSDCA1','ANYSSDCA2','ANYSSDCN1','ANYSSPD1'],
        same_corrector_limits=[20,8],same_adaptive_limits=[4,96],source_spans=spans,
        adoption_pending=['exact option/schema review and product application','fresh source/build/installed authority after B2 freeze'],
        verification_proposal='One connected 12-source-controls/build/installed material-cycle slice under root existing supervisor; no authority consumed here',
        controls_sha256=sha256(controls.read_bytes()).hexdigest(),existing_controls_sha256=sha256(old.read_bytes()).hexdigest(),
        candidate_patch_sha256=sha256((DOC/'CONNECTED_CANDIDATE.patch').read_bytes()).hexdigest(),
        prototype_limits='Domain-only failures halve; cancellation/memory/programming propagate; finite strictly interior proposal or geometric fallback; nonrepresentable amount typed refusal; no floors/caps/threshold changes',
        checkpoint_scope='Selected outer paired adaptive policy only; exact existing fixed inner. Pending cursor policy mismatch refuses. Isolated restore dispatch and rebase preserve outer policy. No new physical acceptance claim.')
    (DOC/'READINESS.json').write_text(json.dumps(readiness,indent=2)+'\n')
    selected={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p.read_bytes()).hexdigest()
              for p in sorted(DOC.rglob('*')) if p.is_file() and p.name!='CANDIDATE_SELECTED.json'}
    for p in (Path(__file__),controls):selected[str(p.relative_to(ROOT)).replace('\\','/')]=sha256(p.read_bytes()).hexdigest()
    (DOC/'CANDIDATE_SELECTED.json').write_text(json.dumps(dict(status=readiness['status'],file_count=len(selected),files=selected),indent=2)+'\n')
    print(json.dumps(dict(status=readiness['status'],source_nodes=len(inventory),selection_count=len(selected),
                          patch=readiness['candidate_patch_sha256'],controls=readiness['controls_sha256'],
                          readiness=sha256((DOC/'READINESS.json').read_bytes()).hexdigest(),
                          selected=sha256((DOC/'CANDIDATE_SELECTED.json').read_bytes()).hexdigest())))

if __name__=='__main__':prepare()
