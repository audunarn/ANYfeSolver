"""Stdlib saved-metadata preparation of one whole-direction branch-domain candidate.

This command evaluates no model, material, assembly, tangent or numerical test.
Its radius is an affine-fibre candidate, not a complete generalized/J2 certificate
or authority to run/adopt a different direction.
"""
from pathlib import Path
from decimal import Decimal as D, localcontext
import argparse
import ast
import hashlib
import json
import math
import struct
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
TANGENT=ROOT/'docs/agent_plans/development/r8-f06-log-domain-20261010/paired-displacement-v1/loaded-x-v1/active-tangent-v1'
DOC=TANGENT/'branch-domain-v1'
SAVED=TANGENT/'attempt-installed-assessment1'
TAU=D('0.000001')
STEPS=(D('0.0001'),D('0.00001'),D('0.000001'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair(words):
    if len(words)!=2:
        raise ValueError('two canonical high/low scalar words required')
    return D.from_float(float(words[0]))+D.from_float(float(words[1]))


def read_vector(path,name):
    """Read only an existing 1-D binary64 NPY member; no NumPy import/pickle."""
    with zipfile.ZipFile(path) as archive:
        raw=archive.read(name+'.npy')
    if raw[:6]!=b'\x93NUMPY':
        raise ValueError('NPY magic required')
    version=tuple(raw[6:8])
    if version==(1,0):
        length=int.from_bytes(raw[8:10],'little');start=10
    elif version in ((2,0),(3,0)):
        length=int.from_bytes(raw[8:12],'little');start=12
    else:
        raise ValueError('unsupported NPY header version')
    header=ast.literal_eval(raw[start:start+length].decode('ascii'))
    if header['descr']!='<f8' or header['fortran_order'] or len(header['shape'])!=1:
        raise ValueError('existing 1-D little-endian binary64 vector required')
    count=header['shape'][0];data=raw[start+length:]
    if len(data)!=8*count:
        raise ValueError('NPY size mismatch')
    values=list(struct.unpack('<'+str(count)+'d',data))
    if not all(math.isfinite(x) for x in values):
        raise ValueError('finite saved vector required')
    return values


def captured_law():
    """Extract existing fixture parameters literally; never import its factory."""
    path=ROOT/'scripts/r8_f06_native_material_breadth_dev.py'
    snapshot=SAVED/'source-snapshots'/path.name
    if sha(path)!=sha(snapshot):
        raise ValueError('saved fixture definition binding changed')
    tree=ast.parse(path.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='definition')
    calls=[n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    fibres=[n for n in calls if n.func.id=='Fibre']
    curves=[n for n in calls if n.func.id=='FlowCurve']
    if len(fibres)!=1 or len(curves)!=1:
        raise ValueError('one explicit captured physical-fibre law required')
    young=ast.literal_eval(fibres[0].args[4])
    points=ast.literal_eval(curves[0].args[0]);stress=ast.literal_eval(curves[0].args[1])
    hardening=ast.literal_eval(curves[0].args[2])
    if points!=(0.,) or len(stress)!=1 or young<=0 or hardening<=0:
        raise ValueError('this prepared affine candidate admits only the existing linear hardening law')
    return dict(young=young,yield_stress=stress[0],hardening=hardening,
                source=path.relative_to(ROOT).as_posix(),sha256=sha(path),
                fibre_span=[fibres[0].lineno,fibres[0].end_lineno],
                curve_span=[curves[0].lineno,curves[0].end_lineno])


def native_material(label,index):
    value=json.loads((SAVED/(label+'-locals.json')).read_text())
    return next(x['material'] for x in value['contributions'] if x['kind']=='native' and x['index']==index)


def saved_input_defects(direction):
    """Report paired saved-input arithmetic departure, never fit/correct a slope."""
    def flat(values):
        return [x for v in values for x in (flat(v) if isinstance(v,list) else [v])]
    saved=[]
    for label in ('base','plus-1e-06','minus-1e-06'):
        value=json.loads((SAVED/(label+'-locals.json')).read_text())
        row=next(x for x in value['contributions'] if x['kind']=='native' and x['index']==0)
        words=row['paired_input']
        with localcontext() as ctx:
            ctx.prec=80
            saved.append((row['slots'][-18:],[pair([h,l]) for h,l in zip(flat(words['resultants']),flat(words['resultants_low']))]))
    if not saved[0][0]==saved[1][0]==saved[2][0] or any(len(x[1])!=18 for x in saved):
        raise ValueError('complete unchanged retained input map required')
    rows=[]
    with localcontext() as ctx:
        ctx.prec=80
        for slot,b,p,m in zip(saved[0][0],saved[0][1],saved[1][1],saved[2][1]):
            ideal=D.from_float(direction[slot])
            rows.append(dict(slot=slot,central_input_slope_defect=str((p-m)/(2*TAU)-ideal),
                             midpoint_defect=str((p+m)/2-b)))
    return rows


def affine_row(base,plus,minus,origin,law):
    """Exact ideal affine-domain equation; saved secant retains precision limits."""
    if not base['fibre_id']==plus['fibre_id']==minus['fibre_id']:
        raise ValueError('same canonical fibre identity required')
    if not base['branch']==plus['branch']==minus['branch'] or base['branch'] in ('YIELD_BOUNDARY','HARDENING_KNOT'):
        raise ValueError('saved secant must stay on the same actual smooth fibre branch')
    e0=pair(base['total_strain']);ep=pair(plus['total_strain']);em=pair(minus['total_strain'])
    z0=pair(origin[:2]);p0=pair(origin[2:]);young=D.from_float(float(law['young']))
    radius=D.from_float(float(law['yield_stress']))+D.from_float(float(law['hardening']))*p0
    low=z0-radius/young;high=z0+radius/young
    if base['branch']=='PLASTIC_POSITIVE':
        distance=e0-high
    elif base['branch']=='PLASTIC_NEGATIVE':
        distance=low-e0
    elif base['branch']=='ELASTIC':
        distance=min(e0-low,high-e0)
    else:
        raise ValueError('unsupported canonical fibre branch')
    if distance<=0:
        raise ValueError('no strictly interior base fibre branch')
    sensitivity=(ep-em)/(2*TAU)
    amount=None if sensitivity==0 else distance/abs(sensitivity)
    return dict(fibre=base['fibre_id'],branch=base['branch'],origin_plastic=str(z0),
        origin_accumulated=str(p0),strain=str(e0),yield_radius=str(radius),
        negative_boundary=str(low),positive_boundary=str(high),strain_margin=str(distance),
        strain_sensitivity=str(sensitivity),two_sided_parameter_radius=None if amount is None else str(amount),
        midpoint_absolute_residual=str(abs((ep+em)/2-e0)),
        forward_backward_sensitivity_disagreement=str(abs((ep-e0)/TAU-(e0-em)/TAU)),
        algorithmic_modulus=base['algorithmic_modulus'],
        saved_modulus_change=max(str(abs(pair(plus['algorithmic_modulus'])-pair(base['algorithmic_modulus']))),
                                 str(abs(pair(minus['algorithmic_modulus'])-pair(base['algorithmic_modulus']))),key=D))


def candidate_scale(radius,max_h):
    """Half-domain interior option fixed from margins, never derivative errors."""
    if radius<=0 or max_h<=0:
        raise ValueError('strict positive domain and inherited maximum h required')
    exact=min(D(1),radius/(2*max_h))
    selected=math.nextafter(float(exact),0.)
    if not 0.<selected<=1. or D.from_float(selected)*max_h>=radius:
        raise ValueError('no representable strict interior whole-direction option')
    return selected


def prepare(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report=dict(status='PREPARATION_INCOMPLETE',numerical_evaluations=0,tests_run=False,
                authority=False,scale_adopted=False,independent_review='Mistral pending',
                qualification=False,source_direction='original frozen RNG818 vector',
                h=[float(x) for x in STEPS])
    try:
        law=captured_law()
        original=json.loads((SAVED/'assessment.json').read_text())
        if original['status']!='UNRESOLVED_BRANCH' or original['steps']!=[float(x) for x in STEPS]:
            raise ValueError('original frozen failed assessment identity required')
        base=native_material('base',0);plus=native_material('plus-1e-06',0);minus=native_material('minus-1e-06',0)
        if not base['origin']==plus['origin']==minus['origin'] or not base['cell_identity']==plus['cell_identity']==minus['cell_identity']:
            raise ValueError('same fixed-origin cell required')
        if not len(base['stations'])==len(plus['stations'])==len(minus['stations'])==len(base['origin']['stations'])==8:
            raise ValueError('complete same-parent eight-station inventory required')
        rows=[]
        report['rows']=rows
        with localcontext() as ctx:
            ctx.prec=80
            for station,(b,p,m,old) in enumerate(zip(base['stations'],plus['stations'],minus['stations'],base['origin']['stations'])):
                if not len(b['fibres'])==len(p['fibres'])==len(m['fibres'])==len(old['rows']):
                    raise ValueError('complete canonical fibre station inventory required')
                for fields in zip(b['fibres'],p['fibres'],m['fibres'],old['rows']):
                    row=affine_row(*fields,law);row['station']=station;rows.append(row)
            radii=[D(row['two_sided_parameter_radius']) for row in rows if row['two_sided_parameter_radius'] is not None]
            if len(rows)!=32 or not radii:
                raise ValueError('complete 32-fibre domain required')
            radius=min(radii);scale=candidate_scale(radius,STEPS[0])
            limiting=[dict(station=r['station'],fibre=r['fibre'],radius=r['two_sided_parameter_radius']) for r in rows
                      if r['two_sided_parameter_radius'] is not None and D(r['two_sided_parameter_radius'])==radius]
        vector_path=SAVED/'direction-and-row-map.npz';direction=read_vector(vector_path,'direction')
        scaled=[scale*x for x in direction]
        if any(a!=0. and b==0. for a,b in zip(direction,scaled)):
            raise ValueError('whole-direction scaling dropped a nonzero component')
        action_groups={name:dict(original_raw_norm=m['raw_action_norm'],original_work_norm=m['work_action_norm'],
            ideal_scaled_raw_norm=scale*m['raw_action_norm'],ideal_scaled_work_norm=scale*m['work_action_norm'])
            for name,m in original['evaluations'][0]['groups'].items()}
        if not all(v['ideal_scaled_raw_norm']>0. and v['ideal_scaled_work_norm']>0. for v in action_groups.values()):
            raise ValueError('unexercised cached group action')
        curvature=[dict(index=x['index'],original_omitted_action_norm=x['omitted_action_norm'],
                        ideal_scaled_omitted_action_norm=scale*x['omitted_action_norm']) for x in original['point_controls']]
        if not curvature or not all(x['ideal_scaled_omitted_action_norm']>0. for x in curvature):
            raise ValueError('unexercised cached multiplier curvature')
        bindings={p.relative_to(ROOT).as_posix():sha(p) for p in (
            Path(__file__).resolve(),ROOT/'tests/test_r8_f06_paired_branch_domain_dev.py',vector_path,
            SAVED/'base-locals.json',SAVED/'plus-1e-06-locals.json',SAVED/'minus-1e-06-locals.json',SAVED/'assessment.json',
            ROOT/law['source'],SAVED/'source-snapshots'/Path(law['source']).name,
            ROOT/'src/anyfesolver/_ge_beam3_fibre_cell.py',ROOT/'src/anyfesolver/_ge_beam3_fibre_section.py',
            ROOT/'src/anyfesolver/_ge_beam3_retained_fibre.py',ROOT/'scripts/r8_f06_paired_active_tangent_dev.py',
            TANGENT/'ROOT_PRELAUNCH_BINDINGS.json',TANGENT/'HANDOFF_SELECTED.json',
            TANGENT.parent.parent/'package/snapshot-bindings.json',SAVED/'artifact-before.json',SAVED/'bindings.json',
            ROOT/'src/anyfesolver/_ge_beam3_generalized_ellipsoid_section.py',ROOT/'src/anyfesolver/plasticity.py',
            ROOT/'src/anyfesolver/_stiffened_shell_q4_algorithmic_pullback.py')}
        source_checks={}
        manifest_path=TANGENT.parent.parent/'package/snapshot-bindings.json'
        manifest=json.loads(manifest_path.read_text())
        original_bindings=json.loads((SAVED/'bindings.json').read_text())
        if original_bindings.get(manifest_path.relative_to(ROOT).as_posix())!=sha(manifest_path):
            raise ValueError('original installed package manifest binding changed')
        for path,digest in bindings.items():
            if path.startswith('src/'):
                snapshot=SAVED/'source-snapshots'/Path(path).name
                if manifest.get(path)!=digest:
                    raise ValueError('original installed package source mismatch: '+path)
                if snapshot.is_file() and sha(snapshot)!=digest:
                    raise ValueError('original installed command source snapshot mismatch: '+path)
                source_checks[path]=dict(package_manifest_sha256=sha(manifest_path),source_sha256=digest,
                    command_snapshot_present=snapshot.is_file())
        report.update(status='AFFINE_FIBRE_DOMAIN_CANDIDATE_ONLY',law=law,rows=rows,
            accepted_sha256=original['accepted_sha256'],rebased_sha256=original['rebased_sha256'],
            radius=str(radius),limiting_fibres=limiting,uniform_scale=scale,uniform_scale_hex=scale.hex(),
            scale_rule='nextafter(min(1,rho/(2*max(original h))),0); half-domain margin, no derivative-error fitting',
            effective_amounts=[float(h)*scale for h in STEPS],vector_sha256=sha(vector_path),
            component_count=len(direction),nonzero_before=sum(x!=0. for x in direction),nonzero_after=sum(x!=0. for x in scaled),
            saved_retained_input_defects=saved_input_defects(direction),source_snapshot_checks=source_checks,
            cached_group_action_predictions=action_groups,cached_curvature_action_predictions=curvature,
            smooth_fibre_secant='saved same-parent +/−1e-6 on unchanged linear FlowCurve branches; ideal inverse affine',
            numerical_precision_limits='stored paired strain/input words and canonical finite return residuals are not exact real sensitivities',
            unproved_materials=['nonlinear generalized ellipsoid local force map','full Q4 deformation/J2 active and hardening map'],
            mandatory_future_refusal='any actual fibre sign/active/knot, generalized branch/active, or J2 active/segment/knot transition at either side of ANY inherited h',
            bindings=bindings)
        (out/'candidate-direction.json').write_text(json.dumps(dict(status=report['status'],uniform_scale=scale,
            direction=scaled,source_npz_sha256=sha(vector_path),h=report['h'],authority=False,scale_adopted=False),indent=2)+'\n')
    except BaseException as exc:
        report['failure']=dict(type=type(exc).__name__,message=str(exc))
        raise
    finally:
        (out/'candidate.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true',required=True)
    parser.add_argument('--out',default=str(DOC))
    args=parser.parse_args()
    result=prepare(args.out)
    forbidden=[name for name in sys.modules if name.split('.')[0] in ('numpy','scipy','numba','anyfesolver','anymaterial','pytest')]
    if forbidden:
        raise AssertionError('preparation imported numerical modules: '+repr(forbidden))
    print(json.dumps({k:result[k] for k in ('status','radius','uniform_scale','limiting_fibres','component_count','nonzero_before','nonzero_after')}))
