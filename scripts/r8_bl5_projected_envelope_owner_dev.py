"""Installed development owner for the declared21 whole-member ratio cases.

One scalar registration precedes each native run. Historical two-case runners
and evidence remain untouched. No formal F03 or connection qualification.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import traceback

from scripts import r8_bl5_projected_ratio_owner_dev as common
from scripts.r8_bl5_projected_envelope_norm_dev import (
    IDENTITY, UNIFORM_IDENTITY, LEGACY_SOURCE_SHA256, CRITERIA, LENGTH, WIDTH, ELL,
    case_from_id, helpers,
)

LOADS = common.LOADS
WHEEL_SHA = '1bf7dabd8771e76ed09825fb66a89e3e48d513dbb0321989c41701096a679632'
# Reviewed optional-reference extension; exact old evidence reuse is valid only
# against this source, not an arbitrary future change to norms or floor formulas.
LEGACY_COMPATIBLE_NORM_SHA256 = '122278bf2e02eb49f5e1b6f5cc96810ce90b94ebee58f6c1f60ba26adfe38468'


def registration_identity(case, path, source):
    record = json.loads(path.read_text())
    reference = record.get('reference_bound', 'original')
    current = common.sha(source/'scripts/r8_bl5_projected_envelope_norm_dev.py')
    # Reuse only the exact, retained pre-extension v1 source; not arbitrary old records.
    sources = {current}
    if reference == 'original' and current == LEGACY_COMPATIBLE_NORM_SHA256:
        sources.add(LEGACY_SOURCE_SHA256)
    identity = IDENTITY if reference == 'original' else UNIFORM_IDENTITY
    common.require(record['status']=='REGISTERED_ENVELOPE_DEVELOPMENT_BOUND' and
        record['identity']==identity and record['fixture_case']==case.fixture_case and
        record['fixture']==case.fixture and record['fixed_criteria']==CRITERIA and
        record['coordinate_norm']==dict(length_m=float(LENGTH), width_m=float(WIDTH), ell_m=float(ELL)) and
        record['counts']==case.counts and record['fixed_count']==case.fixed_count and
        record['supported_kkt_count']==case.supported_kkt_count and
        record['source_sha256'] in sources and
        record['helper_sha256']==helpers(case, reference=reference) and record['arrays_sha256']==common.sha(path.with_suffix('.npz')) and
        record['predicates'] and all(v is True for v in record['predicates'].values()) and
        math.isfinite(record['safe_case_floor']) and record['safe_case_floor'] > 0.,
        'unchanged independently eligible scalar envelope registration required')
    return record, dict(json_sha256=common.sha(path), arrays_sha256=record['arrays_sha256'],
        source_sha256=record['source_sha256'], helper_sha256=record['helper_sha256'],
        json_path=str(path.resolve()), arrays_path=str(path.with_suffix('.npz').resolve()))


def construct(case, coherent, api):
    base, old_policy = coherent.model(case.nx, case.ny, case.halfcells, 1, basis='p0')
    line = replace(base.lines[0], basis='discontinuous_p1_cell_average', multiplier_span=case.span)
    definition = replace(base, lines=(line,))
    policy = dict(old_policy, span_halfcells=case.span, c_lambda=4, c_lambda_actual=4,
        support_endpoints_m=[j/case.supports for j in range(case.supports+1)],
        support_membership=[list(range(j*case.span,(j+1)*case.span)) for j in range(case.supports)],
        basis=line.basis, qualification='EXPERIMENTAL_NOT_QUALIFIED',
        density_basis=('1','reference halfcell mean of 2*(x-center)/support_width'),
        whole_member_shell_ratio=str(case.member_ratio), halfcell_shell_ratio=str(2*case.member_ratio),
        continuous_law='BL5 director bond; explicitly released normal spin')
    return api.create_analysis(definition=definition), definition, policy


def run(case, site, wheel, source, registration, output):
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(status='FAILED', identity=IDENTITY, fixture_case=case.fixture_case,
        fixture=case.fixture, source_sha256=common.sha(__file__), cases={},
        formal_f03_authority_consumed=False, source_root=str(source.resolve()),
        classification='EXPERIMENTAL_NOT_QUALIFIED; installed development only',
        limitations='No uniform stability, physical accuracy, F03 or complete programme acceptance.')
    owner = None; original_error = False; cleanup_error = None
    try:
        site, source = site.resolve(strict=True), source.resolve(strict=True)
        common.require(not any(name.split('.')[0]=='anyfesolver' for name in sys.modules),
                       'fresh process without product preimports required')
        contract, binding = registration_identity(case, registration, source)
        modules = common.installed_identity(site, wheel, WHEEL_SHA, source)
        sys.path[:0] = [str(site), str(source/'scripts'), str(source/'tests')]
        import numpy as np
        import numba
        from threadpoolctl import threadpool_info
        import anyfesolver
        from anyfesolver import stiffened_shell as api
        import b3_e4_bl5_coherent_aggregation_dev as coherent
        import b3_e4_bl5_matched_mz_native_dev as matched
        import b3_e4_bl5_matched_p0_native_refinement_dev as inventory
        import r8_bl5_affine_flexible_native_dev as moments
        inventory.np = np  # Existing detached layout helper requires explicit np.
        common.require(coherent.LOADS==LOADS, 'physical loads unchanged')
        source_helpers = (coherent, matched, inventory, moments, common)
        common.require(all(Path(m.__file__).resolve()==(source/'scripts'/Path(m.__file__).name).resolve()
                           for m in source_helpers), 'source helper locations differ')
        settings = common.effective_settings(numba, threadpool_info)
        summary.update(registration=binding, safe_case_floor=contract['safe_case_floor'],
            fixed_criteria=dict(CRITERIA), wheel=str(wheel.resolve()), wheel_sha256=WHEEL_SHA,
            site=str(site), module_sha256=modules, matching_modules=len(modules),
            helper_sha256={Path(m.__file__).name:common.sha(m.__file__) for m in source_helpers},
            python=sys.version, imported_anyfesolver=anyfesolver.__file__, **settings)
        owner, definition, policy = construct(case, coherent, api)
        common.require(owner.capabilities['status']=='EXPERIMENTAL_NOT_QUALIFIED', 'qualification boundary changed')
        xyz, layout = inventory.layout(owner, definition, policy, case.nx)
        layout['halfcell_count'] = case.halfcells
        stations = inventory.station_inventory(owner)
        maps = matched.rigid_maps(owner, xyz)
        supports = []
        for j in range(case.supports):
            traces = [r for r in stations if int(r['beam_port']['member_index'])*2+
                      int(r['beam_port']['native_cell']) in range(j*case.span,(j+1)*case.span)]
            supports.append(dict(support_id=j, interval_m=[j/case.supports,(j+1)/case.supports],
                native_halfcells=sorted({int(r['beam_port']['member_index'])*2+
                    int(r['beam_port']['native_cell']) for r in traces}),
                multiplier_rows=sorted({int(row) for r in traces for row in r['multiplier_rows']})))
        native = output/'native'; native.mkdir()
        inputs = dict(fixture_case=case.fixture_case, fixture=case.fixture, layout=layout,
            xyz_m=xyz, supports=supports, station_traces=stations, policy=policy,
            loads=LOADS, beam_tip=101+case.halfcells,
            shell_definition=definition.shell, beams=[json.loads(b.raw) for b in definition.beams])
        # Reuse plain serializer but never feed dataclass objects implicitly.
        inputs['shell_definition'] = dict(node_ids=definition.shell.node_ids,
            coordinates_m=definition.shell.coordinates_m, q4_elements=definition.shell.q4_elements,
            thickness_m=definition.shell.thickness_m)
        common.write(native/'inputs.json', inputs)
        matched.save_npz(native/'operators.npz', inventory.sparse_arrays(owner, maps))
        summary['native_input_sha256']={name:common.sha(native/name) for name in ('inputs.json','operators.npz')}
        count_names=('shell_nodes','beam_nodes','physical_dofs','members','retained_cell_rotations',
            'retained_resultants','retained_dofs','full_q','multipliers')
        expected_counts={name:case.counts[name] for name in count_names}
        common.require(layout['counts']==expected_counts and owner._K.shape==(case.counts['full_q'],)*2 and
            owner._C.shape==(case.counts['multipliers'],case.counts['full_q']), 'actual ratio layout/counts differ')
        common.require(len(owner._fixed)==case.fixed_count and
            len(owner._free)+len(owner._fixed)==case.counts['full_q'] and
            not set(owner._free)&set(owner._fixed), 'complete support partition differs')
        common.require(all(s['native_halfcells']==list(range(j*case.span,(j+1)*case.span)) and
            len(s['multiplier_rows'])==10 for j,s in enumerate(supports)) and
            len({row for s in supports for row in s['multiplier_rows']})==10*case.supports,
            'actual complete projected support inventory differs')
        ns, nb, nm = case.counts['shell_nodes'], case.counts['beam_nodes'], case.counts['members']
        common.require(tuple(owner._beam_ids)==tuple(range(101,102+case.halfcells)) and
            np.allclose(xyz[:ns],[(i/case.nx,.2*j/case.ny,0.) for j in range(case.ny+1)
                for i in range(case.nx+1)],rtol=0.,atol=1e-13), 'typed physical geometry differs')
        expected_fixed={('shell',1+j*(case.nx+1),axis) for j in range(case.ny+1) for axis in range(6)}
        expected_fixed|={('beam',101,axis) for axis in range(5)}
        common.require({(f.domain,f.node,f.component) for f in definition.fixed}==expected_fixed,
                       'actual physical root supports differ')
        common.require(np.allclose(xyz[ns:],[(i/case.halfcells,0.,.0675) for i in range(nb)],rtol=0.,atol=1e-13),
                       'actual beam reference axis differs')
        common.require(definition.shell.thickness_m==.035 and definition.shell.young_pa==210e9 and
                       definition.shell.poisson==.3, 'actual shell physical section differs')
        elastic=np.diag((210e9*.01,210e9/2.6*.008,210e9/2.6*.008,
                        210e9/2.6*1.6666666666666667e-5,210e9*8.333333333333333e-6,
                        210e9*8.333333333333333e-6))
        common.require(all(np.array_equal(element.section.elastic,elastic) for element,_,_ in owner._beam_parts),
                       'actual native beam generalized section differs')
        common.require(all(np.allclose(r['beam_port']['frame'],np.eye(3),rtol=0.,atol=1e-13) and
            np.allclose(r['reference_director'],(0.,0.,1.),rtol=0.,atol=1e-13) and
            np.allclose(r['director_basis'],((1.,0.,0.),(0.,1.,0.)),rtol=0.,atol=1e-13) and
            np.allclose(r['reference_arm_m'],(0.,0.,-.05),rtol=0.,atol=1e-13) for r in stations),
            'actual trace physical frame/director/arm differs')
        for name, wrench in LOADS.items():
            folder=native/name; folder.mkdir()
            result=owner.solve_linear((api.NodalAction('beam',101+case.halfcells,wrench),))
            q=np.r_[result.shell_displacement,result.beam_coordinates]
            load=np.array(owner._accepted[1],copy=True)
            lam,reactions=np.array(result.multipliers,copy=True),np.array(result.reactions,copy=True)
            g,ct=owner._K@q,owner._C.T@lam
            raw=dict(q=q,load=load,multipliers=lam,reactions=reactions,Kq=g,
                constraint_resisting=ct,full_equilibrium=g+ct-load,weak_gaps=owner._C@q,
                shell_poses=q[:6*ns].reshape(ns,6),beam_nodal_poses=q[6*ns:6*(ns+nb)].reshape(nb,6),
                beam_retained=q[6*(ns+nb):].reshape(nm,24),shell_node_ids=owner._shell_ids,
                beam_node_ids=owner._beam_ids,xyz_m=xyz,coordinate_scale=owner._coordinate_scale,
                constraint_scale=owner._constraint_scale)
            matched.save_npz(folder/'raw-state.npz',raw)
            common.write(folder/'raw-state.json',dict(status=result.status,generation=result.generation,
                arrays_sha256=common.sha(folder/'raw-state.npz'),counts=layout['counts']))
            common.require(result.status=='completed' and all(np.isfinite(v).all() for v in
                (q,load,lam,reactions,g,ct)), 'finite complete installed solve required')
            transfer=coherent.physical_transfer(owner,result)
            exact=moments.exact_transfer_moments(transfer['station_records'])
            region=moments.exact_transfer_moments(transfer['station_records'],(.25,.75))[0]
            applied={d:m@load for d,m in maps.items()};exerted={d:m@(-ct) for d,m in maps.items()}
            ground={d:m@reactions for d,m in maps.items()};work=float(q@load)
            scale,work_scale=max(map(abs,wrench)),max(abs(work),1e-12)
            checks=dict(equilibrium_ratio=float(result.free_equilibrium_ratio),weak_gap=float(result.constraint_max),
                energy_work=abs(float(q@g)-work)/work_scale,compatible_work=abs(float(q@ct))/work_scale,
                ground_work=abs(float(q@reactions))/work_scale,
                domain_cuts=max(float(np.max(np.abs(applied[d]+exerted[d]+ground[d])))/scale for d in maps),
                global_balance=float(np.max(np.abs(sum(ground.values())+sum(applied.values()))))/scale,
                local_pullback=float(transfer['local_pullback_max_abs']),
                beam_physical_workmap_scaled=float(np.max(np.abs(exerted['beam']-np.asarray(exact[0]))))/scale,
                shell_physical_workmap_scaled=float(np.max(np.abs(exerted['shell']+np.asarray(exact[0]))))/scale)
            data=dict(status=result.status,physical_checks=checks,work_J=work,physical_transfer=transfer,
                exact_transfer_moments=exact,exact_region_wrench=region,
                ground_wrenches=ground,domain_wrenches=dict(applied=applied,connection_exerted=exerted,ground=ground))
            common.write(folder/'physical-actions.json',data);summary['cases'][name]=data
            for key,value in checks.items():
                limit=1e-9 if key=='weak_gap' else 1e-8
                common.require(np.isfinite(value) and value<=limit, f'{name} {key} exceeds unchanged physical budget: {value}')
        summary.update(counts=layout['counts'],factorization_count=owner._linear_factorizations,
            accurate_refinements=owner._linear_refinements,product_imports=common.canonical_imports(site),
            **common.effective_settings(numba,threadpool_info))
        common.require(owner._linear_factorizations==1 and owner._linear_refinements==6,
            'one eligible factorization and two corrections per load required')
        summary['predicates']=dict(actual_inventory=True,physical_work_and_balance=True,
            completed_finite=True,one_factorization=True,two_accurate_corrections=True)
        summary['status']='COMPLETED_INSTALLED_DEVELOPMENT_ONLY'
    except BaseException as error:
        original_error=True
        summary['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        raise
    finally:
        if owner is not None:
            try:owner.close()
            except BaseException:
                summary['status']='FAILED';summary['cleanup_error']=traceback.format_exc()
                cleanup_error=sys.exc_info()[1]
        common.write(output/'summary.json',summary)
        if cleanup_error is not None and not original_error:
            raise cleanup_error
    print(json.dumps(dict(status=summary['status'],fixture_case=case.fixture_case,counts=summary['counts'])))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('case','site','wheel','source','registration','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    run(case_from_id(args.case),Path(args.site),Path(args.wheel),Path(args.source),Path(args.registration),Path(args.output))
