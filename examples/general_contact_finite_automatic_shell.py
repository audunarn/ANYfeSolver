"""Declared automatic finite V2D shell cycle with strict fixed-selection restore."""
from __future__ import annotations
from dataclasses import asdict
import argparse,json
from pathlib import Path
from examples.general_contact_finite_v2d_pair import (
    build as paired_build,solve as paired_solve,report_result as paired_report,json_value,rigid_pose,
)
from anyfesolver.general_contact import (
    AutomaticContactDomain,AutomaticContactDefaults,AutomaticContactParticipant,
    AutomaticContactResolution,resolve_automatic_contact_domain,verify_automatic_contact_resolution,
)

SCHEMA='ANYSOLVER_FINITE_AUTOMATIC_SHELL_EXAMPLE_V1'


def declared_domain(definition):
    return AutomaticContactDomain('finite-automatic-two-physical-bodies',
        tuple(AutomaticContactParticipant(s,'upper-body' if s.element_ids==(1,2) else 'lower-body')
              for s in definition.surfaces),AutomaticContactDefaults(1000.,5e-5,'augmented'),maximum_pairs=1)


def build(*,saved_domain=None,saved_resolution=None):
    model,paired_definition,settings,program=paired_build()
    expected=declared_domain(paired_definition)
    if (saved_domain is None)!=(saved_resolution is None):
        raise ValueError('saved domain and fixed selection are required together')
    if saved_domain is None:
        domain=expected
        resolution=resolve_automatic_contact_domain(model,domain,settings,analysis_control='contact_path')
    else:
        domain=AutomaticContactDomain.from_dict(saved_domain)
        if domain.to_dict()!=expected.to_dict():raise ValueError('saved physical domain differs from fixture')
        resolution=AutomaticContactResolution.from_dict(saved_resolution)
        verify_automatic_contact_resolution(model,domain,resolution,settings)
        if json.loads(resolution.canonical_contract_json)['analysis_control']!='contact_path':
            raise ValueError('saved selection has incompatible analysis control')
    return model,domain,settings,resolution,program


def fixture(model,domain,settings,resolution,program):
    definition=resolution.definition;interaction=definition.interactions[0]
    surfaces={s.name:s for s in definition.surfaces}
    source=surfaces[interaction.surface];target=surfaces[interaction.target]
    pressure=sorted({nid for eid in source.element_ids for nid in model.mesh.elements[eid].node_ids})
    return json_value(dict(nodes=len(model.mesh.nodes),elements=len(model.mesh.elements),dofs=model.mesh.dof_manager.total_dofs,
        source_elements=list(source.element_ids),target_elements=list(target.element_ids),source_pressure_node_ids=pressure,
        pressure_basis='shared_nodal_P1',contact=asdict(definition),settings=asdict(settings),program=program.contract(),
        physical_bodies={'upper':dict(element_ids=[1,2],node_ids=[1,2,3,4],skin='bottom',force_nodes=[2,4]),
                         'lower':dict(element_ids=[3,4],node_ids=[5,6,7,8],skin='top',guided_global_uz_nodes=[6,7,8])}))


def require_report_identity(report,model,domain,settings,resolution,program):
    if type(report) is not dict or report.get('schema')!=SCHEMA:
        raise ValueError('exact automatic finite saved report required')
    if (report.get('automatic_domain')!=domain.to_dict()
            or json_value(report.get('automatic_resolution'))!=json_value(resolution.to_dict())
            or report.get('fixture')!=fixture(model,domain,settings,resolution,program)):
        raise ValueError('saved selection/domain/physical program identity differs')
    checkpoint=report.get('checkpoint')
    if type(checkpoint) is not dict:raise ValueError('saved report checkpoint missing')
    if report.get('displacements')!=checkpoint.get('displacements'):
        raise ValueError('saved terminal pose differs from checkpoint')
    return checkpoint


def solve(model,domain,settings,resolution,program,**kwargs):
    # Always guard the restored fixed selection. The structural owner checks the
    # complete model and analysis fingerprint, including the generated pair ID.
    definition=verify_automatic_contact_resolution(model,domain,resolution,settings)
    result=paired_solve(model,definition,settings,program,**kwargs)
    result.info['automatic_resolution_identity']=resolution.identity
    result.info['automatic_selection_scope']='fixed declared reference selection; no moving re-pair'
    return result


def report_result(result,model,domain,settings,resolution,program):
    report=paired_report(result,model,resolution.definition,settings,program)
    report.update(schema=SCHEMA,fixture=fixture(model,domain,settings,resolution,program),
                  automatic_domain=domain.to_dict(),automatic_resolution=resolution.to_dict())
    return json_value(report)


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--through-step',type=int,choices=range(1,17),default=4)
    parser.add_argument('--restart');parser.add_argument('--domain');parser.add_argument('--selection')
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv);output=Path(args.output)
    if output.exists():raise ValueError('output already exists')
    output.parent.mkdir(parents=True,exist_ok=True);result=None
    try:
        incoming=None;checkpoint=None;saved_domain=None;saved_resolution=None
        if bool(args.domain)!=bool(args.selection):raise ValueError('domain and selection files must be supplied together')
        if args.domain:
            saved_domain=json.loads(Path(args.domain).read_text())
            saved_resolution=json.loads(Path(args.selection).read_text())
        if args.restart:
            incoming=json.loads(Path(args.restart).read_text())
            if incoming.get('schema')==SCHEMA:
                if args.domain:raise ValueError('rich restart already supplies its fixed selection')
                saved_domain=incoming.get('automatic_domain');saved_resolution=incoming.get('automatic_resolution')
                if saved_domain is None or saved_resolution is None:raise ValueError('requested restart fixed selection missing')
                checkpoint=incoming.get('checkpoint')
            else:
                if saved_domain is None:raise ValueError('bare checkpoint requires explicit saved domain and selection')
                checkpoint=incoming
            if type(checkpoint) is not dict:raise ValueError('requested restart checkpoint missing')
        model,domain,settings,resolution,program=build(saved_domain=saved_domain,saved_resolution=saved_resolution)
        if incoming is not None and incoming.get('schema')==SCHEMA:
            require_report_identity(incoming,model,domain,settings,resolution,program)
        result=solve(model,domain,settings,resolution,program,max_load_factor=args.through_step/16.,restart_checkpoint=checkpoint)
        report=report_result(result,model,domain,settings,resolution,program)
    except Exception as exc:
        report=dict(schema=SCHEMA,status='NONPASS',failure_reason=f'{type(exc).__name__}: {exc}',scientific_acceptance=False,released=False)
        if result is not None:
            report['returned_accepted_suffix']=[s.to_dict(include_displacements=True) for s in result.snapshots]
            report['checkpoint']=json_value(result.restart_checkpoint);report['serialization_failure']=True
    temporary=output.with_suffix(output.suffix+'.tmp')
    temporary.write_text(json.dumps(report,default=json_value,allow_nan=False),encoding='utf-8');temporary.replace(output)
    print(json.dumps(dict(status=report['status'],failure_reason=report.get('failure_reason'),
        accepted_suffix_count=len(report.get('accepted_suffix',[])),output=str(output))))
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
