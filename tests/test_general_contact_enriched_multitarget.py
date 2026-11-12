"""Finite V16 operator/layout/atomicity discriminators; no legacy fixture edits."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
from anyfesolver.control import SolveCancelled
from anyfesolver.e4_pl_enriched import EnrichedProfileError
from anyfesolver.general_contact.enriched_overlap import EnrichedGeometryError
from anyfesolver.general_contact.enriched_multitarget_topology import disk_topology,certify_disk
from anyfesolver.general_contact.enriched_multitarget import MultiTargetSystem
from anyfesolver.general_contact.enriched_path import EnrichedPathOwner
from examples.general_contact_enriched_d1_path import build


@pytest.fixture(scope='module')
def layouts():
    return {mode:build(mode) for mode in ('force','displacement')}


def test_explicit_multiedge_disk_and_nonmanifold_refusals():
    faces=((1,2,5),(2,3,5),(3,4,5),(4,1,5))
    disk=disk_topology(faces)
    assert len(disk['internal_edges'])==4 and len(disk['boundary'])==4
    assert all(len(set(edges)&set(disk['internal_edges']))==2 for edges in disk['face_edges'])
    ear=disk_topology(faces+((2,1,6),))
    assert len(ear['boundary'])==5 and len(ear['internal_edges'])==5
    for bad in (faces+(faces[0],),faces+((1,5,6),),((1,2,3),(4,5,6)),((1,2,3),(1,2,4))):
        with pytest.raises(EnrichedGeometryError):disk_topology(bad)


def test_new_selected_operator_preserves_ordinary_full37_value_gradient_hessian(record_property):
    from anyfesolver.general_contact.enriched_overlap import QuadraturePolicy
    from anyfesolver.general_contact.enriched_moment_batch import batch_moments
    from anyfesolver.general_contact.enriched_multitarget_moments import selected_batch_moments
    ref=np.array(((-.5,-.5,0.),(.5,-.5,0.),(.5,.5,0.),(-.5,.5,0.)))
    source=ref.copy();source[:,2]=(.001,-.002,.0015,-.0005)
    mid=source.copy();target=np.array(((-3.,-3.,-.2),(0.,4.,-.21),(4.,-3.,-.19)))
    beta=np.array((1e-4,-2e-5,3e-5,1e-5));direction=np.array((0.,0.,1.))
    policy=QuadraturePolicy(1e-10,1e-8,1e-5,14144)
    args=(ref,source,target,beta,direction,policy);kw=dict(current_midsurface=mid)
    old=batch_moments(*args,**kw)
    new=selected_batch_moments(*args,**kw,internal_edges=(),target_node_ids=(5,6,7))
    evidence={}
    for names,tol in ((('support','weighted_gap','pressure_gram'),1e-10),
                      (('support_gradient','gap_integral_gradient'),1e-8),
                      (('support_hessian','gap_integral_hessian'),1e-5)):
        for name in names:
            error=float(np.max(np.abs(getattr(old,name)-getattr(new,name))))
            assert error<=tol;evidence[name]=error
    assert tuple(old.branch)==tuple(new.branch[:-1]) and new.branch[-1]==0
    evidence.update(old_triangles=old.triangles,new_triangles=new.triangles,
        old_quadrature=old.quadrature_difference,new_quadrature=new.quadrature_difference)
    assert np.all(np.asarray(new.quadrature_difference)<=np.array((1e-10,1e-8,1e-5)))
    with pytest.raises(SolveCancelled):
        selected_batch_moments(*args,**kw,internal_edges=(),target_node_ids=(5,6,7),cancel=lambda:True)
    record_property('protected_ordinary_oracle',json.dumps(evidence))


def test_registered_layout_affine_lift_all8_and_legacy_unchanged(layouts,record_property,tmp_path):
    from anyfesolver.general_contact.enriched_nonlinear import NonlinearEnrichedSystem
    measured={}
    for mode,(model,definition,owner,descriptor) in layouts.items():
        template=owner.template;c=template._control_matrix;t=template.transform
        assert template.physical_dofs==300 and len(template.cells)==16 and len(template.pairs)==512
        assert t.shape[0]==364 and len(template.pressure_ids)==25
        assert len(c)==(2 if mode=='force' else 3)
        assert np.linalg.matrix_rank(template._control_reduced)==len(c)
        assert np.max(np.abs(c @ t[:300]))<=1e-12
        assert t.shape[1]==template._base_transform.shape[1]-len(c)
        assert np.array_equal(t[300:,-64:],np.eye(64))
        for step in range(9):
            made=template.at(owner.program.target_at(step))
            error=np.max(np.abs(c @ made.offset[:300]-owner.program.target_at(step))/owner.control_scales)
            assert error<=1e-6 and made._accepted is None
            assert np.array_equal(made._base_transform,template._base_transform)
        assert owner.accepted_step==0
        with pytest.raises(EnrichedProfileError):owner.checkpoint()
        measured[mode]=dict(free=t.shape[1],base_free=template._base_transform.shape[1],rows=len(c),descriptor=descriptor)
    assert '_pair_maps_and_certificates' in NonlinearEnrichedSystem.__dict__
    artifact=Path(os.environ.get('ENRICHED_REFERENCE_OUTPUT',str(tmp_path)))/'ACTUAL_LAYOUT.json'
    artifact.write_text(json.dumps(measured,sort_keys=True)+'\n')
    record_property('actual_full_case_layout',json.dumps(dict(path=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        layouts={mode:{k:v for k,v in row.items()if k!='descriptor'}for mode,row in measured.items()}),sort_keys=True))


def test_actual_disk_physical_work_abi_and_later_mutation_refusal(layouts):
    model,_,owner,_=layouts['force'];system=owner.template
    maps,certificates=system._pair_maps_and_certificates(system.offset)
    assert len(certificates)==512 and len(maps)==512
    for source in {p.definition.source_element for p in system.pairs}:
        indices=[i for i,p in enumerate(system.pairs) if p.definition.source_element==source]
        actual=certify_disk([system.pairs[i] for i in indices],[maps[i] for i in indices])
        assert len(actual)==32 and len({edge for c in actual for edge in c['internal_edges']})>1
        changed=[system.pairs[i] for i in indices]
        broken=list(changed[1].indices);broken[0]=broken[0]+1
        changed[1]=replace(changed[1],indices=tuple(broken))
        with pytest.raises(EnrichedGeometryError):
            certify_disk(changed,[maps[i] for i in indices])
    node=model.mesh.nodes[1];old=node.z
    try:
        node.z=old+1e-6
        with pytest.raises(EnrichedProfileError):owner.verify()
    finally:node.z=old
    owner.verify();assert owner._state is None


def test_definition_program_cancellation_and_prefix_refusals_are_atomic(layouts):
    model,definition,owner,_=layouts['force'];baseline=owner._publication
    with pytest.raises(SolveCancelled):owner.advance(cancel=lambda:True)
    assert owner._publication is baseline and owner._busy is False
    with pytest.raises(EnrichedProfileError):owner.restore(b'{}',expected_sha256='0'*64)
    assert owner._publication is baseline and owner._busy is False
    interaction=definition.interactions[0]
    for change in (dict(normal_law='penalty'),dict(normal_stiffness=None),dict(gap_tolerance=40e-6),dict(surface=interaction.target,target=interaction.surface)):
        altered=replace(definition,interactions=(replace(interaction,**change),))
        if change.get('surface'):
            # Reversed participants must refuse actual Q4/S3 physical admission.
            with pytest.raises(EnrichedProfileError):
                EnrichedPathOwner(model,altered,owner.program,quadrature=owner.template.quadrature,
                    force_scale_n=.018)
        else:
            with pytest.raises(EnrichedProfileError):
                EnrichedPathOwner(model,altered,owner.program,quadrature=owner.template.quadrature,
                    force_scale_n=.018)
    with pytest.raises(EnrichedProfileError):owner.identity='changed'
    assert owner._publication is baseline
