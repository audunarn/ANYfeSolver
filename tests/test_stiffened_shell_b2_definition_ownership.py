"""Public definition ownership and actual owner workflow, not physics qualification."""
from dataclasses import FrozenInstanceError, replace
from array import array
import numpy as np
import pytest
from scipy import sparse
from anyfesolver import stiffened_shell as api
from test_stiffened_shell_b2_mixed_panel_experimental import _definition, _load
from test_stiffened_shell_b2_experimental import _members
from examples.stiffened_shell_b2_physical_rigid_footprint import definition as physical_definition


KINDS=(api.B2LinearDefinition,api.B2Q4PointDefinition,api.B2Q4PanelDefinition,
       api.B2PhysicalRigidFootprintDefinition,api.B2MixedPanelDefinition,api.B2MixedReferenceModalDefinition)


def mutable_inputs():
    base,_=_definition('q4')
    nodes={n:list(x) for n,x in base.shell_nodes_m.items()}
    ids=list(base.shells[0].node_ids);normal=[0.,0.,1.];offset=[0.,.08,-.03]
    shells=[replace(base.shells[0],node_ids=ids,reference_normal=normal),base.shells[1]]
    points=[replace(base.points[0],offset_local_m=offset),base.points[1]]
    return base,nodes,shells,points,ids,normal,offset


@pytest.mark.parametrize('kind',KINDS)
def test_all_six_public_classes_own_inputs_and_replace_keeps_identity(kind):
    base,nodes,shells,points,ids,normal,point_offset=mutable_inputs()
    T=base.transformation;u0=np.zeros(54);members=list(base.members);selected=np.array([49,50],dtype=np.int32)
    vertices=np.array(list(nodes.values())[:4]);local=[0.,.1,.2];footprints=[]
    if kind is api.B2LinearDefinition:d=kind(base.profile,members,T,u0,selected)
    elif kind is api.B2Q4PointDefinition:d=kind(base.profile,members[0],vertices,1000.,.3,.15,.4,T,u0,local,normal)
    elif kind is api.B2PhysicalRigidFootprintDefinition:d=kind(base.profile,nodes,shells,members,footprints,T,u0)
    else:d=kind(base.profile,nodes,shells,members,points,T,u0,footprints)
    expected=T.toarray();T.data[:]=99.;u0[:]=99.;selected[:]=0;vertices[:]=99.
    nodes[1][0]=99.;ids[:]=[99];normal[:]=[99];point_offset[:]=[99];local[:]=[99]
    shells.clear();points.clear();members.clear();footprints.append(object())
    np.testing.assert_array_equal(d.transformation.toarray(),expected)
    np.testing.assert_array_equal(d.prescribed_offset,np.zeros(54))
    assert not d.prescribed_offset.flags.writeable
    with pytest.raises(ValueError):d.prescribed_offset.setflags(write=True)
    detached=d.prescribed_offset;detached.shape=(9,6)
    assert d.prescribed_offset.shape==(54,)
    matrix=d.transformation;matrix.data=np.full_like(matrix.data,17);matrix.indices=np.zeros_like(matrix.indices);matrix.indptr=np.zeros_like(matrix.indptr);matrix._shape=(1,1)
    np.testing.assert_array_equal(d.transformation.toarray(),expected)
    copied=replace(d);assert type(copied) is kind
    np.testing.assert_array_equal(copied.transformation.toarray(),expected)
    member=d.member if hasattr(d,'member') else d.members[0]
    assert member is base.members[0] and member.section is base.members[0].section
    with pytest.raises(FrozenInstanceError):d.profile='changed'
    if hasattr(d,'shell_nodes_m'):
        assert d.shell_nodes_m[1]==(0.,0.,0.)
        with pytest.raises(TypeError):d.shell_nodes_m[1]=(9.,9.,9.)
        assert d.shells[0].node_ids==(1,2,5,4) and d.shells[0].reference_normal==(0.,0.,1.)
    if hasattr(d,'points'):assert d.points[0].offset_local_m==(0.,.08,-.03)
    if kind is api.B2LinearDefinition:
        assert d.selected_dofs.dtype==np.dtype(np.int32)
        np.testing.assert_array_equal(d.selected_dofs,[49,50])
    if kind is api.B2Q4PointDefinition:
        assert d.offset_local_m==(0.,.1,.2) and d.reference_normal==(0.,0.,1.)
        assert d.shell_vertices_m[0,0]==0.


def test_nested_s3_input_and_dataclass_replace_are_owned():
    base,_=_definition();shell=next(s for s in base.shells if type(s) is api.S3V2DPanelShell)
    ids=list(shell.node_ids);normal=list(shell.reference_normal)
    changed=replace(base,shells=[replace(shell,node_ids=ids,reference_normal=normal)])
    ids.clear();normal.clear()
    assert type(changed.shells[0]) is api.S3V2DPanelShell
    assert changed.shells[0].node_ids==shell.node_ids and changed.shells[0].reference_normal==shell.reference_normal


@pytest.mark.parametrize('kind',('q4','mixed','physical'))
def test_create_analysis_uses_owned_definition_after_original_and_detached_mutation(kind):
    base=physical_definition() if kind=='physical' else _definition(kind)[0]
    nodes={n:np.array(x) for n,x in base.shell_nodes_m.items()};T=base.transformation;offset=np.array(base.prescribed_offset)
    d=replace(base,shell_nodes_m=nodes,transformation=T,prescribed_offset=offset)
    owner=api.create_analysis(definition=d);fresh=None
    try:
        f=np.zeros(54);f[13]=1. if kind=='physical' else 0.;f+=_load() if kind!='physical' else 0.
        first=owner.solve_linear(f);assert first.status=='completed',first.failure_reason
        for x in nodes.values():x[:]=99.
        T.data[:]=99.;offset[:]=99.;d.transformation.data[:]=99.
        fresh=api.create_analysis(definition=replace(d))
        after=fresh.solve_linear(f);assert after.status=='completed',after.failure_reason
        assert first.displacements==after.displacements
        assert owner.solve_linear(f).displacements==first.displacements
    finally:
        owner.close()
        if fresh is not None:fresh.close()


@pytest.mark.parametrize('selected',(np.array([12.]),np.array([True]),np.array([[12]]),np.array([99])))
def test_original_integer_shape_and_range_selection_refusals(selected):
    members,_=_members();d=api.B2LinearDefinition(api.B2_E4_PL_FAST_LINEAR_V1,members,sparse.eye(18,format='csr')[:,6:],np.zeros(18),selected)
    np.testing.assert_array_equal(d.selected_dofs,selected)
    assert d.selected_dofs.dtype==selected.dtype and d.selected_dofs.shape==selected.shape
    with pytest.raises(ValueError):api.create_analysis(definition=d)


def test_nonfinite_offset_and_invalid_profile_still_refuse_at_owner_creation():
    base,_=_definition('q4');bad=np.zeros(54);bad[0]=np.nan
    d=replace(base,prescribed_offset=bad)
    with pytest.raises(ValueError):api.create_analysis(definition=d)
    with pytest.raises(ValueError):api.create_analysis(definition=replace(base,profile='not-admitted'))


def test_accepted_object_offsets_and_zero_dimensional_coordinate_material_inputs_are_owned():
    base,_=_definition('q4');offset=np.zeros(54,dtype=object)
    E=np.array(1000.);normal=[np.array(0.),np.array(0.),np.array(1.)]
    nodes={n:[np.array(v) for v in xyz] for n,xyz in base.shell_nodes_m.items()}
    d=replace(base,prescribed_offset=offset,shell_nodes_m=nodes,
              shells=[replace(s,E=E,reference_normal=normal) for s in base.shells])
    offset[:]=99.;E[...]=99.;normal[2][...]=99.;nodes[1][0][...]=99.
    detached=d.prescribed_offset;detached.setflags(write=True);detached[:]=99.
    assert d.prescribed_offset.dtype==object and np.all(d.prescribed_offset==0.)
    assert d.shells[0].E==1000. and d.shells[0].reference_normal==(0.,0.,1.) and d.shell_nodes_m[1]==(0.,0.,0.)
    owner=api.create_analysis(definition=d)
    try:
        result=owner.solve_linear(_load());assert result.status=='completed',result.failure_reason
    finally:owner.close()


@pytest.mark.parametrize('value',(np.array(8),np.int64(8),True))
def test_chunk_strict_type_refusal_is_not_changed_by_ownership(value):
    members,_=_members();d=api.B2LinearDefinition(api.B2_E4_PL_FAST_LINEAR_V1,members,sparse.eye(18,format='csr')[:,6:],np.zeros(18),np.arange(12,18),value)
    assert type(d.max_rhs_chunk) is type(value)
    with pytest.raises(ValueError,match='RHS chunk'):api.create_analysis(definition=d)


def test_density_zero_dimensional_array_remains_non_real_and_refused():
    base,_=_definition('q4');density=np.array(2.)
    d=replace(base,shells=[replace(s,density_kg_m3=density) for s in base.shells])
    density[...]=99.
    assert type(d.shells[0].density_kg_m3) is np.ndarray
    with pytest.raises(ValueError,match='density'):api.create_analysis(definition=d)


@pytest.mark.parametrize('view',(False,True))
def test_accepted_array_and_memoryview_coordinates_are_owned(view):
    base,_=_definition('q4');inputs={n:array('d',x) for n,x in base.shell_nodes_m.items()}
    d=replace(base,shell_nodes_m={n:memoryview(x) if view else x for n,x in inputs.items()})
    inputs[1][0]=99.
    assert d.shell_nodes_m[1]==(0.,0.,0.)
    with pytest.raises(TypeError):d.shell_nodes_m[1][0]=99.
    owner=api.create_analysis(definition=d)
    try:
        result=owner.solve_linear(_load());assert result.status=='completed',result.failure_reason
    finally:owner.close()


def test_equality_admitted_profile_array_keeps_representation_and_cannot_change_dispatch():
    base,_=_definition('q4');profile=np.array(base.profile)
    d=replace(base,profile=profile);profile[...]='changed'
    assert type(d.profile) is np.ndarray and d.profile.shape==() and d.profile.dtype==profile.dtype
    assert d.profile==base.profile
    with pytest.raises(ValueError):d.profile.setflags(write=True)
    owner=api.create_analysis(definition=d)
    try:
        result=owner.solve_linear(_load());assert result.status=='completed',result.failure_reason
    finally:owner.close()


@pytest.mark.parametrize('view',(False,True))
@pytest.mark.parametrize('kind',('point','panel'))
def test_accepted_declared_array_vectors_are_immutable_through_public_owner(view,kind):
    normal=array('d',[0.,0.,1.]);offset=array('d',[0.,.08,-.03])
    wrap=lambda value:memoryview(value) if view else value
    if kind=='point':
        from test_b2_f10_f11_point_attachment import VERTICES,specimen
        member,_,_,_=specimen();fixed=set(range(6))|set(range(18,30))
        fixed.update(6*i+j for i in (1,2) for j in (3,4,5));free=[i for i in range(36) if i not in fixed]
        d=api.B2Q4PointDefinition(api.B2_E4_PL_FAST_LINEAR_V1,member,VERTICES,1000.,.3,.15,.4,
            sparse.eye(36,format='csr')[:,free],np.zeros(36),wrap(offset),wrap(normal))
        f=np.zeros(36);f[13]=-.09;f[32]=.11
        declared_normal=d.reference_normal;declared_offset=d.offset_local_m
    else:
        base,_=_definition('q4')
        d=replace(base,shells=[replace(s,reference_normal=wrap(normal)) for s in base.shells],
            points=[replace(p,offset_local_m=wrap(offset)) for p in base.points])
        f=_load();declared_normal=d.shells[0].reference_normal;declared_offset=d.points[0].offset_local_m
    normal[2]=99.;offset[1]=99.
    assert declared_normal==(0.,0.,1.) and declared_offset==(0.,.08,-.03)
    with pytest.raises(TypeError):declared_normal[2]=99.
    with pytest.raises(TypeError):declared_offset[1]=99.
    owner=api.create_analysis(definition=d)
    try:
        result=owner.solve_linear(f);assert result.status=='completed',result.failure_reason
        assert owner.solve_linear(f).displacements==result.displacements
    finally:owner.close()
