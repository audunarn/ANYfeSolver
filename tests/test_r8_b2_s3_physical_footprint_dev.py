"""Public S3/mixed extension; independent physical variation and affine oracle."""
from dataclasses import replace
import os
import numpy as np
import pytest
from anyfesolver import stiffened_shell as api
from anyfesolver.control import CancellationToken
from test_r8_b2_physical_footprint_static_foundation_dev import (
    definition as q4_definition, case_definition as q4_case, independent_rows,
    oracle, verify, load, SELECT, save)

pytestmark=pytest.mark.skipif('B2_FOUNDATION_SUPERVISED' not in os.environ,
    reason='requires the fresh bounded S3 development harness')


def definition(topology='mixed',affine=False,reversed_normal=False):
    d=q4_case('affine-offset' if affine else 'coordinate')
    n=(0.,0.,-1.) if reversed_normal else (0.,0.,1.)
    left=tuple(api.S3V2DPanelShell('left-'+str(i),cell,1000.,.3,.15,n)
        for i,cell in enumerate(((1,2,5),(1,5,4))))
    right=(replace(d.shells[1],reference_normal=n),) if topology=='mixed' else tuple(
        api.S3V2DPanelShell('right-'+str(i),cell,1000.,.3,.15,n)
        for i,cell in enumerate(((2,3,6),(2,6,5))))
    g=replace(d.physical_footprints[0],patch_cells=((1,2,5),(1,5,4)),reference_director=n)
    return replace(d,shells=left+right,physical_footprints=(g,))


@pytest.mark.parametrize('topology',('mixed','s3'))
@pytest.mark.parametrize('affine',(False,True))
def test_public_affine_static_and_batch(topology,affine):
    d=definition(topology,affine);owner=api.create_analysis(definition=d)
    try:
        fs=(load(),load(True));refs=[oracle(owner,d,f) for f in fs]
        result=owner.solve_linear(fs[0],selected_footprints=SELECT,
            recover_members={'beam-right':(0.,.5,1.)})
        verify(topology+'-'+str(affine)+'-static',owner,d,fs[0],result,refs[0])
        assert len(result.footprint_wrenches)==1
        w=result.footprint_wrenches[0];shell_force=np.zeros(3);shell_moment=np.zeros(3)
        beam_force=np.zeros(3);beam_moment=np.zeros(3)
        for i,(action,(_,_,ss,bb,arm,n,e)) in enumerate(zip(w.node_actions,refs[0]['local'])):
            lam=np.asarray(result.constraint_multipliers)[5*i:5*i+5]
            force=lam[:3];moment=np.cross(n,e@lam[3:])
            expected=(-force,-moment,force,moment+np.cross(arm,force))
            actual=(action.shell_force_N,action.shell_direct_couple_Nm,
                action.beam_force_N,action.beam_direct_couple_Nm)
            for a,b in zip(actual,expected):assert np.max(abs(np.asarray(a)-b))<1e-10
            shell_force-=force;shell_moment-=moment+np.cross(arm,force)
            beam_force+=force;beam_moment+=moment+np.cross(arm,force)
        for actual,expected in ((w.shell_force_N,shell_force),(w.shell_moment_Nm,shell_moment),
            (w.beam_force_N,beam_force),(w.beam_moment_Nm,beam_moment)):
            assert np.max(abs(np.asarray(actual)-expected))<1e-10
        assert np.max(abs(shell_force+beam_force))<1e-10
        assert np.max(abs(shell_moment+beam_moment))<1e-10
        assert owner.capabilities.footprint_law_id==result.footprint_law_id
        batch=owner.solve_linear_many(tuple((str(i),f) for i,f in enumerate(fs)),
            selected_dofs=tuple(range(54)),selected_footprints=SELECT,
            recover_members={'beam-right':(0.,.5,1.)})
        assert batch.status=='completed',batch.failure_reason
        assert batch.factorization_count==1 and len(batch.cases)==2
        for i,(f,o,c) in enumerate(zip(fs,refs,batch.cases)):
            full=type('Full',(),{})();full.status='completed'
            full.displacements=c.selected_displacements;full.affine_reactions=c.selected_affine_reactions
            full.constraint_multipliers=c.constraint_multipliers;full.constraint_rows=batch.constraint_rows
            full.internal_strain_energy=c.internal_strain_energy
            verify(topology+'-'+str(affine)+'-batch-'+str(i),owner,d,f,full,o)
        unselected=owner.solve_linear(fs[1])
        assert unselected.status=='completed' and not unselected.footprint_wrenches
        save(topology+'-'+str(affine)+'-batch-summary',dict(status=batch.status,
            factorization_count=batch.factorization_count,solve_count=batch.solve_count,
            selected_member_envelopes=len(batch.member_resultant_envelopes)))
    finally:owner.close()


@pytest.mark.parametrize('reverse',(False,True))
def test_actual_native_director_first_variation(reverse):
    d=definition(reversed_normal=reverse);owner=api.create_analysis(definition=d)
    try:
        C,inventory,local=independent_rows(d)
        assert np.max(abs(owner._C.toarray()-C))<1e-12
        for shell,element,material_id,slots in owner._buckling_shells:
            if len(shell.node_ids)!=3:continue
            mesh=owner._buckling_model.mesh;geometry=element._geometry(mesh)
            material=owner._buckling_model.get_material(material_id)
            R=geometry['frame'];p=element.director_polarity
            native=element.native_reference_directors(mesh)
            assert np.max(abs(native-p*R[:,2]))<1e-14
            # Global nodal angular variations projected to native physical tilt.
            theta=np.array(((.2,-.3,.1),(-.1,.4,.2),(.3,.1,-.2)))
            vector=np.zeros(18)
            for i in range(3):vector[6*i+3:6*i+6]=theta[i]
            mapped=geometry['local_from_external']@vector
            order=geometry['internal_order'];tilt=np.cross(theta,native)@R[:,:2]
            gradients=geometry['shape_gradients'];t=tilt[np.array(order)]
            expected=np.array((gradients[:,0]@t[:,0],gradients[:,1]@t[:,1],
                gradients[:,1]@t[:,0]+gradients[:,0]@t[:,1]))
            ops=element._operators(geometry,element._constitutive(material))
            actual=element._effective_station_operators(ops)[1]@mapped
            assert np.max(abs(actual-expected))<1e-13
            # Native ordinary positive/negative polarity is tested without
            # changing its guarded element: the p factor is independently
            # checked against the p-adjusted operator definition.
            spin=np.zeros(18)
            for i in range(3):spin[6*i+3:6*i+6]=native[i]
            assert np.max(abs(ops['B_b']@(geometry['local_from_external']@spin)))<1e-13
            assert np.max(abs(ops['B_s']@(geometry['local_from_external']@spin)))<1e-13
        lam=np.linspace(.2,1.3,20);physical=np.zeros(54)
        for i,(_,_,s,b,arm,n,e) in enumerate(local):
            f=lam[5*i:5*i+3];m=np.cross(n,e@lam[5*i+3:5*i+5])
            physical[s:s+3]+=f;physical[b:b+3]-=f
            physical[s+3:s+6]+=m;physical[b+3:b+6]-=m+np.cross(arm,f)
        assert np.max(abs(C.T@lam-physical))<1e-12
        save('native-variation-'+str(reverse),dict(C_error=float(np.max(abs(owner._C.toarray()-C))),
            transpose_error=float(np.max(abs(C.T@lam-physical))),normal_reversed=reverse))
    finally:owner.close()


def test_patch_admission_owned_inputs_and_legacy_policy():
    d=definition();g=d.physical_footprints[0]
    cells=[[1,2,5],[1,5,4]];nodes=[1,2,5,4];n=np.array((0.,0.,1.));e=np.array(((1.,0.),(0.,1.),(0.,0.)))
    owned=api.B2PhysicalRigidFootprint('owned',nodes,10,n,e,cells)
    cells[0][0]=999;nodes[0]=999;n[2]=0;e[0,0]=0
    assert owned.shell_nodes==(1,2,5,4) and owned.patch_cells==((1,2,5),(1,5,4))
    assert owned.reference_director[2]==1 and owned.tangent_basis[0,0]==1
    with pytest.raises(ValueError):owned.reference_director.setflags(write=True)
    with pytest.raises(ValueError,match='positive B2 footprint node IDs'):
        api.B2NodalRigidFootprint('legacy',(1,2,5),10,((1,2,5),))
    bad=(replace(g,patch_cells=((1,2,3),)),replace(g,shell_nodes=(1,2,5)),
         replace(g,patch_cells=((1,2,5),(2,3,6)),shell_nodes=(1,2,3,5,6)))
    for item in bad:
        with pytest.raises(ValueError):api.create_analysis(definition=replace(d,physical_footprints=(item,)))
    for cells in (((1,2,5),(5,2,1)),((1,2,5,4,3),),((1,2,-5),)):
        with pytest.raises(ValueError):replace(g,patch_cells=cells)
    with pytest.raises(ValueError,match='overlapping'):
        api.create_analysis(definition=replace(d,physical_footprints=(g,replace(g,footprint_id='second'))))
    shells=list(d.shells);shells[-1]=replace(shells[-1],reference_normal=(0.,0.,-1.))
    with pytest.raises(ValueError,match='ambiguous director seam'):
        api.create_analysis(definition=replace(d,shells=tuple(shells)))
    positions=dict(d.shell_nodes_m);positions[4]=(0.,1.,.001)
    with pytest.raises(ValueError,match='coplanar'):
        api.create_analysis(definition=replace(d,shell_nodes_m=positions))


def test_spectral_refusal_cancellation_close(monkeypatch):
    d=definition();owner=api.create_analysis(definition=d)
    def forbidden(*a,**k):raise AssertionError('Q4 chart helper entered for S3 physical family')
    import anyfesolver._stiffened_shell_b2_physical_rigid_footprint as physical
    monkeypatch.setattr(physical,'q4_exp_director_chart_correction',forbidden)
    try:
        assert not owner.physical_footprint_buckling_capabilities.eligible
        with pytest.raises(ValueError,match='S3/mixed physical footprint buckling unsupported'):
            owner.buckling(api.B2PhysicalFootprintBucklingRequest(load(),()))
        with pytest.raises(ValueError,match='footprint second variation'):
            owner.buckling(api.B2ReferenceBucklingRequest(load(),()))
        with pytest.raises(ValueError,match='mass contract'):owner.modes(api.B2ReferenceModesRequest(1))
        token=CancellationToken();token.cancel()
        result=owner.solve_linear_many((('cancelled',load()),),selected_dofs=(0,),cancellation_token=token)
        assert result.status=='cancelled' and not result.cases
        assert owner.solve_linear(load()).status=='completed'
        during=CancellationToken()
        prefix=owner.solve_linear_many((('first',load()),('second',load(True))),
            selected_dofs=(0,),cancellation_token=during,on_case=lambda _:during.cancel())
        assert prefix.status=='cancelled' and len(prefix.cases)==1
        assert owner.solve_linear(load()).status=='completed'
    finally:owner.close()
    with pytest.raises(RuntimeError,match='closed'):owner.solve_linear(load())


def test_connected_s3_q4_cell_union_patch():
    d=definition(affine=False);g=replace(d.physical_footprints[0],
        shell_nodes=(1,2,3,5,6),patch_cells=((1,2,5),(2,3,6,5)))
    d=replace(d,physical_footprints=(g,));owner=api.create_analysis(definition=d)
    try:
        f=load();o=oracle(owner,d,f);r=owner.solve_linear(f,selected_footprints=SELECT)
        assert r.status=='completed',r.failure_reason
        assert owner._C.shape[0]==25 and len(r.constraint_rows)==25
        assert np.max(abs(owner._C.toarray()-o['C']))<1e-12
        for actual,expected in ((r.displacements,o['u']),(r.constraint_multipliers,o['lam']),
            (r.affine_reactions,o['reaction'])):
            assert np.max(abs(np.asarray(actual)-expected))/np.max(abs(expected))<1e-10
        assert max(abs(o['C']@np.asarray(r.displacements)))<1e-10
        assert len(r.footprint_wrenches[0].node_actions)==5
        save('mixed-cell-union',dict(status=r.status,rows=25,vertices=5,patch_cells=2))
    finally:owner.close()


@pytest.mark.parametrize('polarity',(-1,1))
def test_native_polarity_rotated_frame_reversed_winding(polarity):
    from anyfesolver.fe_core import FEModel
    from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
    model=FEModel('independent-native-director')
    for i,x in enumerate(((0.,0.,0.),(0.,0.,1.),(0.,1.,0.)),1):model.add_node(i,*x)
    model.add_material('mat',1000.,.3)
    element=NativeParityE4PLS3V2DShellElement(1,(1,2,3),'mat',thickness=.15,
        reference_normal=(1.,0.,0.),director_polarity=polarity)
    model.add_element(1,element);geometry=element._geometry(model.mesh);R=geometry['frame']
    assert geometry['internal_order']==(0,2,1)
    directors=element.native_reference_directors(model.mesh)
    assert np.max(abs(directors-polarity*np.array((1.,0.,0.))))<1e-14
    theta=np.array(((.2,-.3,.1),(-.1,.4,.2),(.3,.1,-.2)));v=np.zeros(18)
    for i in range(3):v[6*i+3:6*i+6]=theta[i]
    tilt=np.cross(theta,directors)@R[:,:2];t=tilt[np.array(geometry['internal_order'])]
    dx,dy=geometry['shape_gradients'].T
    expected=np.array((dx@t[:,0],dy@t[:,1],dy@t[:,0]+dx@t[:,1]))
    ops=element._operators(geometry,element._constitutive(model.get_material('mat')))
    actual=element._effective_station_operators(ops)[1]@(geometry['local_from_external']@v)
    assert np.max(abs(actual-expected))<1e-13
    save('native-polarity-'+str(polarity),dict(polarity=polarity,
        winding=geometry['internal_order'],curvature_error=float(np.max(abs(actual-expected)))))
