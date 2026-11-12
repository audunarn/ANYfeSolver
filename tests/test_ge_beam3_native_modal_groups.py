"""Complete reference groups; historical fixed-index failure remains intact."""
import numpy as np
import pytest
from scipy.linalg import subspace_angles
from docs.reference_cases.ge_beam3_reference_modal_groups import reference_groups,principal_correlations,POLICY
from docs.reference_cases.ge_beam3_curved_p5_modal_reference import parabolic_modal_reference
from docs.reference_cases import ge_beam3_continuum_frequency_shooting as reference
from anyfesolver import _ge_beam3_native_generalized_modal as modal
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_native_continuum_frequencies import CASES,REFINEMENTS,data,stations
from test_ge_beam3_native_generalized_modal import make
from test_ge_beam3_schur_line_program import save

@pytest.mark.parametrize('count',(True,5,11,7))
def test_reference_mode_count_rejection(count,tmp_path):
    c,j=data(False)
    with pytest.raises(ValueError):
        reference.solve(0.,c,j,np.arange(1.,7.),mode_count=count)
    save(tmp_path/'rejection.json',dict(count=count,rejected=True))

def test_reference_only_group_closes_cutoff(tmp_path):
    omega=np.array([1.,2.,3.,4.,5.,6.,6.1,6.2,9.,12.])
    groups,bands=reference_groups(omega)
    assert groups==((0,),(1,),(2,),(3,),(4,),(5,6,7))
    assert bands[7,1]<bands[8,0]
    save(tmp_path/'groups.json',dict(groups=groups,bands=bands))

@pytest.mark.parametrize('kind',('truncated','shape','order','nonfinite'))
def test_group_input_rejection(kind,tmp_path):
    values=np.arange(1.,11.)
    if kind=='truncated':values=np.array([1.,2.,3.,4.,5.,6.,6.1,6.2,6.3,6.4])
    if kind=='shape':values=values[:-1]
    if kind=='order':values[6]=values[5]
    if kind=='nonfinite':values[0]=np.nan
    with pytest.raises(ValueError):reference_groups(values)
    save(tmp_path/'rejection.json',dict(kind=kind,rejected=True))

@pytest.mark.parametrize('kind',('basis-change','lost-mode','singleton','duplicate'))
def test_principal_correlations(kind,tmp_path):
    x=np.eye(5)[:,:2];y=x.copy()
    if kind=='basis-change':
        x=x@np.array([[2.,.3],[-.1,.7]]);y=y@np.array([[0.,-3.],[.4,.2]])
    if kind=='lost-mode':y[:,1]=np.eye(5)[:,2]
    if kind=='singleton':x=x[:,:1];y=(.98*np.eye(5)[:,0]+.2*np.eye(5)[:,2])[:,None]
    if kind=='duplicate':
        y[:,1]=y[:,0]
        with pytest.raises(np.linalg.LinAlgError):principal_correlations(x.T@x,x.T@y,y.T@y)
        save(tmp_path/'correlation.json',dict(kind=kind,rejected=True));return
    squared=principal_correlations(x.T@x,x.T@y,y.T@y)
    independently=np.sort(np.cos(subspace_angles(x,y))**2)[::-1]
    assert np.linalg.norm(squared-independently)<=1e-11
    if kind=='basis-change':assert np.min(squared)>=1-1e-11
    if kind=='lost-mode':assert np.min(squared)<1e-11
    if kind=='singleton':
        assert abs(squared[0]-float((x[:,0]@y[:,0])**2/((x[:,0]@x[:,0])*(y[:,0]@y[:,0]))))<=1e-11
    save(tmp_path/'correlation.json',dict(kind=kind,squared=squared,independent=independently))

def weighted_fields(model,packet,modes,continuum,height,inertia,macros):
    native=[];exact=[];lookup={float(t):i for i,t in enumerate(continuum.sites)}
    layout=dict(packet.internal_layout)
    for eid,cell,weight,t,measure in stations(macros):
        e=model.mesh.elements[eid];coords=e.operator.reference.coordinates
        lift=.5*weight*(weight-1)*(coords[0]-2*coords[1]+coords[2])
        left=model.mesh.dof_manager.get_node_dofs(e.node_ids[cell])
        right=model.mesh.dof_manager.get_node_dofs(e.node_ids[cell+1])
        spin=modes.full_modes[list(layout[eid][3*cell:3*cell+3])]
        velocity=(1-weight)*modes.full_modes[list(left[:3])]+weight*modes.full_modes[list(right[:3])]
        velocity+=np.cross(spin.T,lift).T
        jac,_,rotation=reference.geometry(t,height)
        factor=np.sqrt(measure*jac)*np.linalg.cholesky(inertia).T@rotation.T
        native.append(factor@np.vstack((velocity,spin)))
        exact.append(factor@continuum.fields[:,lookup[t],:6].T)
    return np.vstack(native),np.vstack(exact)

@pytest.mark.parametrize('case',CASES)
def test_complete_native_modal_groups(case,tmp_path):
    height=.2 if case=='curved-coupled' else 0.;coupled=case!='straight-diagonal'
    c,j=data(coupled)
    low_ritz=parabolic_modal_reference(height,c,j,terms=20,order=96)
    high_ritz=parabolic_modal_reference(height,c,j,terms=24,order=96)
    other_ritz=parabolic_modal_reference(height,c,j,terms=24,order=64)
    window=np.sqrt(high_ritz.squared_frequencies[:10])
    ritz_error=float(np.max(abs(np.sqrt(low_ritz.squared_frequencies[:10])/window-1)))
    quadrature_error=float(np.max(abs(np.sqrt(other_ritz.squared_frequencies[:10])/window-1)))
    assert ritz_error<1e-7 and quadrature_error<1e-11
    groups,bands=reference_groups(window);count=groups[-1][-1]+1
    assert 6<=count<10 and bands[count-1,1]<bands[count,0]
    sites=np.unique(np.r_[[-1.,1.],[r[3] for n in REFINEMENTS for r in stations(n)]])
    low=reference.solve(height,c,j,window[:count],profile='ODE11',sites=sites,mode_count=count)
    high=reference.solve(height,c,j,window[:count],profile='ODE13',sites=sites,mode_count=count)
    omega=np.sqrt(high.eigenvalues)
    profile_error=float(np.max(abs(np.sqrt(low.eigenvalues)/omega-1)))
    method_error=float(np.max(abs(window[:count]/omega-1)))
    assert profile_error<1e-8 and method_error<1e-7
    save(tmp_path/'reference-low.json',low);save(tmp_path/'reference-high.json',high)
    save(tmp_path/'reference-groups.json',dict(policy=POLICY,case=case,window=window,bands=bands,
        groups=groups,closed_count=count,guard_index=count,ritz_error=ritz_error,
        quadrature_error=quadrature_error,profile_error=profile_error,method_error=method_error))
    rows=[]
    for macros in REFINEMENTS:
        print(dict(stage='native-group',case=case,macros=macros,count=count),flush=True)
        model,states,masses=make(bool(height),macros,clamped=True,coupled=coupled)
        n=model.mesh.dof_manager.total_dofs;before=canonical(states)
        assert all(np.array_equal(e.section.elastic,c) for e in model.mesh.elements.values())
        assert all(np.array_equal(m,j) for m in masses.values())
        packet,modes=modal.solve_modes(model,states,np.zeros(n),masses,np.zeros(n),num_modes=count)
        assert np.min(modes.eigenvalues)>0 and canonical(states)==before
        x,y=weighted_fields(model,packet,modes,high,height,j,macros)
        xx,xy,yy=x.T@x,x.T@y,y.T@y
        assert np.linalg.norm(xx-np.eye(count))<=1e-11
        assert np.linalg.norm(yy-np.eye(count))<=1e-8
        comparisons=[]
        for group in groups:
            ix=np.ix_(group,group)
            squared=principal_correlations(xx[ix],xy[ix],yy[ix])
            independent=np.sort(np.cos(subspace_angles(x[:,group],y[:,group]))**2)[::-1]
            assert np.linalg.norm(squared-independent)<=1e-11
            comparisons.append(dict(indices=group,squared=squared,independent=independent,
                minimum=float(np.min(squared))))
        errors=abs(np.sqrt(modes.eigenvalues)/omega-1)
        fixed=np.diag(xy)**2/(np.diag(xx)*np.diag(yy))
        row=dict(macros=macros,max_frequency_error=float(np.max(errors)),
            minimum_group_correlation=min(g['minimum'] for g in comparisons))
        rows.append(row)
        save(tmp_path/('native-n%d.json'%macros),dict(row=row,frequency_errors=errors,
            groups=comparisons,fixed_index_correlations_diagnostic_only=fixed,packet=packet,modes=modes))
        print(dict(stage='group-comparison',case=case,**row),flush=True)
    assert all(b['max_frequency_error']<a['max_frequency_error'] for a,b in zip(rows,rows[1:]))
    assert rows[-1]['max_frequency_error']<.02 and rows[-1]['minimum_group_correlation']>=.95
    save(tmp_path/'assessment.json',dict(case=case,policy=POLICY,groups=groups,count=count,rows=rows,
        old_gate_reclassified=False,production_qualified=False,independent_review='PENDING'))
