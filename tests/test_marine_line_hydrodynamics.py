from dataclasses import replace
import copy
import numpy as np
import pytest
from scipy import sparse
import importlib.util
if importlib.util.find_spec('anyloads') is None:
    pytest.importorskip('anyloads', reason='optional line-hydrodynamics provider is not installed')
from anyloads.marine import AiryEnvironment,AiryComponent,CurrentProfile
from anyloads.marine_line import LineHydrodynamics
from anyfesolver.marine_mooring import (MooringLineError,data_to_dict,data_from_dict,HYDRO_API_VERSION)
from anyfesolver.marine_mechanical import (MarineMechanicalModel,MarineMechanicalRunSpec,
    MarineMechanicalInputError,solve_marine_mechanical,model_to_dict,model_from_dict,result_to_dict)
from anyfesolver.marine_contributions import WorldWrenchContribution
from test_marine_mooring import coupled_case,fixed_case,perturb
from test_marine_mooring_driver import case as driver_case


def section():return LineHydrodynamics(.1,1.,.02,2.,.4,3.,.4)


def hydro_case():
    model,states,nodes=coupled_case()
    env=AiryEnvironment(20.,(AiryComponent(.2,1.3,30.,.2),),CurrentProfile((0.,20.),((.7,.4,0.),(.2,-.1,0.))))
    lines=tuple(replace(line,hydrodynamics=(section(),)*len(line.rest_lengths)) for line in model.lines)
    return replace(model,lines=lines,environment=env),states,nodes


@pytest.mark.parametrize('field,matrix',[('q','stiffness'),('v','damping'),('a','mass')])
def test_sparse_hydrodynamic_pullback_all_finite_body_node_columns(field,matrix):
    model,states,nodes=hydro_case();out=model.evaluate(states,nodes,time=.37);h=7e-6;cols=[]
    for j in range(model.size):
        plus=perturb(states,nodes,field,j,h);minus=perturb(states,nodes,field,j,-h)
        cols.append(-(model.force(*plus,time=.37)-model.force(*minus,time=.37))/(2*h))
    np.testing.assert_allclose(getattr(out,matrix).toarray(),np.asarray(cols).T,atol=8e-7,rtol=3e-7)
    assert sparse.isspmatrix_csr(getattr(out,matrix))
    np.testing.assert_allclose(sum(out.force_components.values()),out.force,atol=2e-14)
    assert out.relative_drag_power>=0. and out.added_mass_kinetic_energy>=0.


def test_closed_form_node_drag_current_geometry_and_anisotropic_reference_mass():
    model,nodes=fixed_case();model=replace(model,gravity_world=(0.,0.,0.))
    env=AiryEnvironment(20.,current=CurrentProfile.uniform(20.,(2.,3.,0.)),density=1000.)
    model=replace(model,lines=(replace(model.lines[0],hydrodynamics=(section(),section())),),environment=env)
    out=model.evaluate({},nodes)
    np.testing.assert_allclose(out.force,[4*np.pi,450.,0.],atol=1e-12)
    np.testing.assert_allclose(out.mass.toarray(),np.diag([1.92,3.2,3.2]),atol=1e-14)
    np.testing.assert_allclose(out.endpoint_forces.sum(axis=0),[8*np.pi,900.,0.],atol=1e-12)
    np.testing.assert_allclose(out.anchor_reactions,-out.endpoint_forces[[0,2]])


def test_hydrodynamic_schema_roundtrip_immutable_and_legacy_refusal():
    model,states,nodes=hydro_case();data=data_to_dict(model)
    assert data['api_version']==HYDRO_API_VERSION
    restored=data_from_dict(data,model.bodies);assert data_to_dict(restored)==data
    np.testing.assert_array_equal(restored.force(states,nodes,time=.5),model.force(states,nodes,time=.5))
    with pytest.raises(Exception):restored.lines[0].hydrodynamics[0].drag_normal=2.
    for mutate in (lambda d:d.update(api_version='marine_mooring_v1'),lambda d:d.pop('environment'),
                   lambda d:d['lines'][0].update(hydrodynamics=None)):
        bad=copy.deepcopy(data);mutate(bad)
        with pytest.raises(MooringLineError):data_from_dict(bad,model.bodies)
    mechanical=MarineMechanicalModel(model.bodies,mooring=model)
    assert model_to_dict(model_from_dict(model_to_dict(mechanical)))==model_to_dict(mechanical)


@pytest.mark.parametrize('failure',['environment','density','gravity','dry','time'])
def test_explicit_environment_and_domain_refusals(failure):
    model,states,nodes=hydro_case()
    with pytest.raises(MooringLineError):
        if failure=='environment':replace(model,environment=None)
        elif failure=='density':replace(model,fluid_density=1000.)
        elif failure=='gravity':replace(model,gravity_world=(0.,0.,-10.))
        elif failure=='dry':model.evaluate(states,replace(nodes,positions=[[0.,0.,1.]]))
        else:model.evaluate(states,nodes,time=True)


@pytest.mark.parametrize('cid',['mooring:elastic_force','mooring:any-future-component'])
def test_public_model_reserves_mooring_diagnostic_namespace(cid):
    model,states,nodes=driver_case()
    load=WorldWrenchContribution(cid,model.bodies[0],(0.,0.,0.),(1.,0.,0.))
    with pytest.raises(MarineMechanicalInputError,match='reserved'):replace(model,contributions=(load,))


def test_same_physical_line_identity_refuses_duplicate_morison_provider():
    from anyloads.marine import MorisonSection,HydrodynamicOwnership
    from anyfesolver.marine_morison import MemberEndpoint,MorisonMember,MorisonMemberSetContribution
    model,states,nodes=driver_case()
    s=section();env=AiryEnvironment(20.)
    lines=tuple(replace(line,hydrodynamics=(s,s)) for line in model.mooring.lines)
    assembler=replace(model.mooring,lines=lines,environment=env)
    member=MorisonMember('0',MemberEndpoint('A',(0.,-1.,0.)),MemberEndpoint('B',(0.,-1.,0.)),
                         MorisonSection(.1),HydrodynamicOwnership())
    provider=MorisonMemberSetContribution('members',model.bodies,(member,),env)
    with pytest.raises(MarineMechanicalInputError,match='duplicate physical'):
        replace(model,mooring=assembler,contributions=(provider,))


def test_common_driver_added_mass_once_against_independent_linear_newmark():
    model,states,nodes=driver_case(force=2.)
    s=replace(section(),drag_normal=0.,drag_tangential=0.,inertia_normal=0.,inertia_tangential=0.)
    lines=tuple(replace(line,displaced_area=[.001,.001],hydrodynamics=(s,s)) for line in model.mooring.lines)
    assembler=replace(model.mooring,lines=lines,environment=AiryEnvironment(20.))
    model=replace(model,mooring=assembler);dt=.001
    result=solve_marine_mechanical(model,states,MarineMechanicalRunSpec(dt,3),initial_nodes=nodes)
    mass=np.diag([8.615,8.615,2.41,2.41,2.41]);K=np.zeros((5,5));f=np.array([0.,2.,0.,0.,0.])
    for j in range(2,5):
        for b in (0,1):K[b,b]+=40.;K[j,j]+=40.;K[b,j]-=40.;K[j,b]-=40.
    q=np.array([-1.2,1.2,0.,0.,0.]);v=np.zeros(5);a=np.linalg.solve(mass,f-K@q)
    for frame in result.accepted:
        np.testing.assert_allclose(np.r_[frame.positions['A'][0]+1.,frame.positions['B'][0]-1.,frame.nodes.positions[:,0]],q,atol=3e-13,rtol=1e-12)
        np.testing.assert_allclose(np.r_[frame.world_accelerations['A'][0],frame.world_accelerations['B'][0],frame.nodes.accelerations[:,0]],a,atol=3e-11)
        np.testing.assert_allclose(sum(frame.mooring_diagnostics['node_force_components'].values()),0.,atol=2e-11)
        qp=q+dt*v+.25*dt**2*a;vp=v+.5*dt*a
        a=np.linalg.solve(mass+.25*dt**2*K,f-K@qp);q=qp+.25*dt**2*a;v=vp+.5*dt*a
    assert result_to_dict(result)['accepted'][0]['mooring_diagnostics']['added_mass_kinetic_energy']==0.


def test_common_driver_prescribed_wave_uses_actual_step_time_and_saved_diagnostics():
    assembler,states,nodes=hydro_case();model=MarineMechanicalModel(assembler.bodies,mooring=assembler)
    result=solve_marine_mechanical(model,states,MarineMechanicalRunSpec(.001,2),initial_nodes=nodes)
    forces=[]
    for frame in result.accepted:
        p=frame.nodes.positions[0];a_fluid=assembler.environment.evaluate(p,frame.time).acceleration
        # Independent node inertia sum over adjacent half reference volumes.
        body=assembler.bodies[0];ep=assembler.lines[0].endpoints[-1]
        other=frame.positions['body']+frame.rotations['body']@ep.point_body
        expected=np.zeros(3)
        for endpoint in (assembler.lines[0].endpoints[0].position,other):
            t=(np.asarray(endpoint)-p);t/=np.linalg.norm(t);Q=np.outer(t,t);P=np.eye(3)-Q
            expected+=1025.*.5*1.1*.001*(3.*P+.4*Q)@a_fluid
        actual=frame.mooring_diagnostics['node_force_components']['fluid_inertia'][0]
        np.testing.assert_allclose(actual,expected,atol=2e-13,rtol=1e-12);forces.append(actual)
    assert not np.array_equal(forces[0],forces[-1])
