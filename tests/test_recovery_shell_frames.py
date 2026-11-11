from types import MethodType
import numpy as np
import pytest
from anyfesolver import FEModel,Material,ShellElement,RecoveryConfig,recover_stress_result
from anyfesolver.recovery_shell_frames import native_recovery_frames


def model8(warp=False):
    model=FEModel('frame batching')
    model.register_material(Material('steel',210e9,0.3,7850))
    coords=[(0,0),(2,0),(2,1),(0,1),(1,0),(2,.5),(1,1),(0,.5)]
    for i,(x,y) in enumerate(coords,1):model.add_node(i,x,y,0.03*x*y if warp else 0.)
    element=ShellElement(1,list(range(1,9)),'steel',thickness=.02)
    model.add_element(1,element)
    u=np.random.default_rng(91).normal(0,1e-5,model.mesh.dof_manager.total_dofs)
    return model,element,u


def same(actual,expected):
    assert actual.keys()==expected.keys()
    for key,value in expected.items():
        if isinstance(value,str):assert actual[key]==value
        else:np.testing.assert_allclose(actual[key],value,rtol=2e-13,atol=1e-7)


@pytest.mark.parametrize('warp',[False,True])
@pytest.mark.parametrize('global_output',[False,True])
def test_q8_recovery_complete_scalar_oracle_and_mutations(monkeypatch,warp,global_output):
    import anyfesolver.recovery_shell_frames as frames
    model,element,u=model8(warp)
    batch=frames.native_recovery_frames
    for change in ('initial','material','node','displacement','thickness'):
        if change=='material':model.get_material('steel').elastic_modulus*=.8
        if change=='node':model.set_node_coordinates(3,2.01,1.,.04)
        if change=='displacement':u*=1.3
        if change=='thickness':element.thickness=.03
        monkeypatch.setattr(frames,'native_recovery_frames',batch)
        actual=recover_stress_result(model,u,return_global=global_output).element_stresses[1]
        monkeypatch.setattr(frames,'native_recovery_frames',lambda *a:None)
        expected=recover_stress_result(model,u,return_global=global_output).element_stresses[1]
        same(actual,expected)


def test_native_custom_points_and_override_fallback(monkeypatch):
    model,element,u=model8(True)
    points=np.array([[-.4,.2],[.7,-.8]])
    coords=element.get_node_coordinates(model.mesh)
    assert native_recovery_frames(element,coords,points,model.get_material('steel')) is not None
    original=element._local_frame_and_derivatives
    calls=[]
    def overridden(self,*args):
        calls.append(1)
        return original(*args)
    element._local_frame_and_derivatives=MethodType(overridden,element)
    assert native_recovery_frames(element,coords,points,model.get_material('steel')) is None
    element.compute_stresses(model.mesh,u,model.get_material('steel'),natural_points=points)
    assert len(calls)==len(points)
    del element._local_frame_and_derivatives
    monkeypatch.setattr(ShellElement,'_local_frame_and_derivatives',lambda *a:None)
    assert native_recovery_frames(element,coords,points,model.get_material('steel')) is None


def test_refusals_and_scalar_degenerate_map():
    model,element,u=model8()
    with pytest.raises(ValueError,match='not found'):
        recover_stress_result(model,u,RecoveryConfig(element_ids=[99]))
    for points in ([],[[2,0]],[[float('nan'),0]]):
        with pytest.raises(ValueError,match='natural stress points'):
            element.compute_stresses(model.mesh,u,model.get_material('steel'),natural_points=points)
    for node_id in model.mesh.nodes:model.set_node_coordinates(node_id,0.,0.,0.)
    assert native_recovery_frames(element,element.get_node_coordinates(model.mesh),element.gauss_points,model.get_material('steel')) is None
    with pytest.raises(ValueError,match='near-zero surface Jacobian'):
        element.compute_stresses(model.mesh,u,model.get_material('steel'))


@pytest.mark.parametrize('scale',[1e-7,2e-6,5e-6])
def test_near_threshold_uses_scalar_refusal_boundary(scale):
    model,element,u=model8()
    for node_id,node in model.mesh.nodes.items():
        model.set_node_coordinates(node_id,node.x*scale,node.y*scale,0.)
    coords=element.get_node_coordinates(model.mesh)
    frames = native_recovery_frames(element,coords,element.gauss_points,model.get_material('steel'))
    assert (frames is None) == (scale != 5e-6)
    if scale==1e-7:
        with pytest.raises(ValueError,match='near-zero surface Jacobian'):
            element.compute_stresses(model.mesh,u,model.get_material('steel'))
    else:
        element.compute_stresses(model.mesh,u,model.get_material('steel'))


def test_intervening_callback_and_material_protocol_stay_scalar(monkeypatch):
    model,element,u=model8(True)
    material=model.get_material('steel')
    coords=element.get_node_coordinates(model.mesh)
    original=element._material_angle
    calls=[]
    def angle(self,frame):
        calls.append(1)
        return original(frame)
    element._material_angle=MethodType(angle,element)
    assert native_recovery_frames(element,coords,element.gauss_points,material) is None
    element.compute_stresses(model.mesh,u,material)
    assert len(calls)==len(element.gauss_points)
    del element._material_angle
    class CustomMaterial(Material):
        pass
    custom=CustomMaterial('custom',210e9,.3,7850)
    assert native_recovery_frames(element,coords,element.gauss_points,custom) is None
    material.elastic_modulus=np.float64(210e9)
    assert native_recovery_frames(element,coords,element.gauss_points,material) is None


def test_skewed_rotated_frame_scalar_parity():
    model,element,_=model8(True)
    rotation=np.linalg.qr(np.random.default_rng(11).normal(size=(3,3)))[0]
    coords=element.get_node_coordinates(model.mesh)
    coords[:,0]+=2.7*coords[:,1]
    coords=coords@rotation
    rows=native_recovery_frames(element,coords,element.gauss_points,model.get_material('steel'))
    assert rows is not None
    for (xi,eta),(n,frame,dx,dy) in zip(element.gauss_points,rows):
        expected_n,a,b=element.compute_shape_functions(xi,eta)
        expected_frame,expected_dx,expected_dy,_=element._local_frame_and_derivatives(coords,a,b)
        for actual,expected in ((n,expected_n),(frame,expected_frame),(dx,expected_dx),(dy,expected_dy)):
            np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=1e-14)


def test_callback_mutating_frame_dispatch_preserves_later_refusal():
    model,element,u=model8()
    original=element._material_angle
    def angle(self,frame):
        def reject(*args):
            raise ValueError('mutated frame dispatch')
        self._local_frame_and_derivatives=reject
        return original(frame)
    element._material_angle=MethodType(angle,element)
    with pytest.raises(ValueError,match='mutated frame dispatch'):
        element.compute_stresses(model.mesh,u,model.get_material('steel'))
