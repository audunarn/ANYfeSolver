"""Closed load boundary for the private nodal-force-only native static adapter.

This is not a distributed-load implementation. In particular shell pressure
fallback and untransformed nodal couples cannot qualify this beam's work.
"""
import numpy as np


def assemble_nodal_forces(load,mesh,dof_manager,*,guard,activity=None):
    from .boundary import LoadCase
    from ._ge_beam3_native_fibre_static_element import NativeFibreStaticElement
    if type(load) is not LoadCase or dof_manager is not mesh.dof_manager:
        raise ValueError('private native beam requires exact load and DOF ownership')
    expected={'name','nodal_loads','element_loads','pressure_loads','gravity','added_node_masses','follower_pressure'}
    if set(vars(load))!=expected or type(load.name) is not str:
        raise ValueError('private native beam load schema not admitted')
    if not mesh.elements or any(type(e) is not NativeFibreStaticElement for e in mesh.elements.values()):
        raise ValueError('private native beam standalone load boundary')
    for element in mesh.elements.values():element._check(mesh)
    if any(type(getattr(load,k)) is not dict for k in ('nodal_loads','element_loads','pressure_loads','added_node_masses')):
        raise ValueError('private native beam requires owned load dictionaries')
    if load.element_loads or load.pressure_loads or load.gravity is not None or load.added_node_masses or load.follower_pressure is not False:
        raise ValueError('private native beam admits nodal forces only; pressure, line work, gravity and follower loads require native qualification')
    if activity is not None or mesh.element_activity is not None or mesh.point_masses or getattr(mesh,'concentrated_masses',None):
        raise ValueError('private native beam activity/dynamics not admitted')
    force=np.zeros(dof_manager.total_dofs)
    for node,value in load.nodal_loads.items():
        if type(node) is not int or node not in mesh.nodes or type(value) is not np.ndarray or value.dtype!=np.float64 or value.shape!=(6,):
            raise ValueError('private native beam exact nodal force row required')
        owned=value.copy()
        if not np.isfinite(owned).all() or np.any(owned[3:]):
            raise ValueError('private native beam nodal couples require native work-conjugate routing')
        mapping=dof_manager.get_node_dofs(node)
        if len(mapping)!=6:raise ValueError('private native beam requires six nodal coordinates')
        force[list(mapping)]+=owned
    guard(stage='private native nodal-force observation')
    return force
