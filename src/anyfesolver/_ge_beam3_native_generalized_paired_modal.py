"""Private paired successor; current material/state capture stays unchanged."""
from ._ge_beam3_native_generalized_factor_modal import prepare
from ._native_paired_factor_chain_modes import solve_paired_factor_chain_modes


def solve_modes(model,element_states,displacement,section_inertias,nodal_spatial_dead_forces,*,
        bounds,num_modes=6,root_width=1e-10,relative_width=1e-12,cancellation_token=None):
    packet,guard=prepare(model,element_states,displacement,section_inertias,nodal_spatial_dead_forces,
        cancellation_token=cancellation_token)
    result=solve_paired_factor_chain_modes(packet.left,packet.right,packet.geometric,packet.kinetic,
        packet.base.free_dofs,packet.base.algebraic_dofs,bounds=bounds,num_modes=num_modes,
        root_width=root_width,relative_width=relative_width,cancellation_token=cancellation_token)
    guard();return packet,result
