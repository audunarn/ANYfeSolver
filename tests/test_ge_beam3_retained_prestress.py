"""Actual nodal Newton preload, exact-support family checks and Euler search."""
from dataclasses import replace
from hashlib import sha256
import json
import numpy as np
import pytest
from anyfesolver import _ge_beam3_retained_nodal_loading as nodal
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from anyfesolver._native_paired_factor_chain_modes import solve_paired_factor_chain_modes as paired
from docs.reference_cases.ge_beam3_exact_family_partition import partition
from test_ge_beam3_native_generalized_modal import make
from test_ge_beam3_schur_line_program import save

from docs.reference_cases.ge_beam3_retained_prestress_point import capture,family_modes,evaluate_point


@pytest.mark.parametrize('mutation',('valid','material','kinetic','geometric','nonfinite','layout'))
def test_original_support_partition(mutation,tmp_path):
    m,p,masses,result=capture(0.,1);assert result.status=='completed',result.failure
    packet,guard=nodal.prepare(m,p,result.checkpoint,masses,expected_sha256=sha256(result.checkpoint).hexdigest())
    n=m.mesh.dof_manager.total_dofs
    if mutation=='valid':
        families=partition(packet,n,guard)
        roots={name:float(family_modes(f).eigenvalues[0]) for name,f in families.items()}
        full=paired(packet.left,packet.right,packet.geometric,packet.kinetic,packet.free_dofs,packet.algebraic_dofs,bounds=(-100.,1e6),num_modes=1)
        assert abs(min(roots.values())-full.eigenvalues[0])<=1e-11*max(1.,abs(full.eigenvalues[0]))
        save(tmp_path/'partition.json',dict(roots=roots,full_root=float(full.eigenvalues[0]),families=families))
        return
    if mutation=='material':
        right=packet.right.copy();row=int(np.flatnonzero(np.any(right[:,::6]!=0.,axis=1))[0]);right[row,1]=1e-200
        packet=replace(packet,right=right)
    elif mutation=='kinetic':
        b=packet.kinetic.copy();row=int(np.flatnonzero(np.any(b[:,::6]!=0.,axis=1))[0]);b[row,1]=1e-200
        packet=replace(packet,kinetic=b)
    elif mutation=='geometric':
        g=packet.geometric.copy();g[0,1]=g[1,0]=1e-200;packet=replace(packet,geometric=g)
    elif mutation=='nonfinite':
        right=packet.right.copy();right[0,0]=np.nan;packet=replace(packet,right=right)
    else:n+=1
    with pytest.raises(ValueError):partition(packet,n)
    save(tmp_path/'rejected.json',dict(mutation=mutation,rejected_without_small_entry_threshold=True))


@pytest.mark.parametrize('macros',(1,2,4,8))
def test_actual_prestress_euler(macros,tmp_path):
    euler=float(np.pi**2/16);rows=[];constant_families={}
    def evaluate(compression):
        row=evaluate_point(compression,macros,len(rows),constant_families,tmp_path)
        rows.append(row)
        return row['lambda_min']
    tension=evaluate(-.5*euler);unloaded=evaluate(0.);compressed=evaluate(.5*euler)
    assert tension>unloaded>compressed>0.
    low,high=0.,2*euler;assert evaluate(high)<0.
    low_value,high_value=unloaded,rows[-1]['lambda_min']
    for _ in range(18):
        midpoint=float((low+high)/2);value=evaluate(midpoint)
        if value>0.:low,low_value=midpoint,value
        else:high,high_value=midpoint,value
    critical=float((low+high)/2);error=abs(critical/euler-1.)
    assert low_value>0. and high_value<=0. and (high-low)/euler<1e-5
    if macros==8:assert error<.02
    save(tmp_path/'buckling.json',dict(macros=macros,rows=rows,critical_bracket=[low,high],
        bracket_eigenvalues=[low_value,high_value],critical_midpoint=critical,euler_reference=euler,
        relative_euler_error=error,finest_engineering_gate=(macros==8),
        search='18_BISECTIONS_OF_ACTUAL_NODAL_NEWTON_PRELOADED_SIGNED_FACTOR_FAMILIES',
        current_frequency_is_not_a_load_factor=True,production_qualified=False,full_spatial_postbuckling_qualified=False))
