"""Coverage and enclosure tests for the fixed proof partition (not new contact faces)."""
from fractions import Fraction as F
from types import SimpleNamespace
import numpy as np
import pytest
from test_general_contact_corner_contact import make_panel, _sphere
import anysolver.general_contact.corner_contact_certificate as cert


def _area(points):
    # Exact shoelace area, preserving orientation.
    return sum(F(float(points[k][0]))*F(float(points[(k+1)%len(points)][1]))
               -F(float(points[(k+1)%len(points)][0]))*F(float(points[k][1]))
               for k in range(len(points)))/2


def test_fixed_partitions_cover_domains_once_and_keep_orientation():
    tri = cert._S3_POINTS[:,1:]
    assert sum(_area(tri[list(c)]) for c in cert._S3_CELLS) == F(1,2)
    assert all(_area(tri[list(c)]) == F(1,8) for c in cert._S3_CELLS)
    quad = cert._Q4_PARAMS
    assert sum(_area(quad[list(c)]) for c in cert._Q4_CELLS) == 1
    assert all(_area(quad[list(c)]) == F(1,4) for c in cert._Q4_CELLS)
    # All interior oriented edges cancel; only the subdivided parent boundary remains.
    for points,cells,boundary_count in ((tri,cert._S3_CELLS,6),(quad,cert._Q4_CELLS,8)):
        edges = {}
        for cell in cells:
            for a,b in zip(cell,cell[1:]+cell[:1]):
                key = tuple(sorted((a,b)))
                edges[key] = edges.get(key,0)+(1 if a<b else -1)
        assert sum(v!=0 for v in edges.values()) == boundary_count
        assert all(abs(v)<=1 for v in edges.values())


def test_exact_interpolation_encloses_positions_input_errors_and_rounding():
    p = np.array([[.1,.2,.3],[.8,.4,.2],[.7,.9,.6],[.2,.8,.4]])
    e = np.full_like(p,1e-13)
    out, radius = cert._exact_affine_points(p,e,cert._Q4_WEIGHTS)
    for j,w in enumerate(cert._Q4_WEIGHTS):
        for axis in range(3):
            exact = sum(F(float(a))*F(float(p[k,axis])) for k,a in enumerate(w))
            required = sum(F(float(a))*F(float(e[k,axis])) for k,a in enumerate(w))
            required += abs(F(float(out[j,axis]))-exact)
            assert F(float(radius[j,axis])) >= required
    # Restriction of a Q1 patch remains that SAME bilinear surface, including twist.
    for cell in cert._Q4_CELLS:
        u,v = np.mean(cert._Q4_PARAMS[list(cell)],axis=0)
        expected = np.array([(1-u)*(1-v),u*(1-v),u*v,(1-u)*v]) @ p
        np.testing.assert_allclose(np.mean(out[list(cell)],axis=0),expected,rtol=0,atol=1e-14)
    t,te = cert._exact_affine_points(p[:3],e[:3],cert._S3_POINTS)
    np.testing.assert_array_equal(t[:3],p[:3])
    assert np.all(te >= 0)


def test_fixed_wrapper_retains_actual_penetration_refusal_and_parent_identity():
    panel,ppose = make_panel(lambda x,y:0.,nx=1)
    smap,spose = _sphere(-.1)
    before = (spose.positions.copy(),ppose.positions.copy(),
              spose.position_errors.copy(),ppose.position_errors.copy())
    result = cert.whole_face_certificates(smap,spose,panel,ppose)
    assert result['minimum_bound_m'] == pytest.approx(-.1,abs=1e-6)
    assert result['minimum_bound_m'] < -cert.PENETRATION_LIMIT_M
    assert result['pairs_total'] == 1 and result['pairs_certified'] == 1
    fixed = result['fixed_partition']
    assert fixed['levels'] == 2
    assert fixed['s3_cells'] == fixed['q4_cells'] == 16
    assert 16 < fixed['pairs_checked']+fixed['pairs_pruned'] <= 272
    assert fixed['unavailable_cells'] == []
    np.testing.assert_array_equal(spose.positions,before[0])
    np.testing.assert_array_equal(ppose.positions,before[1])
    np.testing.assert_array_equal(spose.position_errors,before[2])
    np.testing.assert_array_equal(ppose.position_errors,before[3])


def test_unavailable_parent_or_child_cannot_be_hidden_by_refinement(monkeypatch):
    panel,ppose = make_panel(lambda x,y:0.,nx=1)
    smap,spose = _sphere(-.1)
    calls = []
    def unavailable_parent(*a,**kw):
        calls.append(1)
        return -np.inf
    monkeypatch.setattr(cert,'q4_s3_whole_pair_swept_lower',unavailable_parent)
    r = cert.whole_face_certificates(smap,spose,panel,ppose)
    assert len(calls)==1 and r['unavailable_pairs']==[(0,0)]
    calls.clear()
    def unavailable_child(*a,**kw):
        calls.append(1)
        return -np.inf if len(calls)==2 else -.1
    monkeypatch.setattr(cert,'q4_s3_whole_pair_swept_lower',unavailable_child)
    r = cert.whole_face_certificates(smap,spose,panel,ppose)
    assert r['minimum_bound_m']==-np.inf and r['unavailable_pairs']==[(0,0)]
    assert len(r['fixed_partition']['unavailable_cells'])==1
    assert len(r['fixed_partition']['unavailable_cells'][0][2])==1


def test_invalid_subdivision_errors_are_refused():
    p = np.zeros((3,3))
    for error in (np.full_like(p,-1.),np.full_like(p,np.nan)):
        with pytest.raises(ValueError):
            cert._exact_affine_points(p,error,cert._S3_POINTS)


def test_second_level_unavailable_cannot_be_hidden_by_ancestors(monkeypatch):
    panel,ppose = make_panel(lambda x,y:0.,nx=1)
    smap,spose = _sphere(-.1)
    calls = []
    # Visit every cell so calls 1/2/3 are coarse/intermediate/leaf.
    monkeypatch.setattr(cert,'directional_exclusion',lambda *a:
                        (np.array([False]),np.array([False])))
    def bound(*a,**kw):
        calls.append(1)
        return -np.inf if len(calls)==3 else -.1
    monkeypatch.setattr(cert,'q4_s3_whole_pair_swept_lower',bound)
    r = cert.whole_face_certificates(smap,spose,panel,ppose)
    assert r['minimum_bound_m'] == -np.inf
    assert r['unavailable_pairs'] == [(0,0)]
    fixed = r['fixed_partition']
    assert fixed['unavailable_cells'] == [(0,0,(0,0),(0,0))]
    assert fixed['pairs_checked'] == 272 and fixed['pairs_pruned'] == 0
    assert len(calls) == 273


@pytest.mark.parametrize('intermediate,leaf,expected', [(-.2,-.1,-.1),(-.1,-.2,-.1)])
def test_recursive_bound_retains_both_ancestor_and_complete_child_bounds(
        monkeypatch,intermediate,leaf,expected):
    monkeypatch.setattr(cert,'directional_exclusion',lambda *a:
                        (np.array([False]),np.array([False])))
    calls = []
    def bound(*a,**kw):
        index = len(calls)
        calls.append(1)
        return intermediate if index % 17 == 0 else leaf
    monkeypatch.setattr(cert,'q4_s3_whole_pair_swept_lower',bound)
    lower,checked,pruned,failed = cert._fixed_pair_lower(
        np.zeros((3,3)),np.zeros((3,3)),np.zeros((4,3)),np.zeros((4,3)))
    assert lower == expected
    assert checked == 272 and pruned == 0 and failed == []


@pytest.mark.parametrize('weights,cells,parent,area', [
    (cert._S3_POINTS,cert._S3_CELLS,np.eye(3),F(1,2)),
    (cert._Q4_WEIGHTS,cert._Q4_CELLS,
     np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]]),F(1)),
])
def test_recursive_partition_coverage_and_rounding_enclosures(weights,cells,parent,area):
    # Compose restrictions exactly. P1 barycentric and Q1 corner weights are
    # dyadic, so this also verifies the original parameter-domain leaf tiling.
    first = weights @ parent
    leaves = []
    p = np.array([[.1,.2,.3],[.8,.4,.2],[.7,.9,.6],[.2,.8,.4]])[:len(parent)]
    e = np.full_like(p,1e-13)
    before = p.copy(),e.copy()
    out,radius = cert._exact_affine_points(p,e,weights)
    for cell in cells:
        indices = list(cell)
        parameters = weights @ first[indices]
        leaves.extend(parameters[list(c)] for c in cells)
        child,child_radius = cert._exact_affine_points(out[indices],radius[indices],weights)
        for j,w in enumerate(weights):
            composed = [sum(F(float(w[k]))*F(float(weights[indices[k],i]))
                            for k in range(len(indices))) for i in range(len(parent))]
            for axis in range(3):
                exact = sum(a*F(float(p[i,axis])) for i,a in enumerate(composed))
                required = sum(a*F(float(e[i,axis])) for i,a in enumerate(composed))
                required += abs(F(float(child[j,axis]))-exact)
                assert F(float(child_radius[j,axis])) >= required
    planar = [leaf[:,1:] if len(parent)==3 else leaf[:,:2] for leaf in leaves]
    assert len(planar)==16
    assert all(_area(leaf)==area/16 for leaf in planar)
    assert sum(_area(leaf) for leaf in planar)==area
    edges = {}
    for leaf in planar:
        vertices = [tuple(point) for point in leaf]
        for a,b in zip(vertices,vertices[1:]+vertices[:1]):
            key = tuple(sorted((a,b)))
            edges[key] = edges.get(key,0)+(1 if a<b else -1)
    assert sum(v!=0 for v in edges.values()) == (12 if len(parent)==3 else 16)
    assert all(abs(v)<=1 for v in edges.values())
    np.testing.assert_array_equal(p,before[0])
    np.testing.assert_array_equal(e,before[1])


def test_pruned_subtrees_are_not_counted_as_visited_leaves(monkeypatch):
    monkeypatch.setattr(cert,'directional_exclusion',lambda *a:
                        (np.array([True]),np.array([False])))
    def unexpected_bound(*a,**kw):
        pytest.fail('A pruned subtree must not evaluate a bound')
    monkeypatch.setattr(cert,'q4_s3_whole_pair_swept_lower',unexpected_bound)
    lower,checked,pruned,failed = cert._fixed_pair_lower(
        np.zeros((3,3)),np.zeros((3,3)),np.zeros((4,3)),np.zeros((4,3)))
    assert lower == 0. and checked == 0 and pruned == 16 and failed == []
