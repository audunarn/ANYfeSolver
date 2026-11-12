"""Bounded reference-area friction integration of replayed material history.

Normal work is supplied by the separate continuous normal potential. Integration
nodes have no physical history identity: every query reconstructs its own history
from the same declared path. Error estimates are developmental, not certified
quadrature bounds or acceptance of arbitrary contact histories.
"""
from dataclasses import dataclass,replace
from math import cos,sin,fsum

import numpy as np
from numpy.polynomial.legendre import leggauss

from .native_retained_station import _readonly
from .native_surface_extrema import straight_patch_s3_gap
from .native_surface_history import NativeSurfaceHistoryField,NativeSurfaceReplayBudget
from .native_surface_normal import _breakpoints
from .native_surface_front_precision import PrefixReplayField,contact_side_turning,final_segment_strip_work_bound
from .native_surface_path import material_path_pose


def _interior_material_coordinate(left,right,node,*,allow_rounding=False):
    value=(left+right)/2+(right-left)*float(node)/2
    if allow_rounding and not left<value<right:
        lower=float(np.nextafter(left,right))
        upper=float(np.nextafter(right,left))
        if lower<=upper:
            value=min(max(value,lower),upper)
    return value


def _captured_front_crossing_angles(coefficients,left,right,lower,upper):
    """Partition hints at crossings of two affine captured-pose fronts.

    Cross multiplication avoids dividing by a small slope while isolating
    the angular equation. These hints admit no contact or integral; the
    complete-domain estimator remains authoritative.
    """
    cuts=set()
    for index,(o1,s1,a1,b1) in enumerate(coefficients):
        if s1==0.:continue
        for o2,s2,a2,b2 in coefficients[index+1:]:
            if s2==0.:continue
            c=fsum((s2*o1,-s1*o2))
            a=fsum((s2*a1,-s1*a2));b=fsum((s2*b1,-s1*b2))
            if a==0. and b==0.:continue
            for theta in _breakpoints(c,c,a,b,lower,upper):
                if not lower<theta<upper:continue
                residual=fsum((c,a*cos(theta),b*sin(theta)))
                scale=abs(c)+abs(a)+abs(b)
                if abs(residual)>32.*np.finfo(float).eps*scale:continue
                x1=-fsum((o1,a1*cos(theta),b1*sin(theta)))/s1
                x2=-fsum((o2,a2*cos(theta),b2*sin(theta)))/s2
                if left<x1<right and left<x2<right:cuts.add(float(theta))
    return tuple(sorted(cuts))


def _captured_front_structure(coefficients,left,right,angle):
    roots=[]
    for index,(offset,slope,a,b) in enumerate(coefficients):
        if slope==0.:continue
        root=-fsum((offset,a*cos(angle),b*sin(angle)))/slope
        if left<root<right:roots.append((root,index))
    return tuple(index for _,index in sorted(roots))


def _captured_front_angular_motion_dominates(coefficients,left,right,lower,upper):
    """Zero-replay refinement hint when front motion exceeds window width.

    This chooses an axis only. It is neither a support certificate nor an
    error estimate; the full low/high and global work gates still decide.
    """
    angles={lower,upper,(lower+upper)/2}
    for offset,slope,a,b in coefficients:
        angles.update(_breakpoints(offset+slope*left,offset+slope*right,a,b,lower,upper))
    traces=[];separation=[]
    for theta in sorted(angles):
        roots=[]
        for index,(offset,slope,a,b) in enumerate(coefficients):
            if slope==0.:continue
            root=-fsum((offset,a*cos(theta),b*sin(theta)))/slope
            if left<root<right:roots.append((root,index))
        ordered=sorted(roots)
        separation.extend(b[0]-a[0] for a,b in zip(ordered[:-1],ordered[1:]) if b[0]>a[0])
        traces.append(dict((index,root) for root,index in roots))
    if not separation:return False
    excursions=[]
    for index in range(len(coefficients)):
        values=[row[index] for row in traces if index in row]
        if len(values)>1:excursions.append(max(values)-min(values))
    return bool(excursions and max(excursions)>=min(separation))


def _captured_front_contact_probe(coefficients,left,right,angle):
    """Interior sample beside the final captured front, not a certificate.

    The nearest captured front bounds the sample's material interval.
    Its position moves with angle; all evaluations replay genuine history.
    """
    offset,slope,a,b=coefficients[-1]
    if slope==0.:return None
    final=-fsum((offset,a*cos(angle),b*sin(angle)))/slope
    if not left<final<right:return None
    roots=[left,right]
    for o,k,c,d in coefficients:
        if k:
            root=-fsum((o,c*cos(angle),d*sin(angle)))/k
            if left<root<right:roots.append(root)
    if slope<0.:
        neighbor=min(x for x in roots if x>final)
    else:neighbor=max(x for x in roots if x<final)
    sample=final+1e-3*(neighbor-final)
    if not min(final,neighbor)<sample<max(final,neighbor):return None
    return sample


def _final_segment_depth(prepared_band,partition,left,right):
    """Certified contact-side penetration depth of one strip micro-interval.

    Every certified component's extrema (its two endpoint fronts and any
    stationary root inside it) are already isolated with binary64 intervals
    by the prepared partition. This mirrors the whole-history depth rule with
    exact per-angle intervals instead of angular envelopes.
    """
    from .native_surface_front import _front_components
    components=_front_components(prepared_band.cover)
    roots=partition[2];endpoints=partition[3]
    if len(endpoints)!=2*len(components):
        raise ValueError('complete prepared front extrema required')
    depth=0.
    for i,(start,end,_,_) in enumerate(components):
        extrema=(*endpoints[2*i:2*i+2],*(r for r in roots if start<=r.fraction<=end))
        bl,bu=prepared_band.endpoints[2*i][2][:2]
        if bl>0:depth=max(depth,max(e.axial_interval[1] for e in extrema)-left)
        elif bu<0:depth=max(depth,right-min(e.axial_interval[0] for e in extrema))
        else:raise ValueError('final-segment strip depth denominator unresolved')
    return depth


class _MaterialQuadratureSubdivision(ValueError):
    """One uncertified quadrature query requests a complete panel refinement."""
    def __init__(self,xi,angle):
        super().__init__('material quadrature query requires subdivision')
        self.xi=xi;self.angle=angle


class NativeFrictionIntegrationLimit(MemoryError):
    """Resource non-success with a compact diagnostic, never softened work."""
    def __init__(self,message,diagnostics):
        super().__init__(message)
        self.integration_diagnostics=diagnostics


def _embedded_six_eight_rule():
    """Positive degree-five control embedded in unchanged Gauss8 work.

    Remove the outer Gauss8 pair. Symmetry and the even moments 0,2,4
    determine six positive control weights. Every low query is therefore
    an exact high-query coordinate, without a zero central node or extrapolation.
    This control has lower formal order than Gauss4; unchanged error tolerances
    and adaptive subdivision still govern the accepted high-rule integral.
    """
    high,weights=leggauss(8);positive=high[4:7]
    control=np.linalg.solve(np.array((np.ones(3),positive**2,positive**4)),
        np.array((1.,1/3,1/5)))
    if np.any(control<=0):raise ValueError('embedded friction weights must be positive')
    return ((high[1:7],np.concatenate((control[::-1],control))),(high,weights))



def _embedded_four_six_rule():
    """Positive cubic control at a subset of the Gauss6 work coordinates."""
    high,weights=leggauss(6)
    positive=high[3:5]
    control=np.linalg.solve(np.array((np.ones(2),positive**2)),np.array((1.,1/3)))
    if np.any(control<=0):raise ValueError('embedded friction weights must be positive')
    return ((high[1:5],np.concatenate((control[::-1],control))),(high,weights))

def _square_cubature_degree7_angular_groups():
    """Angular-ordinate groups of the positive 12-point square cubature.

    Clean-room moment derivation (exact Q(sqrt(583)) verification in the
    restart evidence directory): on the reference square [-1,1]^2 the rule
    of four axis nodes (+-r1,0),(0,+-r1) of per-node weight a=98/405 with
    r1^2=6/7 plus two diagonal four-node groups (+-r2,+-r2), (+-r3,+-r3)
    of per-node weights b,c with r2^2,r3^2 the roots of 287 d^2-228 d+27,
    b+c=307/405, b r2^2+c r3^2=31/135, is exact through total degree
    seven with positive interior nodes, and not exact at degree eight.
    Odd monomials cancel by the independent sign symmetry of the node set.

    Twelve nodes replace the sixteen tensor Gauss4 points, so the private
    variable route spends 12+9=21 rather than 25 material samples per
    mapped rectangle. Nodes are grouped by angular ordinate; each group is
    (ordinate, half the ordinate's total node weight, axial abscissas,
    axial weight shares) with share*half_total reproducing each node's
    square weight, so every wx*wa product and every wa-only strip fallback
    weight below remain the proper square Jacobian factors.
    """
    from math import sqrt
    axis_weight=98./405.;axis_radius=sqrt(6./7.)
    root=sqrt(583.)
    d2=(114.+3.*root)/287.;d3=(114.-3.*root)/287.
    diagonal_total=307./405.;diagonal_first=31./135.
    b=(diagonal_first-diagonal_total*d3)/(d2-d3);c=diagonal_total-b
    if not (0.<b<diagonal_total and 0.<c<diagonal_total and 0.<d3<d2<1.):
        raise ValueError('square cubature constants out of range')
    r2=sqrt(d2);r3=sqrt(d3)
    def group(ordinate,pairs):
        total=sum(weight for _,weight in pairs)
        return (ordinate,total/2.,tuple(abscissa for abscissa,_ in pairs),
            tuple(2.*weight/total for _,weight in pairs))
    return (group(0.,((axis_radius,axis_weight),(-axis_radius,axis_weight))),
        group(axis_radius,((0.,axis_weight),)),group(-axis_radius,((0.,axis_weight),)),
        group(r2,((r2,b),(-r2,b))),group(-r2,((r2,b),(-r2,b))),
        group(r3,((r3,c),(-r3,c))),group(-r3,((r3,c),(-r3,c))))

@dataclass(frozen=True)
class NativeSurfaceFrictionIntegrationSettings:
    force_absolute_tolerance: float
    tangent_absolute_tolerance: float
    energy_absolute_tolerance: float
    characteristic_length_m: float
    maximum_panels: int=128
    maximum_queries: int=2048
    maximum_point_paths: int=8192
    maximum_support_intervals: int=128
    maximum_cache_bytes: int=67108864

    def __post_init__(self):
        for name in ('force_absolute_tolerance','tangent_absolute_tolerance','energy_absolute_tolerance',
                     'characteristic_length_m'):
            value=float(getattr(self,name))
            if not np.isfinite(value) or value<=0:raise ValueError('positive dimensional friction integration controls required')
            object.__setattr__(self,name,value)
        for name in ('maximum_panels','maximum_queries','maximum_point_paths','maximum_support_intervals','maximum_cache_bytes'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError('positive integer friction integration caps required')


@dataclass(frozen=True)
class NativeSurfaceFrictionIntegral:
    residual: np.ndarray
    tangent: np.ndarray
    stored_energy_j: float
    dissipated_increment_j: float
    released_storage_j: float
    cumulative_dissipation_j: float
    diagnostics: dict


def _coefficients(field,pose,cell):
    corners=field.patch.target.adapter.current_positions(pose.shell_displacement)
    normal=np.asarray(field.patch.target.physical_pose(pose.shell_displacement).material_frame)[:,2]
    frame=pose.cell_rotations[cell]@field.patch.rule.samples[0].station.reference.nodal_triads[0]
    radius=field.patch.rule.definition['radius']
    base=np.array([fsum((float(pose.positions[cell,i]),float(pose.position_low[cell,i]),
                        -float(corners[0,i]))) for i in range(3)])
    chord=np.array([fsum((float(pose.positions[cell+1,i]),float(pose.position_low[cell+1,i]),
                         -float(pose.positions[cell,i]),-float(pose.position_low[cell,i]))) for i in range(3)])
    # g(xi,angle) = offset + slope*xi + a*cos(angle)+b*sin(angle).
    slope=float(normal@chord)
    offset=float(normal@base)+(slope if cell==0 else 0.)
    return offset,slope,radius*float(normal@frame[:,1]),radius*float(normal@frame[:,2])


_DECLARED_PUBLIC_AXIAL_FRONT_UNCERTAINTY=1e-7
# Declared certified axial-front uncertainty of the public moving-front route:
# the exact certificate resolution already carried by every development
# delivery receipt for the saved moving-front case. It bounds the prepared
# stationary root coordinate intervals; it is not an error tolerance, and no
# acceptance gate or physical meaning depends on its value.


def _captured_fronts_move_between_poses(field):
    """Deterministic captured-pose front-motion detector; no replay queries.

    Exact per-pose linear gap models decide, at deterministic sample angles
    (every captured front's angular breakpoints plus the declared range
    endpoints and midpoints between consecutive samples), whether any
    consecutive pose pair has differing interior front roots at a shared
    angle. True is a sufficient certificate that quadrature panels are
    front-mixed, so the legacy whole-panel classification route is
    inapplicable; False keeps the unchanged legacy route and asserts
    nothing about front motion elsewhere.
    """
    rule=field.patch.rule;lo,hi=rule.definition['xi_range'];alo,ahi=rule.definition['angle_range']
    poses=[field.initial]
    for segment in field.segments:
        poses.append(material_path_pose(poses[-1],segment.translations,segment.angular,
            segment.shell_increment,1.))
    for cell,left,right in ((0,max(-1.,lo),min(0.,hi)),(1,max(0.,lo),min(1.,hi))):
        if right<=left:continue
        coefficients=[_coefficients(field,pose,cell) for pose in poses]
        samples={alo,ahi}
        for offset,slope,a,b in coefficients:
            samples.update(_breakpoints(offset+slope*left,offset+slope*right,a,b,alo,ahi))
        ordered=sorted(value for value in samples if alo<=value<=ahi)
        for x,y in zip(ordered[:-1],ordered[1:]):samples.add((x+y)/2)
        for index in range(len(poses)-1):
            for theta in sorted(value for value in samples if alo<=value<=ahi):
                roots=[]
                for pose in (index,index+1):
                    offset,slope,a,b=coefficients[pose]
                    if not slope:roots.append(None);continue
                    root=-(offset+a*cos(theta)+b*sin(theta))/slope
                    roots.append(float(root) if left<root<right else None)
                if None in roots:continue
                if abs(roots[0]-roots[1])>64*np.finfo(float).eps*max(1.,abs(roots[0]),abs(roots[1])):
                    return True
    return False


def integrate_open_event_friction_fronts(field,settings,*,maximum_axial_uncertainty):
    """Development open-endpoint pulse integration with moving axial cuts.

    This explicit helper is not an analysis-owner/restart admission. Each angular
    band needs complete stationary-branch coverage; uncertainty strips are fully
    integrated. Closed endpoint regions are excluded from this first work slice.
    """
    if not np.isfinite(maximum_axial_uncertainty) or maximum_axial_uncertainty<=0:
        raise ValueError('positive explicit axial-front uncertainty required')
    if field.patch.pairs[0].activation_policy!='transverse_event_sequence':
        raise ValueError('moving-front development requires event-sequence policy')
    return integrate_material_friction(field,settings,_front_uncertainty=maximum_axial_uncertainty)


def integrate_event_friction_fronts(field,settings,*,maximum_axial_uncertainty):
    """Development mixed endpoint integration with locally open front strips.

    One stationary branch over the complete history; closed material outside certified open
    strips contributes its full endpoint force and tangent through point replay.
    """
    if not np.isfinite(maximum_axial_uncertainty) or maximum_axial_uncertainty<=0:
        raise ValueError('positive explicit axial-front uncertainty required')
    if field.patch.pairs[0].activation_policy!='transverse_event_sequence':
        raise ValueError('moving-front development requires event-sequence policy')
    return integrate_material_friction(field,settings,_front_uncertainty=maximum_axial_uncertainty,
        _local_front_strips=True)


def integrate_variable_event_friction_fronts(field,settings,*,maximum_axial_uncertainty):
    """Explicit development route for captured-endpoint stationary transitions.

    All angular guard bands remain represented. Point replay supplies closed
    endpoint forces/tangents. A certified-open endpoint has zero R/K/storage;
    unresolved earlier strip work is retained as an integral enclosure. A
    complete-history work bound controls the spatial work error.
    This is not analysis-owner or restart admission.
    """
    if not np.isfinite(maximum_axial_uncertainty) or maximum_axial_uncertainty<=0:
        raise ValueError('positive explicit axial-front uncertainty required')
    if field.patch.pairs[0].activation_policy!='transverse_event_sequence':
        raise ValueError('moving-front development requires event-sequence policy')
    return integrate_material_friction(field,settings,_front_uncertainty=maximum_axial_uncertainty,
        _local_front_strips=True,_variable_endpoint_branches=True)


def _safeguarded_branch_refinement(probe,a,b,*,relative_width=2.**-20,max_iterations=44):
    """Bounded stick/slip bracket refinement: Illinois secant with a forced-bisection safeguard.

    probe(x) must return the genuine (branch,margin) pair of the material
    point: branch is the constitutive trial branch ('stick' or 'slip') and
    margin is a read-only scalar Coulomb yield margin (None when the genuine
    replay exposes none). Only the branch bit updates the bracket, so every
    retained bracket contains a genuine branch transition; the margin merely
    proposes split points, and the Illinois retention correction (halving
    the proposal weight of an endpoint retained across consecutive
    iterations) removes the one-sided stagnation of plain regula falsi.
    Whenever an iteration fails to halve the bracket, the next split is a
    forced bisection, so after n iterations the bracket width is at most
    (b-a)*2**(-(n//2)) regardless of margin behavior: resolution never
    depends on margin smoothness. The default criterion 2**-20 of the
    bracket gives an algorithmic coordinate resolution near 1e-7 on a
    .08 edge; the dimensioned integration gates remain decisive. Forty-four
    iterations leave at least twenty-two guaranteed halvings, so the
    criterion stays reachable even under complete secant stagnation. This
    locator is not a uniqueness certificate: equal endpoint branches are
    not transition absence (a hidden interior transition between
    same-branch endpoints is reported as 'equal-endpoint-branches', never
    as a certificate), several transitions inside one bracket collapse to
    the one the branch bit exposes, and an exhausted budget returns the
    final bracket as a clearly labelled refinement hint with no
    coordinate. The caller keeps its complete-domain coverage and error
    gates decisive.
    """
    a=float(a);b=float(b)
    def report(resolved,coordinate,bracket,width,criterion,iterations,probes,reason):
        return dict(resolved=resolved,coordinate=coordinate,bracket=bracket,width=width,
            criterion=criterion,iterations=iterations,probes=probes,reason=reason)
    if not a<b:
        return report(False,None,(a,b),0.,0.,0,0,'degenerate-edge')
    criterion=relative_width*(b-a)
    branch_a,margin_a=probe(a);branch_b,margin_b=probe(b);probes=2
    if branch_a not in ('stick','slip') or branch_b not in ('stick','slip'):
        return report(False,None,(a,b),b-a,criterion,0,probes,'unprobed-endpoint')
    if branch_a==branch_b:
        # Sampled endpoint equality is not transition absence: a hidden
        # interior transition between same-branch endpoints stays the
        # caller's unresolved error, never a certificate here.
        return report(False,None,(a,b),b-a,criterion,0,probes,'equal-endpoint-branches')
    for x,margin in ((a,margin_a),(b,margin_b)):
        if margin==0.:
            return report(True,x,(x,x),0.,criterion,0,probes,'yield-margin-zero')
    iterations=0;retained=None;force_bisect=False
    while b-a>criterion and iterations<max_iterations:
        iterations+=1
        split=None
        if not force_bisect and margin_a is not None and margin_b is not None and margin_a!=margin_b:
            proposal=a-(b-a)*margin_a/(margin_b-margin_a)
            if a<proposal<b:split=proposal
        if split is None:split=(a+b)/2
        branch_split,margin_split=probe(split);probes+=1
        if branch_split not in ('stick','slip'):
            return report(False,None,(a,b),b-a,criterion,iterations,probes,'inconsistent-interior')
        if margin_split==0.:
            return report(True,split,(split,split),0.,criterion,iterations,probes,'yield-margin-zero')
        previous=b-a
        if branch_split==branch_a:
            a,margin_a=split,margin_split;retained_end='b'
        else:
            b,margin_b=split,margin_split;retained_end='a'
        if retained==retained_end:
            # Illinois retention correction: the proposal weight of an
            # endpoint retained across consecutive iterations is halved so
            # the secant cannot stall against it. The branch bit alone
            # decides the bracket; this weight only proposes split points.
            if retained_end=='a' and margin_a is not None:margin_a*=.5
            if retained_end=='b' and margin_b is not None:margin_b*=.5
        retained=retained_end
        force_bisect=b-a>.5*previous
    if b-a<=criterion:
        return report(True,(a+b)/2,(a,b),b-a,criterion,iterations,probes,'resolved')
    return report(False,None,(a,b),b-a,criterion,iterations,probes,'exhausted')


def integrate_material_friction(field,settings,*,_front_uncertainty=None,_local_front_strips=False,
        _eager_directional_controls=False,_variable_endpoint_branches=False):
    """Friction-only work at the endpoint of the field's last declared segment.

    Its tangent holds earlier physical segments fixed. Partitions include contact
    fronts at every saved pose, with axial roots resolved for each angular query.
    Zero panels additionally need a conservative whole-history open certificate.
    """
    from .native_surface_support import (classify_history_panel,HistoryPanelSupport,
        history_target_ownership_certified,SupportIntervalBudget)
    if type(field) is not NativeSurfaceHistoryField or type(settings) is not NativeSurfaceFrictionIntegrationSettings:
        raise TypeError('exact material history field and integration settings required')
    if (_front_uncertainty is None and not _local_front_strips and not _eager_directional_controls
            and not _variable_endpoint_branches
            and field.patch.pairs[0].activation_policy in ('transverse_events','transverse_event_sequence')
            and _captured_fronts_move_between_poses(field)):
        # Deterministic preflight route selection, not a failure fallback:
        # certified moving captured fronts make whole-panel classification
        # inapplicable, so the call routes to the explicit moving-front
        # machinery before any support interval or query is spent. Fields
        # without detected moving fronts keep the unchanged legacy route.
        # Both public activation policies keep their own identities and
        # their own point-level event-admission contracts: the field's
        # query still admits first-touch/release events and rejects
        # temporal multi-event segments for transverse_events exactly as
        # before, and the never-contacted shortcut below stays gated on
        # the actual policy. No policy remapping or identity change.
        return integrate_material_friction(field,settings,
            _front_uncertainty=_DECLARED_PUBLIC_AXIAL_FRONT_UNCERTAINTY,
            _local_front_strips=True,_variable_endpoint_branches=True)
    support_budget=SupportIntervalBudget(settings.maximum_support_intervals)
    if not history_target_ownership_certified(field,budget=support_budget):
        raise ValueError('native friction integration requires swept whole-patch target ownership')
    field._bind_whole_history_target_ownership()
    poses=[field.initial]
    orientation_intervals=0
    for segment in field.segments:
        orientation_intervals+=field.patch.rule.certify_orientation_path(poses[-1],segment.translations,
            segment.angular,segment.shell_increment,maximum_intervals=field.maximum_intervals)
        poses.append(material_path_pose(poses[-1],segment.translations,segment.angular,segment.shell_increment,1.))
    for pose in poses:straight_patch_s3_gap(field.patch.rule,field.patch.target,pose)
    rule=field.patch.rule;lo,hi=rule.definition['xi_range'];alo,ahi=rule.definition['angle_range']
    reference=rule.samples[0].station.reference
    if _variable_endpoint_branches and (not _local_front_strips or _front_uncertainty is None):
        raise ValueError('variable endpoint branches require the explicit local-front route')
    panels=[];coefficients={};endpoint_partitions={}

    crossing_cut_breaks=0;structural_axis_panels=0;directional_control_estimates=0

    def front_panels(cell,left,right,lower,upper):
        nonlocal crossing_cut_breaks
        breaks={lower,upper}
        if _variable_endpoint_branches:
            crossing_cuts=_captured_front_crossing_angles(coefficients[cell],left,right,lower,upper)
            crossing_cut_breaks+=len(crossing_cuts)
            breaks.update(crossing_cuts)
        # Variable endpoint branches certify clamped stationary extrema across
        # the full angular band. Endpoint guards remain in that complete proof;
        # they need not each become a separate quadrature rectangle. Genuine
        # captured-pose contact fronts below still partition the integral.
        for offset,slope,a,b in coefficients[cell]:
            for front in _breakpoints(offset+slope*left,offset+slope*right,a,b,lower,upper):
                if lower<front<upper:
                    # Keep a floating-point guard band around a computed front.
                    # The law is unchanged and the band is integrated, not dropped.
                    # An open neighboring panel then has a resolved positive margin.
                    width=np.sqrt(np.finfo(float).eps)*max(1.,abs(front))
                    breaks.update((max(lower,front-width),min(upper,front+width)))
        angles=sorted(breaks)
        return [(cell,left,right,a,b) for a,b in zip(angles[:-1],angles[1:])]

    for cell,left,right in ((0,max(-1.,lo),min(0.,hi)),(1,max(0.,lo),min(1.,hi))):
        if right<=left:continue
        coefficients[cell]=[_coefficients(field,pose,cell) for pose in poses]
        if _variable_endpoint_branches:
            from .native_surface_front_boundary import stationary_endpoint_angular_partition
            endpoint_partitions[cell]=stationary_endpoint_angular_partition(field,cell,(alo,ahi),budget=support_budget)
        panels.extend(front_panels(cell,left,right,alo,ahi))
    budget=NativeSurfaceReplayBudget(settings.maximum_point_paths)
    cache={};branch_cache={};margin_cache={};branch_cut_cache={};branch_cut_hints=0;branch_cut_queries=0;front_cache={};front_error={};front_covers={};front_axes={};front_band_cache={};front_prepared={};support_by_panel={};refinement_axes=[];visited=accepted=empty_certificates=0
    # DEFERRED constitutive-cut schedule: branch_cut_parents holds the exact
    # parent panels on which a constitutive cut attempt has been made. An
    # attempt happens at most once per exact parent, only after the global
    # L1 error gate selects that parent as the worst leaf, and it enables
    # the per-ordinate locator for that parent alone. Baseline estimates
    # of every other panel never probe a branch boundary.
    angular_branch_parents=set();angular_branch_attempts=0;angular_branch_cuts=0
    branch_cut_parents=set();branch_cut_parent_attempts=0;branch_cut_refusals={}
    # NORMALIZED LOCAL AXIAL INTEGRATION REFINEMENT state: per-parent
    # affine dyadic level on the captured-front window edges, the measured
    # window structure of every estimated parent, and the bounded
    # accounting of every attempt (charged window-child lists, executed
    # child integrations, labelled refusals). No hidden budget: each
    # attempt charges maximum_panels and every child node is a query.
    front_work_density={}
    front_open_certificate_bands={}
    last_error=None;last_error_components=None;global_error_components=None
    open_queries=0;open_endpoint_strip_enclosures=0;failed_quadrature_queries=0
    certified_open_zero_leaves=0;zero_work_admissions=0;positive_gap_zero_leaves=0
    open_certificate_band_reuses=0;positive_gap_cover_reuses=0
    pre_charge_admissions=0;pre_gap_admissions=0;pre_gap_refusals=0;pre_charge_none=0;final_pose_contact_skips=0
    charge_none_final_pose=0;charge_none_checks=0;charge_none_mixed_b=0
    post_certificate_admissions=0;post_charge_admissions=0;post_cover_admissions=0
    segment_cover_cache={};segment_cover_reuses=0
    prepared_segment_cache={};prepared_segment_reuses=0
    whole_prefix_attempted=set()
    positive_gap_covers=[]
    prefix_replay=PrefixReplayField(field) if _variable_endpoint_branches else None
    broad_prepared_cache={}
    size=60+60*60+4
    empty=np.zeros(size)
    gap_padding=512*np.finfo(float).eps*max(1.,rule.definition['radius'],
        *(float(np.max(np.abs(p.positions)))+float(np.max(np.abs(p.position_low)))
          +float(np.max(np.abs(field.patch.target.adapter.current_positions(p.shell_displacement))))
          for p in poses))
    rotational=np.ones(60)
    for first in (3,9,15,18,21,45,51,57):rotational[first:first+3]=1/settings.characteristic_length_m
    scale=np.concatenate((rotational/settings.force_absolute_tolerance,
        (rotational[:,None]*rotational[None,:]).ravel()/settings.tangent_absolute_tolerance,
        np.full(4,1/settings.energy_absolute_tolerance)))

    def scaled_error_components(error):
        weighted=scale*error
        return dict(force=float(np.max(weighted[:60])),tangent=float(np.max(weighted[60:3660])),
            stored=float(weighted[-4]),dissipated=float(weighted[-3]),
            released=float(weighted[-2]),cumulative=float(weighted[-1]))

    def query(xi,angle):
        nonlocal open_queries,failed_quadrature_queries
        cell=0 if xi<0 else 1
        # Only a locally RESOLVED panel permits this shortcut. Every segment
        # was proved monotone, strictly open, or strictly closed. A closed
        # segment contradicts all-positive endpoints; in the other cases this
        # predicate proves the queried point has never contacted.
        if support.status!='resolved':
            raise RuntimeError('native friction query requires resolved local support')
        if (field.patch.pairs[0].activation_policy!='transverse_event_sequence'
                and not support.event_covered
                and all(fsum((offset,slope*xi,a*cos(angle),b*sin(angle)))>gap_padding
               for offset,slope,a,b in coefficients[cell])):
            open_queries+=1
            return empty
        key=(float(xi),float(angle))
        if key not in cache:
            if len(cache)+failed_quadrature_queries>=settings.maximum_queries or (len(cache)+1)*size*8+getattr(field,'_query_prefix_reserved_bytes',0)>settings.maximum_cache_bytes:
                raise NativeFrictionIntegrationLimit('native friction material-query/cache cap exhausted',
                    dict(queries=len(cache),cache_bytes=len(cache)*size*8,point_paths=budget.point_paths,
                         panels=visited,accepted_panels=accepted,current_panel=panel,
                         previous_panel_scaled_error=last_error,support_intervals=support_budget.intervals,
                         previous_panel_error_components=last_error_components,
                         previous_global_error_components=global_error_components,
                         directional_control_panels=len(front_axes),refined_panels=len(refinement_axes),
                         pre_charge_admissions=pre_charge_admissions,
                         pre_gap_admissions=pre_gap_admissions,
                         pre_gap_refusals=pre_gap_refusals,
                         pre_charge_none=pre_charge_none,
                         reused_open_certificate_bands=open_certificate_band_reuses,
                         reused_positive_gap_covers=positive_gap_cover_reuses,
                         charge_none_final_pose=charge_none_final_pose,
                         charge_none_checks=charge_none_checks,
                         charge_none_mixed_b=charge_none_mixed_b,
                         post_certificate_admissions=post_certificate_admissions,
                         post_charge_admissions=post_charge_admissions,
                         post_cover_admissions=post_cover_admissions,
                         resolved_branch_cuts=sum(len(value) for value in branch_cut_cache.values()),
                         branch_cut_hints=branch_cut_hints,
                         branch_cut_queries=branch_cut_queries,
                         branch_cut_parent_attempts=branch_cut_parent_attempts,
                         branch_cut_parent_refusals=dict(branch_cut_refusals),
                         angular_branch_attempts=angular_branch_attempts,
                         angular_branch_cuts=angular_branch_cuts,
                         crossing_cut_breaks=crossing_cut_breaks,
                         structural_axis_panels=structural_axis_panels,
                         directional_control_estimates=directional_control_estimates,
                         final_pose_contact_skips=final_pose_contact_skips))
            try:value=field.query(*key,budget=budget)
            except (ValueError,MemoryError) as error:
                # Preserve the typed non-success while binding it to the
                # material coordinate and budgets of this candidate.
                if not hasattr(error,'integration_diagnostics'):
                    error.integration_diagnostics=dict(xi=key[0],angle=key[1],panel=panel,
                        queries=len(cache),point_paths=budget.point_paths,panels=visited,
                        support_intervals=support_budget.intervals,
                        open_endpoint_strip_enclosure_visits=open_endpoint_strip_enclosures,
                        captured_gaps=[fsum((offset,slope*xi,a*cos(angle),b*sin(angle)))
                            for offset,slope,a,b in coefficients[cell]])
                if (_variable_endpoint_branches and type(error) is MemoryError
                        and str(error)=='native material event sequence interval cap exhausted'):
                    # Nodes have no committed history identity. Refine their
                    # complete panel instead of accepting this uncertain query.
                    # Count rejected work against the same query/path budgets.
                    failed_quadrature_queries+=1
                    raise _MaterialQuadratureSubdivision(*key) from error
                raise
            point=value.last_contribution
            packed=np.zeros(size)
            if point is not None:
                packed[:60]=point.residual-point.normal_residual
                packed[60:3660]=(point.spatial_tangent()-point.normal_spatial_tangent()).ravel()
                trial=point.trial
                packed[-4:]=(trial.stored_energy_density,trial.dissipated_increment_density,
                    trial.released_storage_density,value.history.cumulative_dissipation_density)
            if not np.all(np.isfinite(packed)):raise ValueError('nonfinite replayed friction work')
            cache[key]=packed
            if _variable_endpoint_branches and point is not None:
                branch=getattr(point.trial,'branch',None)
                if branch in ('stick','slip'):
                    # Read-only locator exposure of the genuine final trial:
                    # the constitutive branch bit and the scalar Coulomb
                    # yield margin. The replay, its admission, and the packed
                    # contribution stay untouched.
                    pressure=getattr(point,'normal_pressure_Pa',None)
                    margin=None
                    if pressure is not None and float(pressure)>0.:
                        state=getattr(point.trial,'candidate_state',None)
                        elastic=getattr(state,'elastic_slip',None) if state is not None else None
                        if elastic is not None:
                            magnitude=float(np.linalg.norm(np.asarray(elastic)
                                +np.asarray(point.trial.plastic_slip_increment)))
                            pair=field.patch.pairs[0]
                            margin=(pair.friction_coefficient*float(pressure)
                                -pair.tangential_stiffness_Pa_per_m*magnitude)
                    branch_cache[key]=branch;margin_cache[key]=margin
        return cache[key]


    def _final_trial_state(xi,angle):
        """Genuine replay branch plus read-only Coulomb trial yield margin."""
        query(xi,angle)
        key=(float(xi),float(angle))
        if key not in branch_cache:
            # The policy-gated never-contacted shortcut returns the
            # shared empty contribution without a constitutive trial;
            # such a point has no stick/slip branch interface.
            return None,None
        return branch_cache[key],margin_cache[key]

    def _branch_boundary_cut(cell,angle,x0,x1):
        """Bounded per-ordinate stick/slip yield-margin cut inside one edge.

        The final trial's Coulomb yield margin is a read-only function of
        the genuine replay; a branch transition between the two interior
        probe points is refined by the safeguarded locator. Only a
        resolved outcome (final bracket within the algorithmic subdivision
        width) carries a coordinate; an exhausted or unbracketed edge
        returns a labelled refinement hint, never a wide midpoint. A
        final-pose-open edge has no branch interface and is refused
        before any query.
        """
        width=x1-x0
        if not width>0.:return None
        final=coefficients[cell][-1]
        middle=(x0+x1)/2
        gap_middle=fsum((final[0],final[1]*middle,final[2]*cos(angle),final[3]*sin(angle)))
        if gap_middle>=0.:return None
        a,b=x0+1e-3*width,x1-1e-3*width
        if not x0<a<b<x1:return None
        return _safeguarded_branch_refinement(lambda x:_final_trial_state(x,angle),a,b)

    def _angular_branch_hint(parent):
        cell,left,right,lower,upper=parent
        width=upper-lower
        a=lower+1e-3*width;b=upper-1e-3*width
        if not lower<a<b<upper:return None
        def state(theta):
            xi=_captured_front_contact_probe(coefficients[cell],left,right,theta)
            if xi is None:return None,None
            return _final_trial_state(xi,theta)
        return _safeguarded_branch_refinement(state,a,b)


    def _resolved_yield_margin_cuts(cell,angle,edges):
        """Per-ordinate resolved stick/slip cut coordinates for one edge set.

        A cut is a partition refinement only: the unchanged low/high
        estimator still charges and gates every leaf, the complete domain
        stays covered, and no acceptance is taken from the branch bit
        alone. Stationary uncertainty strips keep their enclosure
        accounting untouched. Results are cached per (cell,angle,edges): a
        cut coordinate is reused only for identical angle and edge support,
        never across differing angular ordinates or axial subdivisions. Unresolved outcomes are counted as labelled
        refinement hints; every probe query is spent through the shared
        cache under the unchanged caps.
        """
        nonlocal branch_cut_hints,branch_cut_queries
        front_key=(cell,float(angle))
        key=(*front_key,tuple(edges))
        if key in branch_cut_cache:return branch_cut_cache[key]
        strips=front_cache[front_key][1] if front_key in front_cache else ()
        before=len(cache)+failed_quadrature_queries
        cuts=[]
        for x0,x1 in zip(edges[:-1],edges[1:]):
            if any(a<=x0 and x1<=b for a,b in strips):continue
            outcome=_branch_boundary_cut(cell,angle,x0,x1)
            if outcome is None:continue
            if outcome['resolved']:cuts.append(outcome['coordinate'])
            else:branch_cut_hints+=1
        branch_cut_cache[key]=tuple(cuts)
        branch_cut_queries+=len(cache)+failed_quadrature_queries-before
        return branch_cut_cache[key]

    # Private variable-front candidate: genuine Gauss3 low estimate plus
    # the positive 12-point degree-seven square cubature as the accepted
    # high rule (21 rather than 25 material samples per mapped
    # rectangle). The tensor Gauss3/Gauss4 pair remains for the low
    # estimate and the tensor directional controls. The public legacy
    # pair stays Gauss4/8.
    gauss=(leggauss(3),leggauss(4)) if _variable_endpoint_branches else (leggauss(4),leggauss(8))
    cubature_angular_groups=_square_cubature_degree7_angular_groups() \
        if _variable_endpoint_branches else None
    def _strip_safety_density(panel):
        # Whole-history work density for strips without a certified final-
        # segment turning enclosure. Computed lazily: bands whose strips are
        # genuinely replayed never pay for these envelopes.
        if panel in front_work_density:return front_work_density[panel]
        from .native_surface_front import stationary_front_angular_envelopes
        from .native_surface_front_work import history_front_strip_work_bound
        cell,left,right,lower,upper=panel
        covers=front_covers[panel];prepared=front_prepared[panel]
        envelopes=tuple(stationary_front_angular_envelopes(field,cover,
            maximum_axial_uncertainty=_front_uncertainty,budget=support_budget,
            prepared=item,_include_endpoints=True) for cover,item in zip(covers,prepared))
        owner_index=next(index for index,cover in enumerate(covers) if len(cover))
        density=np.zeros(4)
        for a,b in envelopes[owner_index][:len(covers[owner_index])]:
            a,b=max(left,a),min(right,b)
            if a<b:
                density=np.maximum(density,history_front_strip_work_bound(field,cell,(a,b),prepared,envelopes))
        front_work_density[panel]=density
        return density
    def _whole_history_open_certificate(panel):
        """Genuine whole-time/whole-angle open certificate for one leaf panel.

        Complete stationary angular envelopes (every certified branch and
        every time-component endpoint of every segment) plus each extremum's
        strict B sign decide the contact side. A panel beyond every padded
        front excursion never contacts at any time or material angle. An
        exact-zero quadrature estimate alone never certifies this; envelopes
        are computed lazily per angular band against the shared support
        budget, and a budget failure is cached as uncertified.
        """
        nonlocal open_certificate_band_reuses
        from .native_surface_front import stationary_front_angular_envelopes,_front_components
        cell,left,right,lower,upper=panel
        band=(cell,lower,upper)
        if band not in front_open_certificate_bands:
            covers=front_covers[panel];prepared=front_prepared[panel]
            if len(covers)!=len(prepared):
                raise ValueError('whole-history open certificate requires complete prepared covers')
            # The checks bind the certified branch/component structure. A
            # containing band with the identical structure produced
            # strictly wider excursion envelopes (its angular half-width
            # term) and identical B signs (B carries no angular term at
            # identical time brackets), so its completed checks are
            # conservative on this contained band: they admit only what
            # this band's own proof would admit, and every refusal stays
            # a refusal. A budget failure (None) is never reused.
            structure=tuple((cover.branches,cover.front_components,cover.branch_bounds,
                cover.variable_branches,cover.xi_bounds) for cover in covers)
            reusable=None
            for container,stored in front_open_certificate_bands.items():
                if (stored is not None and container[0]==cell
                        and container[1]<=lower<upper<=container[2]
                        and stored[1]==structure):
                    reusable=stored;break
            if reusable is not None:
                front_open_certificate_bands[band]=reusable
                open_certificate_band_reuses+=1
            else:
                try:
                    segments=[]
                    for cover,band_prepared in zip(covers,prepared):
                        envelopes=stationary_front_angular_envelopes(field,cover,
                            maximum_axial_uncertainty=_front_uncertainty,budget=support_budget,
                            prepared=band_prepared,_include_endpoints=True)
                        components=_front_components(cover)
                        if len(envelopes)!=len(cover.branches)+2*len(components):
                            raise ValueError('whole-history open certificate envelope count mismatch')
                        checks=[]
                        for position,entry in enumerate(band_prepared.branches):
                            checks.append((*entry[2][:2],*envelopes[position]))
                        for position,entry in enumerate(band_prepared.endpoints):
                            checks.append((*entry[2][:2],*envelopes[len(cover.branches)+position]))
                        segments.append(tuple(checks))
                except MemoryError:
                    # The shared budget cannot certify this band; do not retry it
                    # for the next zero leaf of the same band.
                    front_open_certificate_bands[band]=None
                    raise
                front_open_certificate_bands[band]=(tuple(segments),structure)
        stored=front_open_certificate_bands[band]
        if stored is None:return False
        checks_by_segment=stored[0]
        for checks in checks_by_segment:
            for bl,bu,elow,ehigh in checks:
                if bl>0:
                    if left<ehigh:return False
                elif bu<0:
                    if right>elow:return False
                else:return False
        return True
    def _whole_history_open_split(panel):
        """Certified axial split at the whole-history front excursion.

        Uses the same complete per-band envelope checks as the open
        certificate. When every certified branch and endpoint extremum has
        the same contact side, the material beyond the padded excursion
        never contacts at any time or material angle: that child certifies
        open before any query, and only the excursion child spends
        quadrature. Mixed contact sides keep the unsplit panel; no region
        is dropped and no contact work is zero-filled.
        """
        cell,left,right,lower,upper=panel
        stored=front_open_certificate_bands.get((cell,lower,upper))
        if stored is None:return None
        checks_by_segment=stored[0]
        above=None;below=None
        for checks in checks_by_segment:
            for bl,bu,elow,ehigh in checks:
                if bl>0:above=ehigh if above is None else max(above,ehigh)
                elif bu<0:below=elow if below is None else min(below,elow)
                else:return None
        if (above is None)==(below is None):return None
        cut=min(max(above if above is not None else below,left),right)
        if not left<cut<right:return None
        return ((cell,left,cut,lower,upper),(cell,cut,right,lower,upper))
    def _final_pose_minimum(panel):
        """Minimum final-pose gap of the last segment's linear model over
        the complete leaf rectangle. Strictly positive proves the final
        pose open; nonpositive proves final-pose contact, so the
        whole-history open certificate (which admits only never-contacting
        panels) and the strictly-positive-gap cover (which requires
        strictly positive gap) are both guaranteed refusals there.
        """
        cell,left,right,lower,upper=panel
        offset,slope,a,b=coefficients[cell][-1]
        minimum=offset+min(slope*left,slope*right)
        amplitude=float(np.hypot(a,b))
        if amplitude>0.:
            theta=float(np.arctan2(-b,-a))
            while theta<lower:theta+=2.*float(np.pi)
            if upper-lower>=2.*float(np.pi) or theta<=upper:minimum-=amplitude
            else:minimum+=min(a*np.cos(lower)+b*np.sin(lower),
                a*np.cos(upper)+b*np.sin(upper))
        return minimum
    def _final_open_zero_work_certificate(panel):
        nonlocal charge_none_final_pose,charge_none_checks,charge_none_mixed_b
        """Certified zero-work admission for one unresolved exact-zero leaf.

        A strictly open final pose over the complete leaf rectangle proves
        the endpoint residual and tangent exactly zero: the frictional
        residual and its tangent exist only under final-pose contact. The
        remaining whole-history work admits a certified enclosure with the
        same depth-to-work algebra as the strip bounds: each segment's
        padded excursion checks bound the leaf material's penetration depth,
        hence its pressure cap, dissipation, and storage. Half the bound
        enters the accepted value and half the certified error, exactly like
        the strip enclosures; a charge that cannot fit the public budget
        simply keeps the refinement path. Returns the charge, or None.
        """
        cell,left,right,lower,upper=panel
        if _final_pose_minimum(panel)<=gap_padding:
            charge_none_final_pose+=1
            return None
        stored=front_open_certificate_bands.get((cell,lower,upper))
        if stored is None:
            charge_none_checks+=1
            return None
        checks_by_segment=stored[0]
        depths=[]
        for checks in checks_by_segment:
            depth=0.
            for bl,bu,elow,ehigh in checks:
                if bl>0:depth=max(depth,ehigh-left)
                elif bu<0:depth=max(depth,right-elow)
                else:
                    charge_none_mixed_b+=1
                    return None
            depths.append(depth)
        density=np.zeros(4)
        for index,depth in enumerate(depths):
            if depth<=0.:continue
            bound=final_segment_strip_work_bound(field,cell,depth,index)
            density[0]=max(density[0],bound[0])
            density[2]+=bound[2]
            density[3]+=bound[3]
            if index==len(depths)-1:density[1]=bound[1]
        measure=rule.definition['radius']*np.linalg.norm(reference.coordinates[cell+1]-reference.coordinates[cell])
        charge=np.zeros(size)
        charge[-4:]=.5*density*(right-left)*(upper-lower)*measure
        return charge
    def _whole_history_positive_gap_certificate(panel):
        """Strictly positive gap over the complete leaf rectangle and history.

        The whole-history front certificate bounds only the quotient
        excursion; a leaf inside a padded excursion may still never contact,
        and an exact-zero estimate alone never proves that. The existing
        segment gap slab enclosure bounds the actual gap from below over
        the leaf rectangle; a complete dyadic time cover whose every slab
        is strictly positive proves the material never contacted at any
        time, so the friction integral is exactly zero without any no-slip
        argument. The subdivision shares the support budget; an exhausted
        budget, an unresolved slab, or a possible touch refuses and keeps
        the honest refinement path. This certifies separation only, never
        a contact history state.
        """
        nonlocal positive_gap_cover_reuses
        from .native_surface_support import segment_gap_slab_enclosure
        cell,left,right,lower,upper=panel
        # A completed cover binds this immutable field evaluation and the
        # exact rectangle; strictly positive gap over a containing
        # rectangle proves the same over any contained one. Failures are
        # never cached.
        if any(c==cell and l<=left and right<=r and a<=lower and upper<=b
                for c,l,r,a,b in positive_gap_covers):
            positive_gap_cover_reuses+=1
            return True
        for index in range(len(field.segments)):
            enclose=segment_gap_slab_enclosure(field,cell,(left,right),(lower,upper),index)
            pending=[(0.,1.)];visited=0
            while pending:
                if visited>=field.maximum_intervals:return False
                if not support_budget.consume():
                    raise MemoryError('native friction positive-gap cover interval cap exhausted')
                low,high=pending.pop();visited+=1
                bounds=enclose(low,high)
                if bounds.gap_lower>0.:continue
                if bounds.gap_upper<0.:return False
                middle=(low+high)/2
                if not low<middle<high:return False
                pending.extend(((middle,high),(low,middle)))
        positive_gap_covers.append((cell,left,right,lower,upper))
        return True
    def estimate(panel,order):
        cell,left,right,lower,upper=panel
        axial_order,angular_order=(order,order) if isinstance(order,int) else order
        nonlocal open_queries,open_endpoint_strip_enclosures
        result=np.zeros(size)
        if _variable_endpoint_branches and isinstance(order,int) and order==1:
            # Accepted high estimate: sparse 12-point degree-seven square
            # cubature grouped by angular ordinate. wa carries half the
            # ordinate's total node weight, so every wx*wa product and
            # every wa-only strip fallback weight below remain the proper
            # square Jacobian factors of the cubature node weights.
            angular_nodes=cubature_angular_groups
        else:
            points,weights=gauss[angular_order]
            axial_points,axial_weights=gauss[axial_order]
            angular_nodes=tuple((angular,wa,axial_points,axial_weights)
                for angular,wa in zip(points,weights))
        measure=rule.definition['radius']*np.linalg.norm(reference.coordinates[cell+1]-reference.coordinates[cell])
        for angular,wa,axial_points,axial_weights in angular_nodes:
            angle=(lower+upper)/2+(upper-lower)*float(angular)/2
            roots={left,right}
            for offset,slope,a,b in coefficients[cell]:
                if slope:
                    root=-(offset+a*cos(angle)+b*sin(angle))/slope
                    if left<root<right:roots.add(root)
            edges=sorted(roots)
            if _front_uncertainty is not None:
                from .native_surface_front import stationary_front_partition
                key=(cell,float(angle))
                if key not in front_cache:
                    cuts=[];strips=[];prepared_partitions=[];records=[]
                    for index in range(len(field.segments)):
                        try:
                            if _local_front_strips:
                                from .native_surface_front import stationary_front_partition_prepared
                                partition=stationary_front_partition_prepared(field,
                                    front_prepared[panel][index],float(angle),maximum_axial_uncertainty=_front_uncertainty)
                                _,_,stationary,endpoints=partition
                                prepared_partitions.append((front_prepared[panel][index],partition))
                            else:
                                _,_,stationary,endpoints=stationary_front_partition(field,cell,index,float(angle),
                                    maximum_intervals=field.maximum_intervals,maximum_axial_uncertainty=_front_uncertainty,
                                    budget=support_budget,angular_cover=front_covers[panel][index])
                        except MemoryError as error:
                            raise NativeFrictionIntegrationLimit(str(error),dict(panel=panel,panels=visited,
                                queries=len(cache),front_queries=len(front_cache),support_intervals=support_budget.intervals,
                                previous_scaled_error=last_error,axial_refinements=sum(front_axes.values()),
                                angular_refinements=len(front_axes)-sum(front_axes.values()))) from error
                        # Captured-pose coefficient roots already partition endpoint
                        # transitions. Do not add nearly coincident interval edges:
                        # that manufactures unresolved-gap micro-panels. The full
                        # domain remains covered, with the existing adaptive rule.
                        # Stationary strips alone receive enclosure work accounting.
                        for value in (stationary if _local_front_strips else (*stationary,*endpoints)):
                            cuts.extend(value.axial_interval)
                            # The interval already contains the stationary front.
                            # A second nominal cut within that work-controlled
                            # strip can duplicate the actual captured-pose front
                            # to adjacent binary64 coordinates. Captured fronts
                            # still partition endpoint R/K; the strip is intact.
                            if not _variable_endpoint_branches:
                                cuts.append(value.axial_coordinate)
                            elif _front_uncertainty is not None:
                                # The certified stationary coordinate is
                                # this strip's whole-history work edge:
                                # material beyond it never touched during
                                # the segment. Splitting there leaves each
                                # side with one trial branch, so the
                                # accepted Gauss3/cubature pair no longer samples
                                # a mixed-branch kink inside one interval.
                                strip_low,strip_high=value.axial_interval
                                coordinate=value.axial_coordinate
                                if strip_low+1e-3*(strip_high-strip_low)<coordinate<strip_high-1e-3*(strip_high-strip_low):
                                    cuts.append(coordinate)
                        strips.extend(value.axial_interval for value in stationary)
                        if _variable_endpoint_branches:
                            band_prepared=front_prepared[panel][index]
                            for position,root in enumerate(stationary):
                                bounds=band_prepared.branches[position][2]
                                records.append((root.axial_interval,root.kind,bounds,index,
                                    contact_side_turning(bounds)))
                    front_cache[key]=(tuple(cuts),tuple(strips),tuple(prepared_partitions),tuple(records))
                if _variable_endpoint_branches:
                    # A stationary coordinate can sit within binary64
                    # rounding of the captured-pose root already in the
                    # edge set. Only genuinely sub-resolution pairs merge:
                    # the threshold is the binary64 resolution floor of
                    # the accepted node placement itself, so any distinct
                    # pair becomes its own queryable micro-interval whose
                    # full measure is integrated and whose error the
                    # Gauss3/cubature estimator charges; a merged pair leaves the
                    # kink within the same uncharged coordinate-rounding
                    # scale the rule already applies to every node.
                    for x in front_cache[key][0]:
                        if left<x<right and not any(abs(x-y)<=8.*np.finfo(float).eps*max(1.,abs(x)) for y in roots):
                            roots.add(x)
                else:
                    roots.update(x for x in front_cache[key][0] if left<x<right)
                edges=sorted(roots)
                if _variable_endpoint_branches:
                    # DEFERRED constitutive-cut schedule: baseline low/high
                    # estimates never probe branch boundaries. The per-ordinate
                    # safeguarded stick/slip locator runs only for an exact
                    # parent whose leaf the global L1 error gate selected,
                    # once per parent. Only resolved coordinates refine the
                    # partition at the actual angle, and the complete domain
                    # plus the unchanged estimator error gates stay decisive;
                    # no acceptance is ever taken from the branch bit alone.
                    if panel in branch_cut_parents:
                        roots.update(x for x in _resolved_yield_margin_cuts(cell,angle,edges)
                            if left<x<right)
                        edges=sorted(roots)
            for x0,x1 in zip(edges[:-1],edges[1:]):
                if _local_front_strips:
                    from .native_surface_front import prepared_front_open_interval
                    if all(prepared_front_open_interval(prepared,partition,x0,x1)
                           for prepared,partition in front_cache[key][2]):
                        open_queries+=len(axial_points)
                        continue
                in_strip=_front_uncertainty is not None and any(a<=x0 and x1<=b for a,b in front_cache[key][1])
                if in_strip and not _variable_endpoint_branches:
                    # Integral enclosure replaces unresolved point evaluations;
                    # no zero point history or never-contacted claim is created.
                    continue
                record=None
                if in_strip and _variable_endpoint_branches:
                    for item in front_cache[key][3]:
                        if item[0][0]<=x0 and x1<=item[0][1]:
                            if item[4] and item[3]==len(field.segments)-1:
                                record=item;break
                            if record is None:record=item
                if in_strip and _variable_endpoint_branches and record is not None and record[4] \
                        and record[3]==len(field.segments)-1:
                    # Certified contact-side turning point of the final segment:
                    # material inside this strip penetrates during that segment
                    # only across the certified depth below, so the final-segment
                    # work bound certifies every genuine replay and encloses any
                    # replay that requests subdivision.
                    prepared_band,partition=front_cache[key][2][record[3]]
                    depth=_final_segment_depth(prepared_band,partition,x0,x1)
                    bound=final_segment_strip_work_bound(field,cell,depth,record[3])
                    block=np.zeros(size);genuine=True
                    for axial,wx in zip(axial_points,axial_weights):
                        xi=_interior_material_coordinate(x0,x1,axial,allow_rounding=_variable_endpoint_branches)
                        if not x0<xi<x1:
                            error=ValueError('native friction axial resolution exhausted')
                            error.integration_diagnostics=dict(panel=panel,angle=angle,interval=(x0,x1),
                                xi=xi,queries=len(cache),point_paths=budget.point_paths,
                                panels=visited,support_intervals=support_budget.intervals)
                            raise error
                        try:point=query(xi,angle)
                        except _MaterialQuadratureSubdivision:genuine=False;break
                        if (np.any(point[-4:]<0) or point[-4]>bound[0] or point[-3]>bound[1]
                                or point[-2]>2.*bound[2]):
                            raise ValueError('replayed strip work exceeds certified final-segment bound')
                        block+=point*(float(wx*wa)*(x1-x0)*(upper-lower)*measure/4)
                    if genuine:
                        result+=block
                        continue
                    # Both remaining fallbacks zero-fill endpoint residual and
                    # tangent. That is certified only for a strictly open final
                    # micro-interval; a closed strip must genuine replay or
                    # refuse, never zero-fill.
                    offset,slope,a,b=coefficients[cell][-1]
                    final_lower=min(fsum((offset,slope*x,a*cos(angle),b*sin(angle))) for x in (x0,x1))
                    if final_lower<=gap_padding:
                        refusal=ValueError('closed final-segment strip requires genuine replay')
                        refusal.integration_diagnostics=dict(panel=panel,angle=angle,
                            interval=(x0,x1),final_gap_lower=final_lower,queries=len(cache),
                            point_paths=budget.point_paths,panels=visited,
                            support_intervals=support_budget.intervals)
                        raise refusal
                    # Genuine replay requested subdivision. Enclose the strip
                    # value by exact prefix replay at the rule nodes plus the
                    # certified final-segment increments; charge the actual
                    # enclosure half-widths as this strip's error.
                    fallback=np.zeros(size);half=np.zeros(size);enclosed=True
                    for axial,wx in zip(axial_points,axial_weights):
                        xi=_interior_material_coordinate(x0,x1,axial,allow_rounding=_variable_endpoint_branches)
                        weight=float(wx*wa)*(x1-x0)*(upper-lower)*measure/4
                        try:
                            stored,cumulative=prefix_replay.boundary_state(xi,angle,budget)
                        except MemoryError as prefix_error:
                            if str(prefix_error)!='native material event sequence interval cap exhausted':raise
                            enclosed=False;break
                        node=np.zeros(size)
                        node[-4]=.5*bound[0];node[-3]=.5*bound[1]
                        node[-2]=.5*(stored+bound[2]);node[-1]=cumulative+.5*bound[3]
                        fallback+=node*weight
                        charge=np.zeros(size)
                        charge[-4]=.5*bound[0];charge[-3]=.5*bound[1]
                        charge[-2]=.5*(stored+bound[2]);charge[-1]=.5*bound[3]
                        half+=charge*weight
                    if enclosed:
                        result+=fallback
                        front_error[panel]=front_error.get(panel,np.zeros(size))+half
                        open_endpoint_strip_enclosures+=1
                        continue
                    # The prefix itself requested subdivision: keep only the
                    # lazy whole-history enclosure for this micro-interval.
                    density=_strip_safety_density(panel)
                    midpoint=np.zeros(size);midpoint[-4:]=.5*density
                    weight=(x1-x0)*float(wa)*(upper-lower)*measure/2
                    result+=midpoint*weight
                    front_error[panel]=front_error.get(panel,np.zeros(size))+midpoint*weight
                    open_endpoint_strip_enclosures+=1
                    continue
                if in_strip and _variable_endpoint_branches:
                    offset,slope,a,b=coefficients[cell][-1]
                    final_lower=min(fsum((offset,slope*x,a*cos(angle),b*sin(angle))) for x in (x0,x1))
                    if final_lower>gap_padding:
                        # A strictly open final surface has exactly zero endpoint
                        # R/K/storage, independently of any earlier tiny pulse.
                        # Its history work remains an interval enclosure, never
                        # a reset/zero point state. Positive spatial quadrature of
                        # this midpoint is bounded by the same enclosure width.
                        density=_strip_safety_density(panel)
                        midpoint=np.zeros(size);midpoint[-3:]=.5*density[-3:]
                        weight=(x1-x0)*float(wa)*(upper-lower)*measure/2
                        result+=midpoint*weight
                        front_error[panel]=front_error.get(panel,np.zeros(size))+midpoint*weight
                        open_endpoint_strip_enclosures+=1
                        continue
                for axial,wx in zip(axial_points,axial_weights):
                    xi=_interior_material_coordinate(x0,x1,axial,allow_rounding=_variable_endpoint_branches)
                    if not x0<xi<x1:
                        error=ValueError('native friction axial resolution exhausted')
                        error.integration_diagnostics=dict(panel=panel,angle=angle,interval=(x0,x1),
                            xi=xi,queries=len(cache),point_paths=budget.point_paths,
                            panels=visited,support_intervals=support_budget.intervals)
                        raise error
                    point=query(xi,angle)
                    if in_strip and _variable_endpoint_branches:
                        density=_strip_safety_density(panel)
                        if np.any(point[-4:]<0) or np.any(point[-4:]>density):
                            raise ValueError('replayed strip work exceeds complete-history density bound')
                    result+=point*(float(wx*wa)*(x1-x0)*(upper-lower)*measure/4)
        if panel in front_error and not _variable_endpoint_branches:result+=front_error[panel]
        return result

    def subdivide(parent):
        cell,left,right,lower,upper=parent
        if front_axes.get(parent,(right-left)/(hi-lo)>=(upper-lower)/(ahi-alo)):
            middle=(left+right)/2
            if middle in (left,right):raise ValueError('native friction axial panel resolution exhausted')
            return (*front_panels(cell,left,middle,lower,upper),
                    *front_panels(cell,middle,right,lower,upper))
        middle=(lower+upper)/2
        if middle in (lower,upper):raise ValueError('native friction angular panel resolution exhausted')
        return ((cell,left,right,lower,middle),(cell,left,right,middle,upper))

    def directional_controls(parent,high):
        nonlocal structural_axis_panels,directional_control_estimates
        if _variable_endpoint_branches:
            cell,left,right,lower,upper=parent
            signatures={_captured_front_structure(coefficients[cell],left,right,
                (lower+upper)/2+(upper-lower)*float(node)/2)
                for node,_,_,_ in cubature_angular_groups}
            if len(signatures)>1 or _captured_front_angular_motion_dominates(
                    coefficients[cell],left,right,lower,upper):
                front_axes[parent]=False;structural_axis_panels+=1
                return
        directional_control_estimates+=2
        # These estimates select an axis; they do not enter the accepted
        # high/low integral or its error. Local-front candidates can therefore
        # defer them until this parent actually needs refinement. The eager
        # switch is retained solely as a bounded parity oracle.
        axial_control=estimate(parent,(0,1));angular_control=estimate(parent,(1,0))
        axial=float(np.max(scale*np.abs(high-axial_control)))
        angular=float(np.max(scale*np.abs(high-angular_control)))
        # A zero/zero tie carries no directional signal. Spend it on the same
        # aspect-ratio axis as panels without directional controls, instead
        # of always consuming the split axially.
        if axial==0. and angular==0.:
            front_axes[parent]=(parent[2]-parent[1])/(hi-lo)>=(parent[4]-parent[3])/(ahi-alo)
        else:front_axes[parent]=axial>=angular

    unresolved_support_panels=0;support_panels_visited=0
    def subdivide_support(parent):
        # Support enclosures cover complete rectangles analytically. Pressure
        # fronts belong to quadrature partitioning, not this geometric proof;
        # inserting their guard strips here needlessly fragments child proofs.
        cell,left,right,lower,upper=parent
        if (right-left)/(hi-lo)>=(upper-lower)/(ahi-alo):
            middle=(left+right)/2
            if middle in (left,right):raise ValueError('native friction axial support resolution exhausted')
            return ((cell,left,middle,lower,upper),(cell,middle,right,lower,upper))
        middle=(lower+upper)/2
        if middle in (lower,upper):raise ValueError('native friction angular support resolution exhausted')
        return ((cell,left,right,lower,middle),(cell,left,right,middle,upper))

    def resolve_support(parent):
        # Proof subdivisions need not become quadrature panels. Complete child
        # coverage certifies their union; the parent's independent integration
        # error estimate still decides whether its quadrature must subdivide.
        nonlocal unresolved_support_panels,support_panels_visited,segment_cover_reuses,prepared_segment_reuses
        from anyfesolver._ge_beam3_p5_seeded.core import sha
        if sha(field._definition())!=field.identity:
            raise ValueError('native material history field changed')
        if _front_uncertainty is not None:
            from .native_surface_front import stationary_front_angular_cover
            from .native_surface_support import _minimum
            cell,left,right,lower,upper=parent
            all_endpoints_open=True
            for pose in poses:
                gap,padding=_minimum(pose,field.patch.target,reference.nodal_triads[0],
                    rule.definition['radius'],cell,(left,right),(lower,upper))
                all_endpoints_open=all_endpoints_open and gap>padding
            if not _local_front_strips and not all_endpoints_open:
                raise ValueError('moving-front work slice requires open captured endpoints')
            # Angular proof geometry is independent of an axial child panel.
            # Reuse only identical bands within this immutable field evaluation.
            band=(cell,lower,upper)
            if _local_front_strips and band in front_band_cache:
                covers,envelopes,prepared=front_band_cache[band]
            else:
                domain=(max(rule.definition['xi_range'][0],cell-1),min(rule.definition['xi_range'][1],cell))
                completed=[]
                for index in range(len(field.segments)):
                    # A complete parent-band certificate remains conservative
                    # on any contained band. Retain successful prefix segments
                    # even if a later segment requests angular subdivision.
                    cover=next((replace(value,angle_bounds=(lower,upper))
                        for (c,i,a,b),value in segment_cover_cache.items()
                        if c==cell and i==index and a<=lower<upper<=b),None)
                    if cover is None:
                        requested=(lower,upper)
                        if _variable_endpoint_branches and (cell,index) not in whole_prefix_attempted:
                            whole_prefix_attempted.add((cell,index))
                            requested=rule.definition['angle_range']
                        try:
                            cover=stationary_front_angular_cover(field,cell,index,requested,budget=support_budget,
                                xi_bounds=domain if _local_front_strips else None,
                                endpoint_partition=endpoint_partitions.get(cell),
                                _variable_endpoint_branches=_variable_endpoint_branches,
                                _allow_angular_subdivision=_variable_endpoint_branches)
                        except StationaryAngularSubdivision:
                            if requested==(lower,upper):raise
                            cover=stationary_front_angular_cover(field,cell,index,(lower,upper),budget=support_budget,
                                xi_bounds=domain if _local_front_strips else None,
                                endpoint_partition=endpoint_partitions.get(cell),
                                _variable_endpoint_branches=_variable_endpoint_branches,
                                _allow_angular_subdivision=_variable_endpoint_branches)
                        segment_cover_cache[(cell,index,*cover.angle_bounds)]=cover
                        if cover.angle_bounds!=(lower,upper):
                            cover=replace(cover,angle_bounds=(lower,upper))
                    else:segment_cover_reuses+=1
                    completed.append(cover)
                covers=completed
                envelopes=None
                if _local_front_strips:
                    from .native_surface_front import prepare_stationary_band
                    values=[]
                    for index,cover in enumerate(covers):
                        lo,hi=cover.angle_bounds
                        old=next((value for (c,i,a,b),value in prepared_segment_cache.items()
                            if c==cell and i==index and a<=lo<hi<=b),None)
                        if old is not None:
                            values.append(replace(old,cover=cover));prepared_segment_reuses+=1
                            continue
                        if not _variable_endpoint_branches:
                            # Prepare a completed larger prefix cover once; its
                            # whole-band bounds are conservative on every child.
                            broad=next((value for (c,i,a,b),value in segment_cover_cache.items()
                                if c==cell and i==index and a<=lo<hi<=b),cover)
                            value=prepare_stationary_band(field,broad,budget=support_budget)
                            prepared_segment_cache[(cell,index,*broad.angle_bounds)]=value
                            values.append(replace(value,cover=cover))
                            continue
                        # A broad cover's prepared ENDPOINT controls are exact at
                        # shared component times and reusable verbatim; its wide
                        # BRANCH bounds are not (the narrow root-interval gate
                        # would fail), so only endpoint controls are reused here.
                        broad=next((value for (c,i,a,b),value in segment_cover_cache.items()
                            if c==cell and i==index and a<=lo<hi<=b),cover)
                        reuse=None
                        if broad.angle_bounds!=cover.angle_bounds:
                            broad_key=(cell,index,*broad.angle_bounds)
                            if broad_key not in broad_prepared_cache:
                                try:broad_prepared_cache[broad_key]=prepare_stationary_band(field,broad,budget=support_budget)
                                except ValueError:broad_prepared_cache[broad_key]=False
                            broad_prepared=broad_prepared_cache[broad_key]
                            if broad_prepared is not False:reuse=broad_prepared.endpoints
                        try:
                            value=prepare_stationary_band(field,cover,budget=support_budget,_reuse_endpoints=reuse)
                        except ValueError:
                            # This band's uniform branch controls do not resolve;
                            # recompute its exact cover and prepare that instead.
                            cover=stationary_front_angular_cover(field,cell,index,(lo,hi),budget=support_budget,
                                xi_bounds=domain,endpoint_partition=endpoint_partitions.get(cell),
                                _variable_endpoint_branches=True,_allow_angular_subdivision=True)
                            segment_cover_cache[(cell,index,*cover.angle_bounds)]=cover
                            covers[index]=cover
                            value=prepare_stationary_band(field,cover,budget=support_budget)
                        prepared_segment_cache[(cell,index,*cover.angle_bounds)]=value
                        values.append(value)
                    covers=tuple(covers)
                    prepared=tuple(values)
                else:prepared=()
            front_prepared[parent]=prepared
            branch_count=sum(map(len,covers))
            front_covers[parent]=covers
            if branch_count:
                if branch_count!=1 or (not _local_front_strips and len(field.segments)!=1):
                    raise ValueError('front strip work requires one certified stationary branch; legacy helper requires one segment')
                owner_index=next(index for index,cover in enumerate(covers) if len(cover))
                if _variable_endpoint_branches:
                    # Genuine strip replay supplies the strip integral value;
                    # only an actual fallback enclosure is charged as strip
                    # error. Whole-history envelopes remain a lazy safety
                    # bound, never a blanket panel charge.
                    front_band_cache[band]=(covers,(),prepared)
                    return HistoryPanelSupport('resolved',parent,'variable stationary branches; genuine strip replay with certified final-segment enclosure')
                if _local_front_strips:
                    from .native_surface_front import stationary_front_angular_envelopes
                    if envelopes is None:
                        envelopes=stationary_front_angular_envelopes(field,covers[owner_index],
                            maximum_axial_uncertainty=_front_uncertainty,budget=support_budget,
                            prepared=prepared[owner_index])
                    front_band_cache[band]=(covers,envelopes,prepared)
                    for a,b in envelopes:
                        a,b=max(left,a),min(right,b)
                        if a>=b:continue
                        for pose in poses:
                            gap,padding=_minimum(pose,field.patch.target,reference.nodal_triads[0],
                                rule.definition['radius'],cell,(a,b),(lower,upper))
                            if gap<=padding:
                                raise ValueError('stationary uncertainty strip has unresolved or closed captured endpoint')
                from .native_surface_front_work import open_front_strip_work_bound
                density=open_front_strip_work_bound(field,cell,_front_uncertainty,
                    segment_index=owner_index if _local_front_strips else None)
                if _local_front_strips and owner_index<len(field.segments)-1:
                    # Every other segment is proved open on this strip. Its
                    # earlier pulse contributes cumulative work, but cannot
                    # generate final-segment dissipation or storage release.
                    density[1:3]=0.
                measure=rule.definition['radius']*np.linalg.norm(reference.coordinates[cell+1]-reference.coordinates[cell])
                error=np.zeros(size)
                error[-4:]=.5*density*_front_uncertainty*(upper-lower)*measure
                front_error[parent]=error
            if _local_front_strips and not branch_count:
                front_band_cache[band]=(covers,(),prepared)
            # Nonzero B, no stationary H, and open endpoint rectangles exclude
            # all crossings. Stationary branches require explicit moving cuts.
            return HistoryPanelSupport('open' if not branch_count and all_endpoints_open else 'resolved',parent,
                'whole-angle stationary branches; explicit moving cuts')
        pending=[parent];all_open=True;last_unresolved=None;event_covered=False
        while pending:
            if support_panels_visited>=settings.maximum_panels:
                raise MemoryError('native friction support panel cap exhausted')
            current=pending.pop();c,l,r,a,b=current;support_panels_visited+=1
            try:
                result=classify_history_panel(field,c,(l,r),(a,b),budget=support_budget)
            except MemoryError as error:
                if last_unresolved is None:
                    # The shared cap can exhaust inside a region's own
                    # preparation, before any panel classification exists.
                    # That region never obtained a certificate; report it
                    # as the truthful preparation context instead of None.
                    last_unresolved=dict(region=(c,*map(float,(l,r)),*map(float,(a,b))),
                        reason='support interval cap exhausted during panel preparation')
                raise NativeFrictionIntegrationLimit(str(error),dict(
                    queries=len(cache),support_intervals=support_budget.intervals,
                    support_panels_visited=support_panels_visited,support_region=current,
                    quadrature_panel=parent,last_unresolved=last_unresolved)) from error
            if result.status=='unresolved':
                unresolved_support_panels+=1
                last_unresolved=dict(region=result.region,reason=result.reason)
                pending.extend(subdivide_support(result.region))
            else:
                all_open=all_open and result.status=='open'
                event_covered=event_covered or result.event_covered
        return HistoryPanelSupport('open' if all_open else 'resolved',parent,
            'complete union of local proofs',event_covered)
    total=np.zeros(size);errors=np.zeros(size)
    leaves=[];genuine_quadrature_leaves=set()
    zero_leaf_refinements=0;error_leaf_refinements=0
    while panels or leaves:
        if not panels:
            # Sum absolute componentwise estimates over the complete patch.
            # Area-proportional local budgets unnecessarily over-resolve tiny
            # front panels; the public absolute error limits are global.
            total=sum((leaf[1] for leaf in leaves),np.zeros(size))
            errors=sum((leaf[2] for leaf in leaves),np.zeros(size))
            global_error_components=scaled_error_components(errors)
            if all(leaf[3] for leaf in leaves) and np.max(scale*errors)<=1.:
                accepted+=len(leaves)
                break
            error_led=np.max(scale*errors)>1.
            if error_led:
                # Measured error leads refinement. An unresolved zero leaf no
                # longer preempts leaves carrying certified error; it still
                # blocks acceptance, so it is refined before any success.
                index=max(range(len(leaves)),key=lambda i:float(np.max(scale*leaves[i][2])))
                error_leaf_refinements+=1
            else:
                # The certified error already fits the public budget; only
                # unresolved zero leaves still block acceptance.
                index=next(i for i in range(len(leaves)) if not leaves[i][3])
                zero_leaf_refinements+=1
            parent,parent_high,parent_error,parent_safe=leaves.pop(index)
            if (error_led and _variable_endpoint_branches and _front_uncertainty is not None
                    and parent not in angular_branch_parents):
                angular_branch_parents.add(parent);angular_branch_attempts+=1
                panel=parent;support=support_by_panel[parent]
                try:outcome=_angular_branch_hint(parent)
                except _MaterialQuadratureSubdivision:outcome=None
                if outcome is not None and outcome['resolved']:
                    cut=outcome['coordinate'];cell,left,right,lower,upper=parent
                    if lower<cut<upper:
                        angular_branch_cuts+=1
                        panels.extend(front_panels(cell,left,right,lower,cut))
                        panels.extend(front_panels(cell,left,right,cut,upper))
                        continue
            if error_led and _variable_endpoint_branches and _front_uncertainty is not None:
                # DEFERRED constitutive-cut schedule. The global L1 error
                # gate has selected this exact parent as the worst leaf;
                # before any ordinary geometric refinement, attempt the
                # per-ordinate constitutive cuts on this parent ONCE, rebuild
                # its low/high/error on the complete support with the
                # resolved cut coordinates in the partition, and replace the
                # leaf contribution/error atomically. The loop then rechecks
                # the global gate from scratch: acceptance still requires
                # every leaf certified and the complete charged error inside
                # the public budget, never a branch bit alone. A parent is
                # never reprobed: presence in branch_cut_parents is both the
                # enable and the attempt record. Certified (safe/open zero)
                # leaves keep their own error accounting and go straight to
                # geometric refinement; their certified charges are not
                # quadrature differences a cut could reduce.
                if parent_safe and parent not in genuine_quadrature_leaves:
                    branch_cut_refusals['certified_leaf']=branch_cut_refusals.get('certified_leaf',0)+1
                elif parent in branch_cut_parents:
                    branch_cut_refusals['already_attempted']=branch_cut_refusals.get('already_attempted',0)+1
                else:
                    branch_cut_parents.add(parent)
                    branch_cut_parent_attempts+=1
                    # query() reads this outer context, including when
                    # estimate() rebuilds a previously visited leaf.
                    panel=parent;support=support_by_panel[parent]
                    try:
                        rebuild_low=estimate(parent,0);rebuild_high=estimate(parent,1)
                        rebuild_error=np.abs(rebuild_high-rebuild_low)
                    except _MaterialQuadratureSubdivision:
                        # The rebuild requested subdivision at a cut node;
                        # keep the unchanged leaf and let the ordinary
                        # geometric refinement of this same parent proceed.
                        branch_cut_refusals['rebuild_subdivision']=branch_cut_refusals.get('rebuild_subdivision',0)+1
                        leaves.append((parent,parent_high,parent_error,parent_safe))
                        continue
                    if not np.any(rebuild_low) and not np.any(rebuild_high) and np.any(parent_high):
                        # A changed stencil's sampled zero needs the main
                        # zero-history checks; keep the old leaf and refine.
                        branch_cut_refusals['new_sampled_zero']=branch_cut_refusals.get('new_sampled_zero',0)+1
                        leaves.append((parent,parent_high,parent_error,parent_safe))
                        continue
                    if parent in front_error:rebuild_error=rebuild_error+front_error[parent]
                    last_error=float(np.max(scale*rebuild_error))
                    last_error_components=scaled_error_components(rebuild_error)
                    leaves.append((parent,rebuild_high,rebuild_error,parent_safe))
                    continue
            if _local_front_strips and not _eager_directional_controls:
                # query() reads this outer context, including when estimate()
                # uses a previously visited leaf. Reuse its proof and budgets.
                panel=parent;support=support_by_panel[parent]
                directional_controls(parent,parent_high)
            if parent in front_axes:refinement_axes.append((parent,front_axes[parent]))
            panels.extend(subdivide(parent))
        if visited>=settings.maximum_panels:
            # Retain the original typed failure and add bounded context at
            # the decision point; this neither retries nor admits a panel.
            error=MemoryError('native friction integration panel cap exhausted')
            error.integration_diagnostics=dict(stage='quadrature',current_panel=panel,
                panels=visited,accepted_panels=accepted,queries=len(cache),
                failed_quadrature_queries=failed_quadrature_queries,
                point_paths=budget.point_paths,support_intervals=support_budget.intervals,
                cached_angular_front_bands=len(front_band_cache),
                strip_enclosure_panels=len(front_error),
                open_endpoint_strip_enclosures=open_endpoint_strip_enclosures,
                certified_open_zero_leaves=certified_open_zero_leaves,
                zero_leaf_refinements=zero_leaf_refinements,
                error_leaf_refinements=error_leaf_refinements,
                branch_cut_parent_attempts=branch_cut_parent_attempts,
                branch_cut_parent_refusals=dict(branch_cut_refusals),
                previous_panel_scaled_error=last_error,
                previous_panel_error_components=last_error_components,
                previous_global_error_components=global_error_components)
            raise error
        panel=panels.pop();cell,left,right,lower,upper=panel;visited+=1
        from .native_surface_front import StationaryAngularSubdivision
        try:
            support=resolve_support(panel)
        except StationaryAngularSubdivision:
            # This private variable-front request admits nothing and loses no
            # material region. Both children retain the complete original path;
            # all shared interval/panel/query/error budgets still apply.
            middle=(lower+upper)/2
            if not lower<middle<upper:
                raise ValueError('native friction angular panel resolution exhausted')
            panels.extend(((cell,left,right,lower,middle),(cell,left,right,middle,upper)))
            continue
        except (ValueError,MemoryError) as error:
            # Retain the original typed failure and add bounded context at
            # the decision point; this neither retries nor admits the panel.
            if not hasattr(error,'integration_diagnostics'):
                error.integration_diagnostics=dict(stage='support',current_panel=panel,
                    panels=visited,queries=len(cache),point_paths=budget.point_paths,
                    support_intervals=support_budget.intervals,
                    cached_angular_front_bands=len(front_band_cache),
                    previous_panel_error_components=last_error_components,
                    previous_global_error_components=global_error_components)
            raise
        if _local_front_strips:support_by_panel[panel]=support
        if support.status=='open' or field.patch.pairs[0].friction_coefficient==0:
            accepted+=1;empty_certificates+=1
            continue
        if _variable_endpoint_branches:
            # Genuine whole-history open certificate BEFORE any quadrature
            # query: a panel beyond every padded front excursion never
            # contacts, so its friction integral is exactly zero without
            # spending query budget to sample that fact. The same complete
            # envelope proof governs here as after quadrature; a budget
            # failure is already cached as uncertified. A one-sided
            # excursion additionally splits the panel axially so only the
            # excursion child is ever queried. A nonpositive final-pose
            # lower bound cannot certify final-pose separation: the certificate admits
            # only never-contacting panels, so it is a guaranteed refusal
            # there. Skip its envelope expense on such genuine contact
            # leaves; the split below reads only the shared band cache.
            final_contact=_final_pose_minimum(panel)<=0.
            if final_contact:final_pose_contact_skips+=1
            try:pre_certified=False if final_contact else _whole_history_open_certificate(panel)
            except MemoryError:pre_certified=False
            if pre_certified:
                certified_open_zero_leaves+=1
                leaves.append((panel,np.zeros(size),np.zeros(size),True))
                continue
            children=_whole_history_open_split(panel)
            if children is not None:
                panels.extend(children)
                continue
            charge=_final_open_zero_work_certificate(panel)
            if charge is None:pre_charge_none+=1
            if charge is not None and float(np.max(scale*charge))<=.25:
                # The strictly open final pose already proves this leaf's
                # endpoint residual and tangent exactly zero, and the
                # charge is the same certified whole-history enclosure the
                # sampled route admits after an exact-zero estimate.
                # Sampling cannot add information to that certificate;
                # admit without spending query budget on it.
                leaves.append((panel,charge,charge,True))
                zero_work_admissions+=1;pre_charge_admissions+=1
                continue
            if charge is not None:
                # The charge cannot fit the public budget share and the
                # final pose is strictly open. Prove the leaf never
                # contacted over the complete rectangle and history
                # before spending quadrature that could only re-report
                # that fact. A refusal (touch, unresolved slab, budget)
                # keeps the honest refinement path; a final pose that is
                # not strictly open (charge is None) skips this proof and
                # keeps genuine contact leaves on quadrature.
                try:pre_gap=_whole_history_positive_gap_certificate(panel)
                except (MemoryError,ValueError):pre_gap=False;pre_gap_refusals+=1
                if pre_gap:
                    positive_gap_zero_leaves+=1;pre_gap_admissions+=1
                    leaves.append((panel,np.zeros(size),np.zeros(size),True))
                    continue
        try:
            low=estimate(panel,0);high=estimate(panel,1);error=np.abs(high-low)
        except _MaterialQuadratureSubdivision as request:
            if left<request.xi<right:
                panels.extend(((cell,left,request.xi,lower,upper),
                               (cell,request.xi,right,lower,upper)))
            elif lower<request.angle<upper:
                panels.extend(((cell,left,right,lower,request.angle),
                               (cell,left,right,request.angle,upper)))
            else:
                raise ValueError('native friction quadrature panel resolution exhausted') from request
            continue
        if panel in front_error:error+=front_error[panel]
        if _front_uncertainty is not None and (not _local_front_strips or _eager_directional_controls):
            directional_controls(panel,high)
        last_error=float(np.max(scale*error))
        last_error_components=scaled_error_components(error)
        zero=not np.any(high) and not np.any(low)
        if not zero:genuine_quadrature_leaves.add(panel)
        safe_zero=not zero
        if not safe_zero and _variable_endpoint_branches:
            # An exact-zero estimate is sampled evidence, never emptiness.
            # The whole-history envelope proof may certify the leaf open;
            # without it the leaf keeps demanding refinement.
            try:safe_zero=False if final_contact else _whole_history_open_certificate(panel)
            except MemoryError:pass
            if safe_zero:certified_open_zero_leaves+=1;post_certificate_admissions+=1
            else:
                charge=_final_open_zero_work_certificate(panel)
                # The charge is certified error for any magnitude, but a
                # charge that cannot fit the public absolute budget must
                # not be admitted: it would monopolize the error-led
                # refinement without ever converging. The gate admits
                # only charges inside a conservative budget share;
                # refinement shrinks the children's charges, so borderline
                # leaves converge onto this admission path.
                if charge is not None and float(np.max(scale*charge))<=.25:
                    high=high+charge;error=error+charge
                    safe_zero=True;zero_work_admissions+=1;post_charge_admissions+=1
                else:
                    # The quotient envelope is not contact evidence, and a
                    # charge that cannot fit proves nothing about contact.
                    # A complete strictly-positive-gap time cover on the
                    # actual rectangle proves the leaf material never
                    # contacted at any time, bypassing any historical
                    # no-slip question; a refusal keeps the refinement
                    # path. The cover shares the support budget.
                    try:safe_zero=False if final_contact else _whole_history_positive_gap_certificate(panel)
                    except (MemoryError,ValueError):pass
                    if safe_zero:positive_gap_zero_leaves+=1;post_cover_admissions+=1
        if not safe_zero and support_budget.intervals>=settings.maximum_support_intervals:
            raise MemoryError('native friction aggregate support interval cap exhausted')
        leaves.append((panel,high,error,safe_zero))
    # Also covers a final refinement whose children all have exact-zero proofs.
    total=sum((leaf[1] for leaf in leaves),np.zeros(size))
    errors=sum((leaf[2] for leaf in leaves),np.zeros(size))
    return NativeSurfaceFrictionIntegral(_readonly(total[:60]),_readonly(total[60:3660].reshape(60,60)),
        *map(float,total[-4:]),dict(panels=visited,accepted_panels=accepted,unresolved_support_panels=unresolved_support_panels,support_panels_visited=support_panels_visited,queries=len(cache),
            point_paths=budget.point_paths,cache_bytes=len(cache)*size*8+getattr(field,'_query_prefix_reserved_bytes',0),zero_panels_certified=empty_certificates,
            failed_quadrature_queries=failed_quadrature_queries,
            material_query_attempts=len(cache)+failed_quadrature_queries,
            open_material_queries_certified=open_queries,
            open_endpoint_strip_enclosures=open_endpoint_strip_enclosures,
            certified_open_zero_leaves=certified_open_zero_leaves,
            zero_work_admissions=zero_work_admissions,
            positive_gap_zero_leaves=positive_gap_zero_leaves,
            reused_open_certificate_bands=open_certificate_band_reuses,
            reused_positive_gap_covers=positive_gap_cover_reuses,
            final_pose_contact_skips=final_pose_contact_skips,
            resolved_branch_cuts=sum(len(value) for value in branch_cut_cache.values()),
            branch_cut_hints=branch_cut_hints,
            branch_cut_queries=branch_cut_queries,
            branch_cut_parent_attempts=branch_cut_parent_attempts,
            branch_cut_parent_refusals=dict(branch_cut_refusals),
            angular_branch_attempts=angular_branch_attempts,
            angular_branch_cuts=angular_branch_cuts,
            crossing_cut_breaks=crossing_cut_breaks,
            structural_axis_panels=structural_axis_panels,
            directional_control_estimates=directional_control_estimates,
            front_profile='VARIABLE_ENDPOINT_STRIP_DEV_V11_DEFERRED_BRANCH_CUT' if _variable_endpoint_branches else 'LEGACY_FRONT_OR_PANEL_DEV',
            certified_intervals=orientation_intervals+budget.certified_intervals,
            support_intervals=support_budget.intervals,
            cached_angular_front_bands=len(front_band_cache),
            reused_segment_covers=segment_cover_reuses,
            reused_prepared_segment_controls=prepared_segment_reuses,
            directional_control_panels=len(front_axes),refinement_axes=refinement_axes,
            stationary_strip_work_error_j=sum((front_error.get(leaf[0],np.zeros(size))[-4:] for leaf in leaves),np.zeros(4)).tolist(),
            maximum_scaled_error_estimate=float(np.max(scale*errors)),estimator='GAUSS3_CUBATURE7_DIFFERENCE' if _variable_endpoint_branches else 'GAUSS_4_8_DIFFERENCE'))
