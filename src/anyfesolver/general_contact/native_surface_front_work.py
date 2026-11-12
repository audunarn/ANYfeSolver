"""Work density enclosures near a contact-producing stationary front.

The released-origin specialization requires open captured endpoints. The complete-
history route instead takes every certified component's extrema and retains stored
work. These developmental bounds concern an integral, never a point history.
"""
import numpy as np
from .native_surface_path import material_path_pose
from .native_surface_support import _base


def _segment_motion_bounds(field,cell,segment_index):
    if type(segment_index) is not int or not 0<=segment_index<len(field.segments):
        raise ValueError('strip work enclosure requires a valid explicit segment index')
    # Every segment's start includes its actual prefix motion, including shell
    # displacement and finite rotations. Frame/slip bounds cover the full path.
    start=field.initial
    for prefix in field.segments[:segment_index]:
        start=material_path_pose(start,prefix.translations,prefix.angular,prefix.shell_increment,1.)
    motion=field.segments[segment_index];target=field.patch.target
    end=material_path_pose(start,motion.translations,motion.angular,motion.shell_increment,1.)
    poses=(start,end);corners=[target.adapter.current_positions(p.shell_displacement) for p in poses]
    velocities=np.asarray(target.adapter.work_map@motion.shell_increment).reshape(3,3)
    edges=[(y[1]-y[0],y[2]-y[0]) for y in corners]
    a,b=edges[0];c,d=edges[1];da,db=velocities[1]-velocities[0],velocities[2]-velocities[0]
    controls=(np.cross(a,b),.5*(np.cross(a,d)+np.cross(c,b)),np.cross(c,d))
    reference=target.adapter.reference_positions
    normal=np.cross(reference[1]-reference[0],reference[2]-reference[0]);normal/=np.linalg.norm(normal)
    coordinate_scale=max(1.,*(float(np.max(np.abs(y))) for y in corners),float(np.max(np.abs(reference))))
    coordinate_error=1024*np.finfo(float).eps*coordinate_scale
    edge_scale=max(np.linalg.norm(e) for pair_edges in edges for e in pair_edges)
    area_error=1024*np.finfo(float).eps*max(1.,edge_scale**2)+8*coordinate_error*edge_scale+16*coordinate_error**2
    area_lower=min(float(normal@v) for v in controls)-area_error
    edge_change=c-a
    # Reverse triangle inequality avoids an ill-conditioned minimizer division.
    edge_lower=max(np.linalg.norm(a),np.linalg.norm(c))-np.linalg.norm(edge_change)-8*coordinate_error
    if area_lower<=0 or edge_lower<=0:raise ValueError('strip full-frame motion unresolved')
    velocity_error=1024*np.finfo(float).eps*max(1.,float(np.max(np.abs(velocities))))
    numerator_error=1024*np.finfo(float).eps*max(1.,edge_scale*(np.linalg.norm(da)+np.linalg.norm(db)))
    numerator_error+=8*(coordinate_error*(np.linalg.norm(da)+np.linalg.norm(db))+velocity_error*edge_scale)
    normal_rate=(max(np.linalg.norm(np.cross(da,e2)+np.cross(e1,db)) for e1,e2 in edges)+numerator_error)/area_lower
    tangent_rate=(np.linalg.norm(da)+4*velocity_error)/edge_lower
    frame_rate=2*(normal_rate+tangent_rate)
    radius=field.patch.rule.definition['radius'];omega=np.linalg.norm(motion.angular[cell])
    xis=(float(cell-1),float(cell))
    chord=max(np.linalg.norm(_base(p,cell,xis[1],ys[0])-_base(p,cell,xis[0],ys[0])) for p,ys in zip(poses,corners))
    relative=max(np.linalg.norm(_base(p,cell,xi,y)) for p,ys in zip(poses,corners) for xi in xis for y in ys)+radius
    speed=max(np.linalg.norm(motion.translations[i]-v) for i in (cell,cell+1) for v in velocities)+omega*radius
    scale=max(1.,relative,speed,chord,frame_rate,*(float(np.max(np.abs(y))) for y in corners))
    padding=1024*np.finfo(float).eps*scale
    return speed+frame_rate*relative+padding,chord,padding


def open_front_strip_work_bound(field,cell,axial_width,*,segment_index=None):
    if segment_index is None and len(field.segments)!=1:
        raise ValueError('strip work enclosure currently requires one released-origin segment')
    if segment_index is None:segment_index=0
    if not np.isfinite(axial_width) or axial_width<=0:
        raise ValueError('positive explicit strip width required')
    slip,chord,padding=_segment_motion_bounds(field,cell,segment_index)
    penetration=chord*axial_width+padding
    pair=field.patch.pairs[0];pressure=pair.normal_stiffness_Pa_per_m*penetration
    cap=pair.friction_coefficient*pressure;kt=pair.tangential_stiffness_Pa_per_m
    dissipation=cap*(slip+2*pair.event_history_substeps*cap/kt)
    release=cap*cap/(2*kt)
    result=np.array((0.,dissipation,release,dissipation))
    if not np.all(np.isfinite(result)):raise ValueError('nonfinite strip work enclosure')
    return result


def history_front_strip_work_bound(field,cell,strip_bounds,prepared,envelopes):
    """Work-only upper density for a complete, bound stationary history.

    Unlike the released-origin specialization, this includes captured-endpoint
    storage and inherited elastic state. It never supplies a residual/tangent or
    replaces a point history. Positive quadrature and exact strip work both lie
    in [0, area*density]; their difference is bounded by that same upper bound.
    Envelopes list potential extrema first, then every component endpoint.
    """
    from .native_surface_front import PreparedStationaryBand,_front_components
    left,right=map(float,strip_bounds)
    if not np.isfinite((left,right)).all() or not left<right:
        raise ValueError('ordered finite work-strip bounds required')
    if len(prepared)!=len(field.segments) or len(envelopes)!=len(prepared):
        raise ValueError('complete history work enclosure required')
    depths=[];counts=[];motions=[]
    for index,(band,values) in enumerate(zip(prepared,envelopes)):
        if (type(band) is not PreparedStationaryBand or band.cover.field_identity!=field.identity
                or band.cover.cell!=cell or band.cover.segment_index!=index
                or band.cover.xi_bounds is None
                or not band.cover.xi_bounds[0]<=left<right<=band.cover.xi_bounds[1]):
            raise ValueError('history work enclosure identity/domain mismatch')
        components=_front_components(band.cover);roots=len(band.branches)
        if len(values)!=roots+2*len(components):
            raise ValueError('complete component extrema envelopes required')
        if any(not np.isfinite(value).all() or not value[0]<=value[1] for value in values):
            raise ValueError('invalid component extrema enclosure')
        depth=0.
        for i,(start,end,_,_) in enumerate(components):
            extrema=(*values[roots+2*i:roots+2*i+2],*(values[j] for j,(a,b,_,_) in enumerate(band.branches)
                if start<=a<=b<=end))
            bl,bu=band.endpoints[2*i][2][:2]
            if bl>0:depth=max(depth,max(v[1] for v in extrema)-left)
            elif bu<0:depth=max(depth,right-min(v[0] for v in extrema))
            else:raise ValueError('history work enclosure denominator unresolved')
        # Each stationary witness divides one monotone interval. The extra
        # interval covers strictly open excluded slabs and boundary clamping.
        counts.append((field.patch.pairs[0].event_history_substeps+2)*(len(components)+roots+1))
        motions.append(_segment_motion_bounds(field,cell,index));depths.append(depth)
    penetration=max(chord*depth+padding for depth,(_,chord,padding) in zip(depths,motions))
    pair=field.patch.pairs[0];kt=pair.tangential_stiffness_Pa_per_m
    cap=pair.friction_coefficient*pair.normal_stiffness_Pa_per_m*penetration
    storage=cap*cap/(2*kt)
    dissipations=[cap*(slip+2*maps*cap/kt) for (slip,_,_),maps in zip(motions,counts)]
    result=np.array((storage,dissipations[-1],counts[-1]*storage,sum(dissipations)))
    if not np.all(np.isfinite(result)) or np.any(result<0):
        raise ValueError('nonfinite history strip work enclosure')
    return result
