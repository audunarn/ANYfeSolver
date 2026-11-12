"""Consistent nearest-curve B2 chain work with genuine endpoint potentials."""
from collections import Counter
from dataclasses import replace
from fractions import Fraction

import numpy as np
from scipy import sparse

from .beam_chain_pair import _BeamChainPairManager, validate_beam_chain_pair_contact
from .beam_pair_ends import _line_definition, _EndPairResults
from .beam_pair import _projection_work
from .beam_end_topology import physical_beam_ends
from .beam_plane import _beam_types
from .beam_pair_end_geometry import minimum_gap, admissible_line_projection_path
from .beam_end_radial_geometry import regular_curve_path
from .capped_cylinder import _sqrt_bound
from .core import ContactSolverSettings, ContactValidationReport

PROFILE = 'CLASSICAL_B2_CHAIN_ROUND_TUBE_PHYSICAL_ENDS_CONSISTENT_STATIC_DEV_V1'


def validate_beam_chain_ends(model, definition, settings=None):
    controls = ContactSolverSettings() if settings is None else settings
    if tuple(map(len,(definition.surfaces,definition.rigid_bodies,definition.interactions)))!=(2,0,1):
        return ContactValidationReport('invalid',('two explicit classical B2 chains and one pair required',),PROFILE)
    issues=[]
    if controls.beam_chain_tangent!='consistent_dev':
        issues.append("physical-end chains require beam_chain_tangent='consistent_dev'")
    if any(s.beam_end_caps!='hemisphere' for s in definition.surfaces) or (
            definition.interactions[0].endpoint_stiffness is None):
        issues.append('both chains need hemisphere geometry and explicit endpoint_stiffness N/m')
    if all(s.beam_radius is not None for s in definition.surfaces) and (
            definition.interactions[0].gap_tolerance>=sum(s.beam_radius for s in definition.surfaces)):
        issues.append('physical-end chain gap tolerance must be below the combined radius')
    report=validate_beam_chain_pair_contact(model,_line_definition(definition),controls)
    issues.extend(report.issues)
    return ContactValidationReport('invalid' if issues else 'ok',tuple(issues),PROFILE)


class BeamChainEndManager(_BeamChainPairManager):
    def __init__(self,model,definition,settings=None,initial_displacement=None):
        controls=ContactSolverSettings() if settings is None else settings
        report=validate_beam_chain_ends(model,definition,controls)
        if not report.valid:raise ValueError('unsupported physical-end chains: '+'; '.join(report.issues))
        super().__init__(model,_line_definition(definition),controls,initial_displacement)
        self.definition=definition;self.profile=PROFILE;self.end_caps='hemisphere'
        self.endpoint_stiffness=definition.interactions[0].endpoint_stiffness
        self.ends=tuple(physical_beam_ends(model,side,_beam_types()) for side in self.segments)
        self.target_nodes=tuple(tuple(cell.node_ids for cell in side) for side in self.segments)
        self.allowed_target_nodes=tuple(set(ends)|{node for node,count in
            Counter(n for cell in side for n in cell.node_ids).items() if count==2}
            for side,ends in zip(self.segments,self.ends))
        self.cell_indices=tuple({cell.element_id:i for i,cell in enumerate(side)} for side in self.segments)

    def _minimum_gap(self,u):
        a,b=self._positions(u)
        return min(minimum_gap(x,y,self.radius) for x in a for y in b)

    def _endpoint_work(self,poses,tangent):
        from .beam_chain_end_geometry import nearest_endpoint_feature
        n=self.model.mesh.dof_manager.total_dofs
        f=np.zeros(n);rows=[];cols=[];values=[];E=0.;fields=[];seen=set()
        for side in (0,1):
            for node,(cell,endpoint,_) in sorted(self.ends[side].items()):
                source=poses[side][self.cell_indices[side][cell.element_id]]
                choice=nearest_endpoint_feature(source,poses[1-side],self.target_nodes[1-side],endpoint,self.radius)
                feature=choice.feature;target_cell=self.segments[1-side][choice.owner]
                target=poses[1-side][choice.owner]
                row={'source_surface':self.definition.surfaces[side].name,
                    'source_element_id':cell.element_id,'source_node_id':node,
                    'target_surface':self.definition.surfaces[1-side].name,
                    'target_element_ids':[self.segments[1-side][i].element_id for i in choice.tied_owners],
                    'target_coordinate':float(feature.projection),'gap':feature.gap,'state':'open'}
                if feature.gap_sign>=0:fields.append(row);continue
                if feature.ownership>0:
                    row['state']='side_owned';fields.append(row);continue
                if choice.target_node is not None:
                    target_node=choice.target_node
                    if target_node not in self.allowed_target_nodes[1-side]:
                        raise ValueError('active endpoint projects to a nonphysical selected chain boundary')
                    row['target_node_id']=target_node
                    if target_node in self.ends[1-side]:
                        reverse=nearest_endpoint_feature(target,poses[side],self.target_nodes[side],
                            feature.target_endpoint,self.radius)
                        if reverse.target_node!=node or reverse.feature.ownership>=0:
                            raise ValueError('active cap pair is not reciprocal nearest physical ends')
                        key=('end_end',tuple(sorted(((side,node),(1-side,target_node)))))
                        row['feature_kind']='end_end'
                    else:
                        if len(choice.tied_owners)<2:
                            raise ValueError('selected chain junction lacks equivalent adjacent point work')
                        key=('end_junction',side,node,target_node)
                        row['feature_kind']='end_junction'
                else:
                    key=('end_interior',side,node,target_cell.element_id)
                    row['feature_kind']='end_interior'
                if key in seen:
                    row['state']='reciprocal_owned_once';fields.append(row);continue
                seen.add(key)
                t=float(feature.projection)
                _,grad,H=_projection_work(np.array((1.-endpoint,float(endpoint))),
                    np.array((1.-t,t)),np.array([float(x) for x in feature.separation]),
                    target[1]-target[0],feature.interior,self.radius)
                dofs=np.r_[cell.dofs.ravel(),target_cell.dofs.ravel()]
                gap=feature.gap;ke=self.endpoint_stiffness
                E+=.5*ke*gap*gap;np.add.at(f,dofs,ke*gap*grad)
                if tangent:
                    r,c=np.meshgrid(dofs,dofs,indexing='ij')
                    block=ke*(np.outer(grad,grad)+gap*H)
                    rows.extend(r.ravel());cols.extend(c.ravel());values.extend(block.ravel())
                row.update(state='active',endpoint_force_N=-ke*gap);fields.append(row)
        K=sparse.coo_matrix((values,(rows,cols)),shape=(n,n)).tocsr() if tangent else None
        return f,K,E,tuple(fields),len(seen)

    def evaluate(self,displacement,*,tangent=True):
        u=np.asarray(displacement,dtype=float).reshape(-1)
        n=self.model.mesh.dof_manager.total_dofs
        if u.size!=n or not np.all(np.isfinite(u)):raise ValueError('physical-end chain displacement is invalid')
        poses=self._positions(u)
        if not all(regular_curve_path(p,p,self.settings.maximum_swept_subdivisions)
                   for side in poses for p in side):
            raise ValueError('physical-end chain curve regularity is unresolved')
        ef,eK,end,fields,active=self._endpoint_work(poses,tangent)
        f,K,line=super().evaluate(u,tangent=tangent)
        f+=ef
        if tangent:K+=eK
        wrenches=[]
        for side in self.segments:
            force=np.zeros(3);moment=np.zeros(3)
            for node in sorted({n for cell in side for n in cell.node_ids}):
                dofs=self.model.mesh.dof_manager.get_node_dofs(node)[:3]
                physical=-f[dofs];position=np.asarray(self.model.mesh.get_node(node).coords())+u[dofs]
                force+=physical;moment+=np.cross(position,physical)
            wrenches.append((tuple(force),tuple(moment)))
        result=_EndPairResults(self.profile,line.minimum_gap,line.contact_energy+end,
            line.active_source_intervals,line.integration_points,line.maximum_integration_relative_error,
            wrenches[0][0],wrenches[1][0],wrenches[0][1],wrenches[1][1],line.gap_tolerance_met,
            self.stiffness,line.contact_energy,end,self.endpoint_stiffness,
            sum(map(len,self.ends)),active,fields)
        return f,K,result

    def _swept_admissible(self,start,end):
        first=self._positions(start)
        last=tuple(tuple(cell.positions(end) for cell in side) for side in self.segments)
        if not all(np.all(np.isfinite(p)) for side in last for p in side):return False
        depth=self.settings.maximum_swept_subdivisions
        if not all(regular_curve_path(a,b,depth) for x,y in zip(first,last) for a,b in zip(x,y)):return False
        def rational(poses):return tuple(tuple(np.array([[Fraction.from_float(float(x)) for x in row]
            for row in cell],dtype=object) for cell in side) for side in poses)
        q0,q1=rational(first),rational(last)
        delta=tuple(tuple(b-a for a,b in zip(x,y)) for x,y in zip(q0,q1))
        speed=max(_sqrt_bound(sum((x-y)**2 for x,y in zip(p,q)),upper=True)
            for a in delta[0] for p in a for b in delta[1] for q in b)
        def gap(t):return min(minimum_gap(a+t*da,b+t*db,self.radius)
            for a,da in zip(q0[0],delta[0]) for b,db in zip(q0[1],delta[1]))
        g0,g1=gap(Fraction(0)),gap(Fraction(1));tol=Fraction.from_float(self.gap_tolerance)
        if g0<-self.gap_tolerance:raise ValueError('committed physical-end chain exceeds gap tolerance')
        if g1<-self.gap_tolerance:return False
        def certify(l,r,gl,gr,k):
            if min(Fraction.from_float(gl),Fraction.from_float(gr))-speed*(r-l)/2>=-tol:return True
            if k>=depth:return False
            mid=(l+r)/2;gm=gap(mid)
            if gm<-self.gap_tolerance:return False
            return certify(l,mid,gl,gm,k+1) and certify(mid,r,gm,gr,k+1)
        if not certify(Fraction(0),Fraction(1),g0,g1,0):return False
        if not self._endpoint_path_admissible(q0,q1,depth):return False
        for a,da in zip(first[0],last[0]):
            for b,db in zip(first[1],last[1]):
                if not admissible_line_projection_path(a,da,b,db,self.radius,depth):return False
        return True

    def _endpoint_path_admissible(self,first,last,maximum_depth):
        """Common intervals bind endpoint nearest branches and cap reciprocity.

        Independent per-end certificates do not establish matching cap partners.
        On one common interval, every endpoint must be open or have a fixed
        nearest feature and ownership sign. Non-open cap partners must choose
        each other. Ambiguous bounds subdivide; samples only disprove admission.
        """
        from .beam_chain_end_geometry import (admissible_chain_endpoint_path,
            nearest_endpoint_feature,_polynomials,_distance_bounds)
        from .beam_pair_end_geometry import _affine
        R2=Fraction.from_float(self.radius)**2
        def opening(source0,source1,targets0,targets1,endpoint):
            source=tuple(_affine(p,q) for p,q in zip(source0,source1))
            targets=tuple(tuple(_affine(p,q) for p,q in zip(a,b)) for a,b in zip(targets0,targets1))
            return all(_distance_bounds(_polynomials(source,t,endpoint))[0]>=R2 for t in targets)
        def certify(a,b,depth):
            mid=tuple(tuple((x+y)/2 for x,y in zip(first,last)) for first,last in zip(a,b))
            choices={};opened={};fixed=True
            for side in (0,1):
                for node,(cell,endpoint,_) in self.ends[side].items():
                    index=self.cell_indices[side][cell.element_id]
                    source0,source1=a[side][index],b[side][index]
                    try:
                        choices[side,node]=nearest_endpoint_feature(mid[side][index],mid[1-side],
                            self.target_nodes[1-side],endpoint,self.radius)
                    except ValueError:return False
                    opened[side,node]=opening(source0,source1,a[1-side],b[1-side],endpoint)
                    fixed &= admissible_chain_endpoint_path(source0,source1,a[1-side],b[1-side],
                        self.target_nodes[1-side],endpoint,self.radius,0,
                        allowed_target_nodes=self.allowed_target_nodes[1-side])
            for (side,node),choice in choices.items():
                feature=choice.feature
                if opened[side,node] or feature.ownership>=0 or choice.target_node not in self.ends[1-side]:continue
                reverse=choices[1-side,choice.target_node]
                if reverse.target_node!=node or reverse.feature.ownership>=0:
                    if feature.gap_sign<0:return False
                    fixed=False  # Opening has not been proved over this interval.
            if fixed:return True
            if depth>=maximum_depth:return False
            return certify(a,mid,depth+1) and certify(mid,b,depth+1)
        return certify(first,last,0)

    def restart_contract(self):
        return {**super().restart_contract(),'endpoint_stiffness':self.endpoint_stiffness,
            'physical_end_nodes':[sorted(ends) for ends in self.ends],
            'represented_junction_nodes':[sorted(nodes) for nodes in self.allowed_target_nodes],
            'point_law':'nearest_owned_endpoint_reciprocal_cap_or_material_junction_v1',
            'endpoint_geometry_revision':'exact_affine_nearest_feature_projection_ownership_v1'}
