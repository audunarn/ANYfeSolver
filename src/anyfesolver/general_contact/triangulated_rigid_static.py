"""Planar finite rigid mesh against the existing affine S3 physical skin map.

All face integrals are accumulated before E = k/2 sum(w_i**2/s_i).
Geometry derivatives come from the existing S3 overlap Jet2 primitive.
"""
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import numpy as np
from scipy import sparse
from .core import ContactSolverSettings, ContactValidationReport, _contact_settings_contract
from .triangulated_rigid import TriangulatedRigidContactBody
from .s3_skin_work import S3SkinWorkAdapter
from .s3_patch_mortar_energy import MovingS3PatchPrimalMortarPenalty, MovingS3PatchMortarWork, S3PatchFace
from .s3_patch_static import GeneralS3PatchManager, _unique_work_map
from .s3_overlap_derivatives import s3_overlap_derivatives

PROFILE = 'S3_PLANAR_TRIANGULATED_RIGID_SHARED_PENALTY_STATIC_DEV_V1'
PATCH_PROFILE = 'S3_PATCH_PLANAR_TRIANGULATED_RIGID_SHARED_PENALTY_STATIC_DEV_V1'
LOCAL_DERIVATIVE_WORK_LIMIT_BYTES = 128*1024**2

def derivative_work_storage_bound(n,m,pairs):
    if any(type(v) is not int or v<1 for v in (n,m,pairs)):
        raise ValueError('positive integer derivative-work inventory required')
    # Js/Jw/Hs/Hw + returned H/residual + twelve dense matrix temporaries,
    # pressure gram, pair state/coordinate scratch, and sequential 18D Jets.
    return 8*(2*n*m*m+2*n*m+13*m*m+16*m+n*n+512*pairs)+4*1024**2

def _storage_guard(n,m,pairs):
    estimate=derivative_work_storage_bound(n,m,pairs)
    if estimate>LOCAL_DERIVATIVE_WORK_LIMIT_BYTES:
        raise ValueError('128MiB local derivative-work budget exceeded before dense allocation')
    return estimate


def _source_overlap_guard(points,faces):
    """Exact projected positive overlap refuses duplicate source ownership."""
    def sub(a,b): return [x-y for x,y in zip(a,b)]
    def dot(a,b): return sum(x*y for x,y in zip(a,b))
    def cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    def orient(a,b,c): return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    triangles=[[[Fraction(float(x)) for x in points[int(j)]] for j in ids] for ids in faces]
    for i,a in enumerate(triangles):
        n=cross(sub(a[1],a[0]),sub(a[2],a[0]));square=dot(n,n)
        if square<=0: raise ValueError('source patch contains degenerate geometry')
        drop=max(range(3),key=lambda j:abs(n[j]));axes=[j for j in range(3) if j!=drop]
        first=[tuple(p[j] for j in axes) for p in a]
        if orient(*first)<0: first.reverse()
        for b in triangles[i+1:]:
            projected=[[x-nj*dot(n,sub(p,a[0]))/square for x,nj in zip(p,n)] for p in b]
            second=[tuple(p[j] for j in axes) for p in projected]
            if orient(*second)==0: raise ValueError('source patch projection ownership is unresolved')
            if orient(*second)<0: second.reverse()
            poly=list(first)
            for x,y in zip(second,second[1:]+second[:1]):
                if not poly: break
                out=[];old=poly[-1];od=orient(x,y,old)
                for current in poly:
                    d=orient(x,y,current)
                    if (od>=0)!=(d>=0):
                        f=od/(od-d);out.append(tuple(u+f*(v-u) for u,v in zip(old,current)))
                    if d>=0: out.append(current)
                    old,od=current,d
                poly=out
            area=sum(x[0]*y[1]-x[1]*y[0] for x,y in zip(poly,poly[1:]+poly[:1])) if poly else 0
            if area>0: raise ValueError('source reference/current faces have overlapping projected interiors')


class _JetPenalty(MovingS3PatchPrimalMortarPenalty):
    def evaluate(self,source,target):
        state=super().evaluate(source,target)
        if not getattr(self,'stable_plane_gap',False):
            return state
        from dataclasses import replace
        support=np.zeros_like(state.reference_support)
        weighted=np.zeros_like(state.reference_weighted_gap)
        for index,pair in enumerate(state.pair_states):
            if pair is None or pair.overlap_vertices==0:
                continue
            si=self.source_face_indices[index//len(self.target_faces)]
            ti=self.target_face_indices[index%len(self.target_faces)]
            d=s3_overlap_derivatives(self.reference_source[si],source[si],target[ti],stable_plane_gap=True)
            support[si]+=d.reference_support
            weighted[si]+=d.reference_weighted_gap
        if np.any(support<=0) or not np.all(np.isfinite(weighted)):
            raise ValueError('stable shared penalty requires every positive finite support')
        normalized=weighted/support
        pressure=self.stiffness*np.maximum(-normalized,0.)
        energy=.5*self.stiffness*float(support@np.minimum(normalized,0.)**2)
        return replace(state,reference_support=support,reference_weighted_gap=weighted,
                       normalized_gap=normalized,pressure=pressure,energy=energy)

    def work(self, source, target, *, tangent=True):
        n, m = len(self.source_node_ids), self.n_coordinates
        _storage_guard(n,m,len(self.pair_laws))
        state = self.evaluate(source, target)
        js, jw = np.zeros((n,m)), np.zeros((n,m))
        hs, hw = np.zeros((n,m,m)), np.zeros((n,m,m))
        count = 0
        for index, pair in enumerate(state.pair_states):
            if pair is None or pair.overlap_vertices == 0:
                continue
            si = self.source_face_indices[index//len(self.target_faces)]
            ti = self.target_face_indices[index%len(self.target_faces)]
            d = s3_overlap_derivatives(self.reference_source[si], source[si], target[ti],
                                      stable_plane_gap=getattr(self,'stable_plane_gap',False))
            coords = np.r_[np.concatenate([np.arange(3*j,3*j+3) for j in si]),
                           np.concatenate([np.arange(3*n+3*j,3*n+3*j+3) for j in ti])]
            for row, node in enumerate(si):
                js[node,coords] += d.support_gradient[row]
                jw[node,coords] += d.weighted_gap_gradient[row]
                hs[node][np.ix_(coords,coords)] += d.support_hessian[row]
                hw[node][np.ix_(coords,coords)] += d.weighted_gap_hessian[row]
            count += 1
        force, hessian = np.zeros(m), np.zeros((m,m))
        for i,(s,w) in enumerate(zip(state.reference_support,state.reference_weighted_gap)):
            if s <= 0 or w >= 0:
                continue
            force += self.stiffness*(w/s*jw[i]-.5*w*w/s**2*js[i])
            hessian += self.stiffness*(np.outer(jw[i],jw[i])/s
                -w/s**2*(np.outer(jw[i],js[i])+np.outer(js[i],jw[i]))
                +w*w/s**3*np.outer(js[i],js[i])+w/s*hw[i]-.5*w*w/s**2*hs[i])
        return MovingS3PatchMortarWork(state, force, hessian if tangent else None, count)


def _prepare(model, definition, settings):
    if len(definition.surfaces)!=1 or len(definition.rigid_bodies)!=1 or len(definition.interactions)!=1:
        raise ValueError('first rigid mesh profile requires one shell surface and one rigid body')
    body, surface, interaction = definition.rigid_bodies[0], definition.surfaces[0], definition.interactions[0]
    if type(body) is not TriangulatedRigidContactBody:
        raise ValueError('explicit TriangulatedRigidContactBody required')
    if (interaction.surface != surface.name or interaction.target != body.name
            or interaction.normal_law!='penalty'
            or surface.beam_radius is not None or interaction.line_stiffness is not None
            or interaction.endpoint_stiffness is not None or interaction.normal_stiffness is None):
        raise ValueError('only frictionless shell-source penalty is supported')
    if (settings.pair_motion!='general' or settings.normal_discretization!='primal_mortar'
            or settings.general_s3_difference_step is None or settings.d1_interior_profile
            or settings.q4_source_gap_certificate!='whole_face'):
        raise ValueError('rigid mesh requires explicit general primal mortar settings')
    from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
    adapters=[]
    for eid in surface.element_ids:
        if type(model.mesh.elements[eid]) is not NativeParityE4PLS3V2DShellElement:
            raise ValueError('only formulation-owned e4-pl-s3-v2d physical skins are supported; Q4 work remains unverified')
        material=model.materials[model.mesh.elements[eid].material_name]
        if 0 < float(getattr(material,'yield_stress',0.)) < float('inf'):
            raise ValueError('rigid mesh profile currently requires elastic material')
        adapters.append(S3SkinWorkAdapter.from_model(model, 1, eid, surface.side))
    if settings.enforcement_solver!='nested_augmented':
        raise ValueError('rigid mesh profile requires nested_augmented enforcement')
    if len(adapters)>1:
        traces={}
        for adapter in adapters:
            for local,node in enumerate(adapter.node_ids):
                position=adapter.reference_positions[local]
                row=adapter.work_map[3*local:3*local+3]
                if node in traces:
                    old_position,old_row=traces[node]
                    if not np.array_equal(position,old_position) or (row-old_row).nnz:
                        raise ValueError('source shared physical trace/work rows must agree exactly')
                else:traces[node]=(position,row)
        neighbours={i:set() for i in range(len(adapters))};edges={}
        for i,a in enumerate(adapters):
            for first,last in zip(a.node_ids,a.node_ids[1:]+a.node_ids[:1]):
                key=tuple(sorted((first,last)));entries=edges.setdefault(key,[])
                if len(entries)==2 or any((first,last)==e[1:] for e in entries):
                    raise ValueError('source patch has nonmanifold or inverted shared edges')
                entries.append((i,first,last))
        for entries in edges.values():
            if len(entries)==2:
                i,j=entries[0][0],entries[1][0];neighbours[i].add(j);neighbours[j].add(i)
        reached={0};pending=[0]
        while pending:
            for i in neighbours[pending.pop()]-reached:
                reached.add(i);pending.append(i)
        if len(reached)!=len(adapters): raise ValueError('source patch must be connected through complete shared edges')
    positions=body.planar_positions()
    source=tuple(S3PatchFace(a.node_ids,a.reference_positions) for a in adapters)
    offset=max(node for a in adapters for node in a.node_ids)+1
    target=tuple(S3PatchFace(tuple(j+offset for j in ids),positions[list(ids)]) for _,ids in body.faces)
    nsource=len({j for face in source for j in face.node_ids});ntarget=len({j for face in target for j in face.node_ids})
    if nsource>settings.maximum_patch_normal_dofs: raise ValueError('shared pressure node budget exceeded')
    _storage_guard(nsource,3*(nsource+ntarget),len(source)*len(target))
    reference,_=_unique_work_map(adapters,tuple(sorted({j for face in source for j in face.node_ids})))
    if len(adapters)>1:
        node_index={node:i for i,node in enumerate(sorted({j for face in source for j in face.node_ids}))}
        _source_overlap_guard(reference,tuple(tuple(node_index[j] for j in f.node_ids) for f in source))
    law=_JetPenalty(source,target,interaction.normal_stiffness,
        difference_step=settings.general_s3_difference_step,
        maximum_energy_evaluations=settings.maximum_general_s3_energy_evaluations,
        differentiation='jet2_s3', maximum_candidate_pairs=settings.maximum_candidate_pairs)
    law.maximum_candidate_pairs=settings.maximum_candidate_pairs
    if len(law.source_node_ids)>settings.maximum_patch_normal_dofs:
        raise ValueError('shared pressure node budget exceeded')
    return adapters,law,interaction


def validate_triangulated_rigid(model, definition, settings=None, **kwargs):
    try:
        if kwargs.get('analysis_control','force') not in ('force','displacement'):
            raise ValueError('rigid mesh static profile requires force/displacement control')
        _prepare(model,definition,settings or ContactSolverSettings())
    except (ValueError,KeyError,TypeError) as exc:
        profile=PATCH_PROFILE if len(definition.surfaces)==1 and len(definition.surfaces[0].element_ids)>1 else PROFILE
        return ContactValidationReport('unsupported',(str(exc),),profile)
    return ContactValidationReport('ok',(),PATCH_PROFILE if len(definition.surfaces[0].element_ids)>1 else PROFILE)


class TriangulatedRigidManager(GeneralS3PatchManager):
    def __init__(self,model,definition,settings=None,initial_displacement=None):
        self.settings=settings or ContactSolverSettings()
        self.adapters,self.law,interaction=_prepare(model,definition,self.settings)
        self.model,self.definition,self.profile=model,definition,(PATCH_PROFILE if len(self.adapters)>1 else PROFILE)
        source,map_=_unique_work_map(self.adapters,self.law.source_node_ids)
        self.reference=(source,self.law.reference_target)
        self.n_dof=model.mesh.dof_manager.total_dofs
        self.map=sparse.vstack((map_,sparse.csr_matrix((3*len(self.law.target_node_ids),self.n_dof)))).tocsr()
        self.stiffness,self.gap_tolerance=interaction.normal_stiffness,interaction.gap_tolerance
        self.augmented=self.projected_work=False
        self.finite_skin=None
        self.swept_trial_cutbacks=0
        from .s3_pair_static import _normal
        self.reference_normals=(_normal(source[self.law.source_face_indices[0]]),_normal(self.law.reference_target[self.law.target_face_indices[0]]))
        self._binding=self._contract_bytes()
        u=np.zeros(self.n_dof) if initial_displacement is None else initial_displacement
        lower=self._gap_lower(self._positions(u))
        scale=max(1.,float(np.max(np.abs(np.concatenate(self.reference)))))
        if lower < -256*np.finfo(float).eps*scale:
            raise ValueError('initial rigid mesh overclosure is unsupported')

    def _contract_bytes(self):
        contract={'definition':asdict(self.definition),'settings':_contact_settings_contract(self.settings),
            'profile':self.profile,'reference':[a.reference_positions.tolist() for a in self.adapters],
            'model_reference':[S3SkinWorkAdapter.from_model(self.model,a.body_id,a.element_id,a.side).reference_positions.tolist() for a in self.adapters],
            'model_maps':[S3SkinWorkAdapter.from_model(self.model,a.body_id,a.element_id,a.side).work_map.toarray().tolist() for a in self.adapters],
            'law_source':self.law.reference_source.tolist(),'law_target':self.law.reference_target.tolist(),
            'law_faces':[[(f.node_ids,f.reference_positions.tolist()) for f in side] for side in (self.law.source_faces,self.law.target_faces)],
            'references':[r.tolist() for r in self.reference],
            'operator':[self.law.stiffness,self.stiffness,self.gap_tolerance,self.law.differentiation,
                [list(map(int,i)) for i in self.law.source_face_indices],[list(map(int,i)) for i in self.law.target_face_indices],
                [n.tolist() for n in self.reference_normals],
                [(p.stiffness,p.reference_source.tolist(),p.reference_target.tolist()) for p in self.law.pair_laws],
                self.law.source_node_ids,self.law.target_node_ids,self.law.n_coordinates,
                self.law.maximum_candidate_pairs,self.law.maximum_energy_evaluations],
            'map_data':self.map.data.tolist(),'map_indices':self.map.indices.tolist(),'map_indptr':self.map.indptr.tolist()}
        if self.profile==PATCH_PROFILE:
            contract['patch_derivative_work']={'policy':'explicit128MiB-v1','limit_bytes':LOCAL_DERIVATIVE_WORK_LIMIT_BYTES,
                'estimate_bytes':derivative_work_storage_bound(len(self.law.source_node_ids),self.law.n_coordinates,len(self.law.pair_laws))}
        return json.dumps(contract,sort_keys=True).encode()

    def _positions(self,u):
        if self._contract_bytes()!=self._binding:
            raise ValueError('rigid mesh contact binding mutated')
        positions=super()._positions(u)
        if self.profile==PATCH_PROFILE: _source_overlap_guard(positions[0],self.law.source_face_indices)
        return positions

    def _gap_lower(self,positions,*,endpoint_errors=None):
        if endpoint_errors is not None:
            raise ValueError('affine rigid profile does not consume finite endpoint boxes')
        state=self.law.evaluate(*positions)
        value=state.minimum_physical_gap
        self._reject_edge_only(positions,state)
        if value is None:
            return float('inf')
        scale=max(1.,*(float(np.max(np.abs(p))) for p in positions))
        return float(value)-256*np.finfo(float).eps*scale

    def _reject_edge_only(self,positions,state):
        """Every unrepresented singular intersection needs represented ownership."""
        def q(p): return [Fraction(float(x)) for x in p]
        def sub(a,b): return [x-y for x,y in zip(a,b)]
        def dot(a,b): return sum(x*y for x,y in zip(a,b))
        def cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
        def orient(a,b,c): return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        def clip(first,second):
            poly=first
            for a,b in zip(second,second[1:]+second[:1]):
                if not poly: break
                out=[];old=poly[-1];od=orient(a,b,old)
                for current in poly:
                    d=orient(a,b,current)
                    if (od>=0)!=(d>=0):
                        f=od/(od-d)
                        out.append(tuple(x+f*(y-x) for x,y in zip(old,current)))
                    if d>=0: out.append(current)
                    old,od=current,d
                poly=out
            return poly
        for source_index,si in enumerate(self.law.source_face_indices):
            source=[q(p) for p in positions[0][si]]
            ns=cross(sub(source[1],source[0]),sub(source[2],source[0]))
            reference=[q(p) for p in positions[1][self.law.target_face_indices[0]]]
            nt=cross(sub(reference[1],reference[0]),sub(reference[2],reference[0]))
            heights=[dot(nt,sub(p,reference[0])) for p in source]
            if min(heights)>0: continue
            denominator=dot(nt,ns)
            if denominator<=0: raise ValueError('unresolved rigid rim polarity')
            projected=[[x-n*h/denominator for x,n in zip(p,ns)] for p,h in zip(source,heights)]
            drop=max(range(3),key=lambda i:abs(nt[i]));axes=[i for i in range(3) if i!=drop]
            original=[tuple(p[i] for i in axes) for p in projected]
            first=original if orient(*original)>0 else list(reversed(original))
            inventory=[];represented=[]
            for target_index,ti in enumerate(self.law.target_face_indices):
                second=[tuple(q(p)[i] for i in axes) for p in positions[1][ti]]
                if orient(*second)<0: second.reverse()
                poly=clip(first,second)
                pair=state.pair_states[source_index*len(self.law.target_faces)+target_index]
                area=sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(poly,poly[1:]+poly[:1])) if poly else Fraction(0)
                if pair is not None and pair.overlap_vertices>0:
                    if area<=0: raise ValueError('unresolved positive rigid overlap representation')
                    represented.append(poly)
                elif poly:
                    inventory.append(poly)
            determinant=orient(*original)
            def height(point):
                b=orient(original[0],point,original[2])/determinant
                c=orient(original[0],original[1],point)/determinant
                return (1-b-c)*heights[0]+b*heights[1]+c*heights[2]
            for poly in inventory:
                if min(height(point) for point in poly)>0: continue
                # A represented polygon is convex. Covering every endpoint of
                # the closed intersection proves coverage of its whole hull.
                covered=any(candidate and all(all(orient(a,b,point)>=0 for a,b in zip(candidate,candidate[1:]+candidate[:1])) for point in poly) for candidate in represented)
                if not covered:
                    raise ValueError('unrepresented physical rigid rim contact is unsupported')

    def restart_contract(self):
        self._positions(np.zeros(self.n_dof))
        return {'profile':self.profile,'rigid_mesh_contract_sha256':hashlib.sha256(self._binding).hexdigest()}

    def _swept_admissible(self,start,end):
        """Bound the matched normal gap on a superset of finite overlap.

        Frozen tool plane and affine physical S3 positions make each vertex
        height affine and the unnormalized source normal quadratic. All dot
        and cross coefficients below use exact stored binary64 arithmetic.
        """
        first,last=self._positions(start),self._positions(end)
        if self._gap_lower(first)<-self.gap_tolerance:
            raise ValueError('committed rigid mesh gap exceeds tolerance')
        try:
            if self._gap_lower(last)<-self.gap_tolerance: return False
        except ValueError:
            return False
        def q(p): return [Fraction(float(x)) for x in p]
        def sub(a,b): return [x-y for x,y in zip(a,b)]
        def dot(a,b): return sum(x*y for x,y in zip(a,b))
        def cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
        def normal(p): return cross(sub(p[1],p[0]),sub(p[2],p[0]))
        def down(x):
            f=float(x)
            return np.nextafter(f,-np.inf) if Fraction(f)>x else f
        def up(x):
            f=float(x)
            return np.nextafter(f,np.inf) if Fraction(f)<x else f
        def sqrt_up(x):
            f=float(np.sqrt(up(x)))
            while Fraction(f)**2<x: f=np.nextafter(f,np.inf)
            return f
        tool=[q(p) for p in first[1][self.law.target_face_indices[0]]]
        nt=normal(tool)
        tool_norm=sqrt_up(dot(nt,nt))
        # Enclose ref + map*u before cancellation, including all affine path t.
        refs=np.concatenate(self.reference).reshape(-1)
        endpoints=[]; magnitude=Fraction(1)
        for displacement in (start,end):
            values=[]
            for row,ref in enumerate(refs):
                begin,stop=self.map.indptr[row:row+2]
                terms=[Fraction(float(self.map.data[j]))*Fraction(float(displacement[self.map.indices[j]])) for j in range(begin,stop)]
                values.append(Fraction(float(ref))+sum(terms))
                magnitude=max(magnitude,abs(Fraction(float(ref)))+sum(abs(t) for t in terms))
            endpoints.append([values[j:j+3] for j in range(0,len(values),3)])
        err=Fraction(1024)*Fraction(float(np.finfo(float).eps))*magnitude
        spatial_error=Fraction(sqrt_up(3*err*err))
        guard=spatial_error

        for ids in self.law.source_face_indices:
            a,b=([pose[int(j)] for j in ids] for pose in endpoints)
            middle=[[(x+y)/2 for x,y in zip(p,r)] for p,r in zip(a,b)]
            n0,nm,n1=normal(a),normal(middle),normal(b)
            controls=(n0,[2*x-(y+z)/2 for x,y,z in zip(nm,n0,n1)],n1)
            d=min(dot(nt,n) for n in controls)
            norm=max(sqrt_up(dot(n,n)) for n in controls)
            edge1=max(sqrt_up(dot(sub(p[1],p[0]),sub(p[1],p[0]))) for p in (a,b))
            edge2=max(sqrt_up(dot(sub(p[2],p[0]),sub(p[2],p[0]))) for p in (a,b))
            edge_error=2*spatial_error
            normal_error=(Fraction(edge1)+Fraction(edge2))*edge_error+edge_error**2
            d-=Fraction(tool_norm)*normal_error
            norm=up(Fraction(norm)+normal_error)
            if d<=0 or norm<=0: return False
            # A position error enclosure is applied before the normal quotient.
            height=min(dot(nt,sub(p,tool[0])) for p in (*a,*b))-guard*Fraction(tool_norm)
            lower=(down(height*Fraction(norm)/d) if height<0 else down(height/Fraction(tool_norm)))
            if lower < -self.gap_tolerance: return False
        return True
