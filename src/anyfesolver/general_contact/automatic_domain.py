"""Opt-in declared shell/rigid domain compiler; no geometric discovery.

Imports of solver adapters and numerical packages occur only on resolution.
The output is an ordinary ContactDefinition admitted by existing validators.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import json
from statistics import median

POLICY = 'DECLARED_SHELL_RIGID_AUTOMATIC_DOMAIN_DEV_V1'
AREA_POLICY = 'SELECTED_PLANAR_SKIN_S3_EXACT_Q4_ANALYTIC_JACOBIAN_V1'


def _settings_contract(settings):
    # Keep this compiler module free of eager solver/numerical imports.
    from .core import _contact_settings_contract
    return _contact_settings_contract(settings)


class AutomaticContactDomainError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def _error(code, message):
    raise AutomaticContactDomainError(code,message)


def _name(value):
    return type(value) is str and bool(value.strip())


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _restore_rigid_component(value):
    """Restore a canonical owner's unit vector without renormalizing its bits.

    All component fields still pass the ordinary owner constructor. Only its
    canonical finite unit normal/axis is restored after that validation;
    noncanonical fields or vectors refuse rather than silently normalize.
    """
    from math import isfinite
    from .core import RigidContactBody
    normal=value.get('normal') if type(value) is dict else None
    if normal is not None:
        if (type(normal) not in (tuple,list) or len(normal)!=3
                or any(type(x) not in (float,int) or isinstance(x,bool) or not isfinite(x) for x in normal)
                or any(abs(float(x))>1.+64*2.220446049250313e-16 for x in normal)
                or abs(sum(float(x)**2 for x in normal)-1.)>64*2.220446049250313e-16):
            _error('invalid_persistence','stored rigid normal/axis must be a canonical finite unit vector')
    try:
        body=RigidContactBody(**value)
    except (TypeError,ValueError) as exc:
        _error('invalid_persistence',f'rigid owner constructor refused persisted fields: {exc}')
    if normal is not None:
        object.__setattr__(body,'normal',tuple(float(x) for x in normal))
    if _canonical(asdict(body))!=_canonical(value):
        _error('invalid_persistence','rigid fields are not an exact canonical owner record')
    return body


@dataclass(frozen=True)
class AutomaticContactParticipant:
    participant: object
    body_id: str

    def __post_init__(self):
        from .core import ContactSurface,RigidContactBody
        if type(self.participant) not in (ContactSurface,RigidContactBody) or not _name(self.body_id):
            _error('invalid_participant','exact shell/rigid participant and explicit nonempty body_id required')
        if not _name(self.participant.name):
            _error('invalid_participant','participant ID must be a nonempty string')
        if type(self.participant) is ContactSurface:
            object.__setattr__(self,'participant',replace(self.participant,element_ids=tuple(sorted(self.participant.element_ids))))


@dataclass(frozen=True)
class AutomaticContactDefaults:
    normal_stiffness_n_per_m3: float
    gap_tolerance_m: float
    normal_law: str = 'penalty'

    def __post_init__(self):
        from .core import ContactInteraction
        for value in (self.normal_stiffness_n_per_m3,self.gap_tolerance_m):
            if type(value) not in (int,float) or isinstance(value,bool):
                _error('invalid_defaults','dimensioned defaults must be finite positive numbers')
        check=ContactInteraction('defaults','source','target',self.normal_stiffness_n_per_m3,
                                 self.gap_tolerance_m,self.normal_law)
        object.__setattr__(self,'normal_stiffness_n_per_m3',check.normal_stiffness)
        object.__setattr__(self,'gap_tolerance_m',check.gap_tolerance)


@dataclass(frozen=True)
class AutomaticContactDomain:
    name: str
    participants: tuple[AutomaticContactParticipant,...]
    defaults: AutomaticContactDefaults
    excluded_pairs: tuple[tuple[str,str],...] = ()
    pair_overrides: tuple = ()
    maximum_pairs: int = 100

    def __post_init__(self):
        from .core import ContactInteraction
        participants=tuple(self.participants)
        if (not _name(self.name) or type(self.defaults) is not AutomaticContactDefaults
                or len(participants)<2 or any(type(p) is not AutomaticContactParticipant for p in participants)):
            _error('invalid_domain','named domain, defaults and at least two exact participants required')
        names=tuple(p.participant.name for p in participants)
        if len(set(names))!=len(names):
            _error('duplicate_participant','participant IDs must be unique across shell and rigid declarations')
        if type(self.maximum_pairs) is not int or self.maximum_pairs<1:
            _error('invalid_pair_cap','maximum_pairs must be a positive integer')
        excluded=[]
        for pair in self.excluded_pairs:
            if (type(pair) not in (tuple,list) or len(pair)!=2 or any(not _name(x) for x in pair)
                    or pair[0]==pair[1] or not set(pair)<=set(names)):
                _error('invalid_exclusion','exclusions require two distinct declared participant IDs')
            excluded.append(tuple(sorted(pair)))
        overrides=tuple(self.pair_overrides)
        if any(type(p) is not ContactInteraction for p in overrides):
            _error('invalid_override','pair overrides must be exact ContactInteraction records')
        object.__setattr__(self,'participants',tuple(sorted(participants,key=lambda p:p.participant.name)))
        object.__setattr__(self,'excluded_pairs',tuple(sorted(set(excluded))))
        object.__setattr__(self,'pair_overrides',overrides)

    def to_dict(self):
        from .core import ContactSurface
        return json.loads(_canonical(dict(schema=POLICY,name=self.name,
            participants=[dict(body_id=p.body_id,kind='shell' if type(p.participant) is ContactSurface else 'rigid',participant=asdict(p.participant)) for p in self.participants],
            defaults=asdict(self.defaults),excluded_pairs=self.excluded_pairs,
            pair_overrides=sorted((asdict(pair) for pair in self.pair_overrides),key=_canonical),maximum_pairs=self.maximum_pairs)))

    @classmethod
    def from_dict(cls,value):
        from .core import ContactSurface,RigidContactBody,ContactInteraction
        if type(value) is not dict or set(value)!={'schema','name','participants','defaults','excluded_pairs','pair_overrides','maximum_pairs'} or value['schema']!=POLICY:
            _error('invalid_persistence','exact supported domain schema required; unknown fields are not dropped')
        participants=[]
        for row in value['participants']:
            if type(row) is not dict or set(row)!={'body_id','kind','participant'} or row['kind'] not in ('shell','rigid'):
                _error('invalid_persistence','exact declared participant persistence required')
            participant=(ContactSurface(**row['participant']) if row['kind']=='shell'
                         else _restore_rigid_component(row['participant']))
            participants.append(AutomaticContactParticipant(participant,row['body_id']))
        return cls(value['name'],tuple(participants),AutomaticContactDefaults(**value['defaults']),
                   tuple(tuple(pair) for pair in value['excluded_pairs']),tuple(ContactInteraction(**pair) for pair in value['pair_overrides']),value['maximum_pairs'])


@dataclass(frozen=True)
class AutomaticContactPairSelection:
    participants: tuple[str,str]
    disposition: str
    source: str | None = None
    target: str | None = None
    original_interaction_name: str | None = None
    compiled_interaction_name: str | None = None
    reason: str = ''
    source_reference_median_area_m2: float | None = None
    target_reference_median_area_m2: float | None = None


@dataclass(frozen=True)
class AutomaticContactResolution:
    definition: object
    selections: tuple[AutomaticContactPairSelection,...]
    identity: str
    canonical_contract_json: str
    reference_json: str
    settings_json: str
    definition_json: str
    outside_family_pair_count: int = 0

    def to_dict(self):
        return dict(identity=self.identity,contract=json.loads(self.canonical_contract_json),
                    selections=[asdict(row) for row in self.selections],definition=asdict(self.definition),outside_family_pair_count=self.outside_family_pair_count)

    @classmethod
    def from_dict(cls,value):
        """Restore the exact fixed selection; never silently resolve a new one.

        The digest binds this canonical record, not an external producer's
        authenticity. The ordinary model/domain/settings verification is still
        required immediately before solve, and checkpoint identity remains a
        separate owner check.
        """
        required={'identity','contract','selections','definition','outside_family_pair_count'}
        if type(value) is not dict or set(value)!=required:
            _error('invalid_persistence','exact supported resolution fields required')
        contract=value['contract'];identity=value['identity']
        keys={'policy','area_policy','domain_name','participants','defaults','excluded_pairs','maximum_pairs',
              'explicit_pairs','settings','domain_pair_overrides','explicit_input_provenance','persisted_domain',
              'outside_family_pair_count','analysis_control','reference','selections'}
        if (type(contract) is not dict or set(contract)!=keys or contract['policy']!=POLICY
                or contract['area_policy']!=AREA_POLICY or type(identity) is not str
                or len(identity)!=64 or any(c not in '0123456789abcdef' for c in identity)
                or type(value['outside_family_pair_count']) is not int or value['outside_family_pair_count']<0):
            _error('invalid_persistence','exact canonical resolution contract and digest required')
        try:
            canonical=_canonical(contract)
            if hashlib.sha256(canonical.encode()).hexdigest()!=identity:
                _error('invalid_persistence','resolution contract digest changed')
            domain=AutomaticContactDomain.from_dict(contract['persisted_domain'])
            if _canonical(domain.to_dict())!=_canonical(contract['persisted_domain']):
                _error('invalid_persistence','domain persistence requires normalization')
            declaration=dict(domain_name=domain.name,participants=[dict(body_id=p.body_id,participant=asdict(p.participant)) for p in domain.participants],
                             defaults=asdict(domain.defaults),excluded_pairs=domain.excluded_pairs,maximum_pairs=domain.maximum_pairs)
            if any(_canonical(v)!=_canonical(contract[k]) for k,v in declaration.items()):
                _error('invalid_persistence','domain declaration mirrors changed')
            overrides=sorted({_canonical(asdict(pair)) for pair in domain.pair_overrides})
            if overrides!=contract['domain_pair_overrides']:
                _error('invalid_persistence','domain override provenance changed')
            from .core import ContactSolverSettings,ContactInteraction
            settings=ContactSolverSettings(**contract['settings'])
            if _canonical(_settings_contract(settings))!=_canonical(contract['settings']):
                _error('invalid_persistence','settings are not exact canonical owner fields')
            for field in ('explicit_pairs','explicit_input_provenance'):
                if type(contract[field]) is not list:
                    _error('invalid_persistence','explicit interaction inventory must be a list')
                for pair in contract[field]:
                    if type(pair) is not dict or _canonical(asdict(ContactInteraction(**pair)))!=_canonical(pair):
                        _error('invalid_persistence','explicit interaction fields require normalization')
            _verify_stored_selection_policy(contract,domain)
            definition,records=_compiled_from_contract(contract,identity)
            restored=cls(definition,records,identity,canonical,_canonical(contract['reference']),
                         _canonical(contract['settings']),_canonical(asdict(definition)),contract['outside_family_pair_count'])
            if _canonical(restored.to_dict())!=_canonical(value):
                _error('invalid_persistence','selection, definition or metadata changed')
            return restored
        except AutomaticContactDomainError:
            raise
        except (KeyError,TypeError,ValueError,AttributeError,OverflowError) as exc:
            _error('invalid_persistence',f'invalid canonical resolution record: {exc}')


def _verify_stored_selection_policy(contract,domain):
    """Validate recorded decisions against recorded declarations/reference.

    No current model observation, resolver or replacement selection is used.
    Current geometry/settings must still pass the ordinary verification guard.
    """
    from math import isfinite
    from collections import Counter
    from .core import ContactSurface,ContactInteraction
    declarations={p.participant.name:p for p in domain.participants}
    shells=tuple(n for n,p in declarations.items() if type(p.participant) is ContactSurface)
    rigids=tuple(n for n in declarations if n not in shells)
    areas={}
    reference=contract['reference']
    if type(reference) is not list or len(reference)!=len(declarations):
        _error('invalid_persistence','complete reference participant inventory required')
    face_keys={'element_id','node_ids','formulation','mesh_reference_positions_m','reference_positions_m',
               'physical_director','physical_offset_m','reference_area_m2'}
    for row,declared in zip(reference,domain.participants):
        surface=declared.participant
        expected={'name','body_id','side','faces'} if type(surface) is ContactSurface else {'name','body_id','rigid'}
        if type(row) is not dict or set(row)!=expected or row['name']!=surface.name or row['body_id']!=declared.body_id:
            _error('invalid_persistence','reference declaration order/identity/schema changed')
        if type(surface) is not ContactSurface:
            if _canonical(row['rigid'])!=_canonical(asdict(surface)):
                _error('invalid_persistence','rigid reference properties changed')
            continue
        faces=row['faces']
        if row['side']!=surface.side or type(faces) is not list or [f.get('element_id') for f in faces]!=list(surface.element_ids):
            _error('invalid_persistence','complete canonical shell reference inventory required')
        face_areas=[]
        for f in faces:
            if type(f) is not dict or set(f)!=face_keys:
                _error('invalid_persistence','unknown/missing reference face fields')
            ids=f['node_ids'];points=f['reference_positions_m'];mesh=f['mesh_reference_positions_m']
            if (type(f['element_id']) is not int or type(ids) is not list or len(ids) not in (3,4)
                    or any(type(n) is not int or n<1 for n in ids) or len(set(ids))!=len(ids)
                    or type(f['formulation']) is not str or not f['formulation']):
                _error('invalid_persistence','invalid stored face inventory')
            for matrix in (points,mesh):
                if (type(matrix) is not list or len(matrix)!=len(ids)
                        or any(type(v) is not list or len(v)!=3 or any(type(x) not in (int,float) or not isfinite(x) for x in v) for v in matrix)):
                    _error('invalid_persistence','invalid stored face coordinates')
            director=f['physical_director']
            if (type(director) is not list or len(director)!=3 or any(type(x) not in (int,float) or not isfinite(x) for x in director)
                    or abs(sum(x*x for x in director)-1)>64*2.220446049250313e-16
                    or type(f['physical_offset_m']) not in (int,float) or not isfinite(f['physical_offset_m'])
                    or type(f['reference_area_m2']) not in (int,float) or not isfinite(f['reference_area_m2']) or f['reference_area_m2']<=0):
                _error('invalid_persistence','invalid stored physical skin/area')
            face_areas.append(f['reference_area_m2'])
        areas[surface.name]=float(median(face_areas))
    if contract['analysis_control'] not in ('force','displacement','contact_path'):
        _error('invalid_persistence','unsupported stored analysis control')
    excluded=set(domain.excluded_pairs)
    if any(not set(key).intersection(shells) for key in excluded):
        _error('invalid_persistence','unused rigid-rigid exclusion')
    complete_keys=sorted(tuple(sorted(k)) for k in itertools.chain(itertools.combinations(shells,2),itertools.product(shells,rigids)))
    count=len(complete_keys)-len(excluded)
    if count<1 or count>domain.maximum_pairs:
        _error('invalid_persistence','complete surviving pair count violates domain cap')
    provenance=contract['explicit_input_provenance']
    if _canonical(sorted(provenance,key=_canonical))!=_canonical(provenance):
        _error('invalid_persistence','explicit provenance order is noncanonical')
    available=Counter(_canonical(v) for v in provenance)
    required=Counter(_canonical(asdict(v)) for v in domain.pair_overrides)
    if any(available[v]<n for v,n in required.items()):
        _error('invalid_persistence','domain override missing from explicit provenance')
    overrides={};names={}
    for record in provenance:
        pair=ContactInteraction(**record);key=tuple(sorted((pair.surface,pair.target)))
        if len(set(key))!=2 or not set(key)<=declarations.keys():
            _error('invalid_persistence','explicit pair references invalid participants')
        if key in excluded:continue
        if not set(key).intersection(shells) or (key in overrides and overrides[key]!=pair) or (pair.name in names and names[pair.name]!=key):
            _error('invalid_persistence','explicit pair precedence/conflict invalid')
        overrides[key]=pair;names[pair.name]=key
    if _canonical([asdict(overrides[k]) for k in sorted(overrides)])!=_canonical(contract['explicit_pairs']):
        _error('invalid_persistence','effective explicit overrides differ from recorded provenance')
    expected=[]
    for key in complete_keys:
        a,b=(declarations[n] for n in key)
        if key in excluded:
            expected.append(AutomaticContactPairSelection(key,'excluded',reason='explicit exclusion precedence'));continue
        if a.body_id==b.body_id:
            _error('invalid_persistence','unexcluded self-contact is unsupported')
        if key in overrides:
            pair=overrides[key]
            if pair.surface not in areas:_error('invalid_persistence','explicit source must be deformable')
            source,target=pair.surface,pair.target;original=pair.name
            reason='explicit override preserves orientation and physical properties'
        else:
            source=(_coarser_source(key,areas) if all(n in shells for n in key) else next(n for n in key if n in shells))
            target=next(n for n in key if n!=source);original=None
            reason=('coarser reference median face area; participant ID breaks exact ties' if all(n in shells for n in key)
                    else 'deformable integration side against rigid target')
        expected.append(AutomaticContactPairSelection(key,'resolved',source,target,original,None,reason,areas.get(source),areas.get(target)))
    outside=len(rigids)*(len(rigids)-1)//2
    if outside==1:expected.append(AutomaticContactPairSelection(rigids,'outside_family',reason='rigid-rigid is not automatic contact'))
    if (type(contract['outside_family_pair_count']) is not int or contract['outside_family_pair_count']!=outside
            or _canonical([asdict(r) for r in expected])!=_canonical(contract['selections'])):
        _error('invalid_persistence','complete pair/coarser/explicit selection policy changed')


def _face_area(points):
    """Actual reference map: affine S3 or bilinear Q4 Jacobian integral."""
    import numpy as np
    from fractions import Fraction
    p=np.asarray(points,dtype=float)
    if p.shape not in ((3,3),(4,3)) or not np.all(np.isfinite(p)):
        _error('invalid_reference_geometry','three or four finite reference corners required')
    exact=tuple(tuple(Fraction(float(x)) for x in row) for row in p)
    cross=lambda a,b:(a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    dot=lambda a,b:sum(x*y for x,y in zip(a,b))
    if p.shape==(3,3):
        normal=cross(tuple(b-a for a,b in zip(exact[0],exact[1])),tuple(b-a for a,b in zip(exact[0],exact[2])))
        value=float(np.sqrt(float(dot(normal,normal)/4)))
    elif p.shape==(4,3):
        normal=cross(tuple(b-a for a,b in zip(exact[0],exact[1])),tuple(b-a for a,b in zip(exact[0],exact[3])))
        planar=dot(normal,tuple(b-a for a,b in zip(exact[0],exact[2])))==0
        dx=tuple((-a+b+c-d)/4 for a,b,c,d in zip(*exact))
        dy=tuple((-a-b+c+d)/4 for a,b,c,d in zip(*exact))
        jac=cross(dx,dy)
        for xi in (-1,1):
            for eta in (-1,1):
                cx=tuple((-(1-eta)*a+(1-eta)*b+(1+eta)*c-(1+eta)*d)/4 for a,b,c,d in zip(*exact))
                cy=tuple((-(1-xi)*a-(1+xi)*b+(1+xi)*c+(1-xi)*d)/4 for a,b,c,d in zip(*exact))
                if dot(cross(cx,cy),jac)<=0:
                    _error('invalid_reference_geometry','Q4 reference Jacobian folds or degenerates')
        if planar:
            # For a valid planar bilinear map the signed Jacobian is affine
            # and has fixed orientation. Integrating it gives four times the
            # centre Jacobian; area_squared is evaluated in exact coordinates.
            value=float(np.sqrt(float(16*dot(jac,jac))))
        else:
            nodes,weights=np.polynomial.legendre.leggauss(16)
            value=0.
            for xi,wx in zip(nodes,weights):
                for eta,wy in zip(nodes,weights):
                    dx=np.array((-(1-eta),1-eta,1+eta,-(1+eta)))/4
                    dy=np.array((-(1-xi),-(1+xi),1+xi,1-xi))/4
                    value+=float(wx*wy*np.linalg.norm(np.cross(dx@p,dy@p)))
    else:
        _error('unsupported_shell','only exact Q4 and S3 reference faces are supported')
    if not np.isfinite(value) or value<=0:
        _error('invalid_reference_geometry','reference face area must be finite and positive')
    return value


def _reference(model,participants):
    import numpy as np
    from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
    from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
    from .core import ContactSurface
    from .q4_skin_work import Q4SkinWorkAdapter
    from .s3_skin_work import S3SkinWorkAdapter
    rows=[];areas={};node_bodies={}
    for declared in participants:
        surface=declared.participant
        if type(surface) is not ContactSurface:
            rows.append(dict(name=surface.name,body_id=declared.body_id,rigid=asdict(surface)))
            continue
        if surface.beam_radius is not None or surface.beam_end_caps!='none':
            _error('unsupported_beam','automatic beam/native domains require separate feature ownership')
        faces=[]
        for eid in sorted(surface.element_ids):
            element=model.mesh.elements.get(eid)
            if type(element) not in (QualifiedE4PLShellElement,NativeParityE4PLS3V2DShellElement):
                _error('unsupported_shell',f'element {eid} is not exact qualified Q4 or S3 V2D')
            nodes=tuple(element.node_ids)
            for node in nodes:
                if node in node_bodies and node_bodies[node]!=declared.body_id:
                    _error('body_topology_conflict','shared material nodes cannot have different declared body identities')
                node_bodies[node]=declared.body_id
            mesh_points=np.asarray(element.get_node_coordinates(model.mesh),dtype=float)
            adapter_type=Q4SkinWorkAdapter if type(element) is QualifiedE4PLShellElement else S3SkinWorkAdapter
            try:
                adapter=adapter_type.from_model(model,1,eid,surface.side)
            except ValueError as exc:
                _error('unsupported_reference_skin',f'element {eid}: {exc}')
            points=adapter.reference_positions
            if not np.all(np.isfinite(points)) or not np.all(np.isfinite(mesh_points)):
                _error('invalid_reference_geometry','nonfinite shell reference coordinates')
            faces.append(dict(element_id=eid,node_ids=nodes,formulation=str(element.formulation_id),
                              mesh_reference_positions_m=mesh_points.tolist(),reference_positions_m=points.tolist(),
                              physical_director=adapter.director.tolist(),physical_offset_m=adapter.physical_offset,
                              reference_area_m2=_face_area(points)))
        areas[surface.name]=float(median(face['reference_area_m2'] for face in faces))
        rows.append(dict(name=surface.name,body_id=declared.body_id,side=surface.side,faces=faces))
    return rows,areas


def _guard_model(model):
    from anyfesolver.fe_core import FEModel
    from anyfesolver.current_state_tangent import require_exact_qualified_component_lifecycle_api
    if type(model) is not FEModel:
        _error('invalid_model','exact FEModel required')
    require_exact_qualified_component_lifecycle_api(model,context='automatic domain reference observation')


def _coarser_source(names,areas):
    return min(names,key=lambda name:(-areas[name],name))


def resolve_automatic_contact_domain(model,domain,settings=None,*,explicit_pairs=(),analysis_control='force'):
    """Compile every eligible surviving pair once, then validate execution.

    Rigid-rigid is outside this domain family and explicitly reported. Every
    shell-containing cross-body pair survives unless explicitly excluded.
    Shared-body shell pairs refuse rather than claim self-contact support.
    """
    from .core import ContactSurface,ContactInteraction,ContactDefinition,ContactSolverSettings,validate_general_contact
    if type(domain) is not AutomaticContactDomain:
        _error('invalid_domain','exact AutomaticContactDomain required')
    settings=ContactSolverSettings() if settings is None else settings
    if type(settings) is not ContactSolverSettings:
        _error('invalid_settings','exact ContactSolverSettings required')
    _guard_model(model)
    declarations={p.participant.name:p for p in domain.participants}
    shell_names=tuple(name for name,p in declarations.items() if type(p.participant) is ContactSurface)
    rigid_names=tuple(name for name,p in declarations.items() if type(p.participant) is not ContactSurface)
    excluded=set(domain.excluded_pairs)
    for key in excluded:
        if not set(key).intersection(shell_names):
            _error('unused_exclusion','rigid-rigid exclusions are outside this domain family')
    count=len(shell_names)*len(rigid_names)+len(shell_names)*(len(shell_names)-1)//2-len(excluded)
    if count>domain.maximum_pairs:
        _error('pair_cap_exceeded',f'complete surviving pair count {count} exceeds {domain.maximum_pairs}; no pairs dropped')
    if count<1:
        _error('empty_domain','no executable shell-containing pairs remain')
    reference,areas=_reference(model,domain.participants)
    overrides={};names={};explicit_inputs=[]
    for interaction in (*domain.pair_overrides,*tuple(explicit_pairs)):
        if type(interaction) is not ContactInteraction or interaction.surface==interaction.target or not {interaction.surface,interaction.target}<=declarations.keys():
            _error('invalid_override','every explicit pair must reference two distinct declared participants')
        key=tuple(sorted((interaction.surface,interaction.target)))
        explicit_inputs.append(asdict(interaction))
        if key in excluded:
            continue
        if not set(key).intersection(shell_names):
            _error('unsupported_rigid_pair','rigid-rigid overrides are outside this domain family')
        if key in overrides and overrides[key]!=interaction:
            _error('override_conflict','equal-precedence pair overrides conflict')
        if interaction.name in names and names[interaction.name]!=key:
            _error('override_conflict','explicit interaction names must be unique')
        overrides[key]=interaction;names[interaction.name]=key
    rows=[];interactions=[];used=set()
    keys=itertools.chain(itertools.combinations(shell_names,2),itertools.product(shell_names,rigid_names))
    for first,second in sorted(tuple(sorted(key)) for key in keys):
        key=(first,second);a,b=(declarations[x] for x in key)
        if key in domain.excluded_pairs:
            rows.append(AutomaticContactPairSelection(key,'excluded',reason='explicit exclusion precedence'))
            continue
        shell_a,shell_b=type(a.participant) is ContactSurface,type(b.participant) is ContactSurface
        if not shell_a and not shell_b:
            if key in overrides:
                _error('unsupported_rigid_pair','rigid-rigid overrides are outside this domain family')
            rows.append(AutomaticContactPairSelection(key,'outside_family',reason='rigid-rigid is not automatic contact'))
            continue
        if a.body_id==b.body_id:
            _error('unsupported_self_contact','shared-body/self-contact pair requires explicit exclusion or a future owner')
        if key in overrides:
            pair=overrides[key]
            if pair.surface not in areas:
                _error('invalid_explicit_role','explicit integration side must be deformable; orientation is not reversed')
            reason='explicit override preserves orientation and physical properties'
        else:
            if shell_a and shell_b:
                source=_coarser_source(key,areas);target=second if source==first else first
                reason='coarser reference median face area; participant ID breaks exact ties'
            else:
                source,target=(first,second) if shell_a else (second,first)
                reason='deformable integration side against rigid target'
            pair=ContactInteraction('generated',source,target,domain.defaults.normal_stiffness_n_per_m3,
                                    domain.defaults.gap_tolerance_m,domain.defaults.normal_law)
        interactions.append((key,pair));used.update(key)
        rows.append(AutomaticContactPairSelection(key,'resolved',pair.surface,pair.target,
                    pair.name if key in overrides else None,None,reason,areas.get(pair.surface),areas.get(pair.target)))
    if not interactions:
        _error('empty_domain','no executable shell-containing pairs remain')
    outside_count=len(rigid_names)*(len(rigid_names)-1)//2
    if outside_count==1:
        rows.append(AutomaticContactPairSelection(rigid_names,'outside_family',reason='rigid-rigid is not automatic contact'))
    contract=dict(policy=POLICY,area_policy=AREA_POLICY,domain_name=domain.name,
                  participants=[dict(body_id=p.body_id,participant=asdict(p.participant)) for p in domain.participants],
                  defaults=asdict(domain.defaults),excluded_pairs=domain.excluded_pairs,maximum_pairs=domain.maximum_pairs,
                  explicit_pairs=[asdict(overrides[key]) for key in sorted(overrides)],settings=_settings_contract(settings),
                  domain_pair_overrides=sorted({_canonical(asdict(pair)) for pair in domain.pair_overrides}),
                  explicit_input_provenance=sorted(explicit_inputs,key=_canonical),
                  persisted_domain=domain.to_dict(),outside_family_pair_count=outside_count,
                  analysis_control=analysis_control,reference=reference,selections=[asdict(r) for r in rows])
    canonical=_canonical(contract);identity=hashlib.sha256(canonical.encode()).hexdigest()
    compiled=[];final_rows=[]
    for key,pair in interactions:
        suffix=hashlib.sha256(_canonical(key).encode()).hexdigest()
        compiled.append(replace(pair,name=f'automatic:{identity}:{suffix}'))
    lookup={tuple(sorted((pair.surface,pair.target))):pair.name for pair in compiled}
    final_rows=tuple(replace(row,compiled_interaction_name=lookup[row.participants])
                     if row.disposition=='resolved' else row for row in rows)
    surfaces=tuple(replace(p.participant,element_ids=tuple(sorted(p.participant.element_ids)))
                   for p in domain.participants if p.participant.name in used and type(p.participant) is ContactSurface)
    rigid=tuple(p.participant for p in domain.participants if p.participant.name in used and type(p.participant) is not ContactSurface)
    definition=ContactDefinition(surfaces,rigid,tuple(compiled))
    report=validate_general_contact(model,definition,settings,analysis_control=analysis_control)
    if not report.valid:
        _error('unsupported_execution','resolved orientation is not executable: '+'; '.join(report.issues))
    _guard_model(model)
    after,_=_reference(model,domain.participants)
    if _canonical(after)!=_canonical(reference):
        _error('stale_resolution','reference geometry changed during compilation')
    return AutomaticContactResolution(definition,final_rows,identity,canonical,_canonical(reference),
                                      _canonical(_settings_contract(settings)),_canonical(asdict(definition)),outside_count)


def _compiled_from_contract(contract,identity):
    """Reconstruct selected roles/properties, never recompute pairing or roles."""
    from .core import ContactSurface,RigidContactBody,ContactInteraction,ContactDefinition
    originals={tuple(sorted((p['surface'],p['target']))):p for p in contract['explicit_pairs']}
    pairs=[];records=[];used=set()
    for row in contract['selections']:
        row=dict(row);row['participants']=tuple(row['participants'])
        if row['disposition']=='resolved':
            key=row['participants'];used.update(key)
            name=f'automatic:{identity}:{hashlib.sha256(_canonical(key).encode()).hexdigest()}'
            if key in originals:pair=ContactInteraction(**originals[key])
            else:
                defaults=contract['defaults']
                pair=ContactInteraction('generated',row['source'],row['target'],defaults['normal_stiffness_n_per_m3'],defaults['gap_tolerance_m'],defaults['normal_law'])
            pairs.append(replace(pair,name=name));row['compiled_interaction_name']=name
        records.append(AutomaticContactPairSelection(**row))
    surfaces=[];bodies=[]
    for declaration in contract['participants']:
        data=declaration['participant']
        if data['name'] not in used:continue
        if 'element_ids' in data:surfaces.append(ContactSurface(**data))
        else:bodies.append(_restore_rigid_component(data))
    return ContactDefinition(tuple(surfaces),tuple(bodies),tuple(pairs)),tuple(records)


def verify_automatic_contact_resolution(model,domain,resolution,settings=None):
    """Guard the fixed reference selection immediately before ordinary solve.

    This verifies provenance and reference binding, never silently re-pairs.
    The existing solver still owns material, geometry and analysis admission.
    """
    from .core import ContactSolverSettings
    if type(domain) is not AutomaticContactDomain or type(resolution) is not AutomaticContactResolution:
        _error('invalid_resolution','exact domain and resolution records required')
    settings=ContactSolverSettings() if settings is None else settings
    if type(settings) is not ContactSolverSettings:
        _error('invalid_settings','exact ContactSolverSettings required')
    _guard_model(model)
    contract=json.loads(resolution.canonical_contract_json)
    if domain.to_dict()!=contract['persisted_domain']:
        _error('stale_resolution','automatic domain declaration changed')
    declaration=json.loads(_canonical(dict(domain_name=domain.name,participants=[dict(body_id=p.body_id,participant=asdict(p.participant)) for p in domain.participants],
                     defaults=asdict(domain.defaults),excluded_pairs=domain.excluded_pairs,maximum_pairs=domain.maximum_pairs)))
    if any(declaration[key]!=contract[key] for key in declaration):
        _error('stale_resolution','automatic domain declaration changed')
    # Domain overrides remain bound even when exclusion took precedence.
    if sorted({_canonical(asdict(pair)) for pair in domain.pair_overrides})!=contract['domain_pair_overrides']:
        _error('stale_resolution','automatic domain overrides changed')
    reference,_=_reference(model,domain.participants)
    expected,records=_compiled_from_contract(contract,resolution.identity)
    if (_canonical(asdict(expected))!=resolution.definition_json or resolution.selections!=records
            or resolution.outside_family_pair_count!=contract['outside_family_pair_count']
            or _canonical(contract['reference'])!=resolution.reference_json
            or _canonical(contract['settings'])!=resolution.settings_json):
        _error('stale_resolution','compiled definition/selection/metadata differs from canonical contract')
    if (type(settings) is not ContactSolverSettings or _canonical(_settings_contract(settings))!=resolution.settings_json
            or _canonical(reference)!=resolution.reference_json or _canonical(asdict(resolution.definition))!=resolution.definition_json
            or hashlib.sha256(resolution.canonical_contract_json.encode()).hexdigest()!=resolution.identity):
        _error('stale_resolution','reference geometry/topology/formulation, settings or compiled definition changed')
    _guard_model(model)
    return resolution.definition


__all__=['AutomaticContactDomainError','AutomaticContactParticipant','AutomaticContactDefaults',
         'AutomaticContactDomain','AutomaticContactPairSelection','AutomaticContactResolution',
         'resolve_automatic_contact_domain','verify_automatic_contact_resolution']
