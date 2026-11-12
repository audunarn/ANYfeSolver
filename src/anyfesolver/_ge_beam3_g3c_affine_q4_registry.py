"""Finite, recipe-bound private Q4 numerical contexts; no general admission claim."""
from dataclasses import dataclass
from hashlib import sha256
import math
import numpy as np
from ._ge_beam3_g3c_local_shell import owned, canonical
from ._ge_beam3_pose_joint import _exp_terms

REGISTRY = 'GE_BEAM3_Q4_AFFINE_REGISTERED_CONSTRUCTIONS_V1'
SHAPES = ('SQUARE', 'RECTANGLE', 'RHOMBUS')
SCALES = ('0.01', '1', '10')
POSES = ('ZERO', 'MEMBRANE', 'BENDING', 'SHEAR', 'CHECKERBOARD', 'MIXED')
VARIANTS = ('BASE', 'SHUFFLED_INSERTION', 'RENUMBERED', 'CONNECTIVITY_REVERSED', 'PROPER_GLOBAL_TRANSFORM')

@dataclass(frozen=True)
class Construction:
    construction_id: str
    node_ids: tuple
    coordinates: np.ndarray
    normal: np.ndarray
    material_direction: np.ndarray
    director_polarity: int
    recipe: bytes
    recipe_sha256: str
    canonical_coordinates: np.ndarray
    permutation: tuple
    passive_rotation: np.ndarray
    element_id: int

def base_ids():
    return tuple(s+'::'+v for s in SHAPES for v in SCALES)

def graph_ids():
    return tuple(g+'::'+v for g in ('J_Q4_PAIR', 'J_MULTIFAMILY_LOOP') for v in VARIANTS)

def _exp(v):
    return _exp_terms(np.asarray(v, dtype=np.float64))[0]

def construction(construction_id):
    if type(construction_id) is not str:
        raise ValueError('exact registered construction ID required')
    parts = construction_id.split('::')
    if len(parts) not in (2, 3):
        raise ValueError('unregistered construction')
    key, variant = parts[:2]
    nodes = (101, 102, 103, 104); element = 1
    permutation = (0, 1, 2, 3); W = np.eye(3); shift = np.zeros(3); polarity = 1
    if key in SHAPES and variant in SCALES:
        xy = {'SQUARE': ((0,0),(1,0),(1,1),(0,1)),
              'RECTANGLE': ((0,0),(2,0),(2,1),(0,1)),
              'RHOMBUS': ((-8/5,-4/5),(2/5,-4/5),(8/5,4/5),(-2/5,4/5))}[key]
        ref = np.array([(x,y,1/8) for x,y in xy], dtype=np.float64)*float(variant)
        if len(parts) == 3:
            if variant != '1': raise ValueError('transport registered only at unit scale')
            suffix = parts[2]
            if suffix.startswith('D4:') and suffix[3:] in tuple(str(i) for i in range(8)):
                k = int(suffix[3:]); base = (0,1,2,3) if k < 4 else (0,3,2,1)
                permutation = tuple((i+k%4)%4 for i in base)
            elif suffix in ('DIRECTOR:-1', 'DIRECTOR:1'):
                polarity = int(suffix.split(':')[1])
            elif suffix == 'PASSIVE':
                W = _exp((.31,-.22,.17))
                L = max(np.linalg.norm(ref[(i+1)%4]-ref[i]) for i in range(4))
                shift = np.array((2.,-3.,1.))*L
            else: raise ValueError('unregistered reference transport')
    elif key in ('J_Q4_PAIR', 'J_MULTIFAMILY_LOOP') and variant in VARIANTS and len(parts) == 2:
        if key == 'J_Q4_PAIR':
            ref = np.array(((0,0,.125),(1,0,.125),(1,1,.125),(0,1,.125)), dtype=np.float64); element = 11
        else:
            ref = np.array(((0,.5,.125),(2,.5,.125),(2,1.5,.125),(0,1.5,.125)), dtype=np.float64)
            nodes = (301,302,303,304); element = 13
        if variant == 'RENUMBERED':
            nodes = tuple(10000+7*n for n in nodes); element = 20000+5*element
        elif variant == 'CONNECTIVITY_REVERSED': permutation = (0,3,2,1)
        elif variant == 'PROPER_GLOBAL_TRANSFORM':
            W = np.array(((0.,-1.,0.),(1.,0.,0.),(0.,0.,1.))); shift = np.array((2.,-3.,1.))
    else:
        raise ValueError('foreign reference geometry is not admitted')
    X = owned((ref@W.T+shift)[list(permutation)])
    normal = owned(W@np.array((0.,0.,1.))); direction = owned(W@np.array((1.,0.,0.)))
    ids = tuple(nodes[i] for i in permutation)
    body = canonical(dict(registry=REGISTRY, construction_id=construction_id, node_ids=ids,
        coordinates=X, coordinate_bytes_sha256=sha256(X.tobytes()).hexdigest(), normal=normal,
        material_direction=direction, director_polarity=polarity, element_id=element,
        permutation=permutation, ideal_recipe='CONTRACT_SECTION_4_NO_SNAPPING'))
    return Construction(construction_id, ids, X, normal, direction, polarity, body,
                        sha256(body).hexdigest(), owned(ref), permutation, owned(W), element)

def pose(construction_id, pose_id, amplitude=1.0):
    c = construction(construction_id)
    if pose_id not in POSES or type(amplitude) not in (int,float) or not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError('registered finite positive pose amplitude required')
    if amplitude != 1 and (pose_id != 'MIXED' or amplitude not in (1e-6,1e-3) or construction_id not in tuple(s+'::1' for s in SHAPES)):
        raise ValueError('unregistered tiny-response context')
    X = c.canonical_coordinates; centered = X-X.mean(axis=0)
    L = max(np.linalg.norm(X[(i+1)%4]-X[i]) for i in range(4))
    u = np.zeros((4,6)); rv = np.zeros((4,3))
    for i,(x,y,z) in enumerate(centered):
        if pose_id == 'MEMBRANE': u[i,:3] = (.012*x+.004*y,-.003*x+.007*y,0)
        elif pose_id == 'BENDING':
            u[i,2] = .009*(x*x+2*y*y+x*y)/L; rv[i] = (-.018*y/L,.018*x/L,0)
        elif pose_id == 'SHEAR': u[i,2] = .011*x-.006*y
        elif pose_id == 'CHECKERBOARD': u[i,2] = .008*L*(1,-1,1,-1)[i]
        elif pose_id == 'MIXED':
            u[i,:3] = .009*L*np.sin(6*i+np.arange(3)+.4)
            rv[i] = np.array((.09,-.04,.06))+.004*i*np.array((1,2,-1))
            u[i,3:] = .009*np.sin(6*i+np.arange(3,6)+.4)
    u *= amplitude; rv *= amplitude
    qa = np.array([_exp(v) for v in rv]); W = c.passive_rotation
    u[:,:3] = u[:,:3]@W.T; u[:,3:] = u[:,3:]@W.T
    qa = np.array([W@q@W.T for q in qa])
    return owned(u[list(c.permutation)].reshape(24)), owned(qa[list(c.permutation)])

def common_motion(displacement, accepted_rotations, coordinates, index):
    if type(index) is not int or index not in range(4): raise ValueError('registered motion index required')
    W = _exp(((0,0,0),(.4,-.3,.2),(math.pi,0,0),(0,1.4*math.pi,0))[index])
    X = np.asarray(coordinates); u = np.asarray(displacement).reshape(4,6)
    L = max(np.linalg.norm(X[(i+1)%4]-X[i]) for i in range(4))
    result = np.empty((4,6)); result[:,:3] = (X+u[:,:3])@W.T+np.array((2,-3,1))*L-X
    result[:,3:] = u[:,3:]@W.T
    return owned(result.reshape(24)), owned([W@q for q in accepted_rotations]), owned(W)

def rebase(displacement, accepted_rotations):
    u = np.array(displacement, copy=True).reshape(4,6)
    qa = owned([_exp(row[3:])@q for row,q in zip(u,accepted_rotations)])
    u[:,3:] = 0
    return owned(u.reshape(24)), qa
