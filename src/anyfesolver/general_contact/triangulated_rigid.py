"""Explicit rigid mesh data; the first executable profile is planar S3 contact."""
from dataclasses import dataclass
from fractions import Fraction
import numpy as np


@dataclass(frozen=True)
class TriangulatedRigidContactBody:
    name: str
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, tuple[int, int, int]], ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("rigid mesh needs a name")
        vertices = tuple(tuple(float(x) for x in p) for p in self.vertices)
        faces = tuple((i, tuple(nodes)) for i, nodes in self.faces)
        if not 3 <= len(vertices) <= 256 or not 1 <= len(faces) <= 256:
            raise ValueError("rigid mesh exceeds the explicit 256 vertex/face budget")
        if any(len(p) != 3 or not all(np.isfinite(p)) for p in vertices):
            raise ValueError("rigid vertices need finite xyz coordinates")
        seen, triangles, edges = set(), set(), {}
        for fid, nodes in faces:
            if type(fid) is not int or fid <= 0 or fid in seen:
                raise ValueError("rigid face IDs must be distinct positive integers")
            if len(nodes) != 3 or len(set(nodes)) != 3 or any(type(j) is not int or not 0 <= j < len(vertices) for j in nodes):
                raise ValueError("invalid rigid triangle indices")
            key = tuple(sorted(nodes))
            if key in triangles:
                raise ValueError("duplicate rigid triangle")
            seen.add(fid); triangles.add(key)
            a, b, c = (np.asarray(vertices[j]) for j in nodes)
            if not np.linalg.norm(np.cross(b-a, c-a)) > 0:
                raise ValueError("degenerate rigid triangle")
            for a, b in zip(nodes, nodes[1:]+nodes[:1]):
                edge = tuple(sorted((a,b)))
                old = edges.setdefault(edge, [])
                if len(old) == 2 or (a,b) in old:
                    raise ValueError("nonmanifold or inverted shared rigid edge")
                old.append((a,b))
        object.__setattr__(self, 'vertices', vertices)
        object.__setattr__(self, 'faces', tuple(sorted(faces)))

    @property
    def kind(self):
        return 'triangulated'

    @property
    def path_translation(self):
        return None

    def planar_positions(self):
        """Reject nonplanarity using the exact stored binary64 coordinates."""
        p = [[Fraction(x) for x in v] for v in self.vertices]
        nodes = self.faces[0][1]
        a,b,c = (p[j] for j in nodes)
        u,v = [b[i]-a[i] for i in range(3)], [c[i]-a[i] for i in range(3)]
        n = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
        if any(sum(n[i]*(q[i]-a[i]) for i in range(3)) != 0 for q in p):
            raise ValueError("nonplanar rigid mesh is outside the planar mechanical profile")
        result = np.asarray(self.vertices)
        if len(set(self.vertices))!=len(self.vertices):
            raise ValueError('coincident rigid vertices have unresolved edge ownership')
        normal = np.asarray([float(x) for x in n])
        for _, ids in self.faces:
            q = result[list(ids)]
            if np.dot(np.cross(q[1]-q[0],q[2]-q[0]),normal) <= 0:
                raise ValueError("rigid faces have inconsistent orientation")
        # Exact planar polygon clipping detects every positive overlap, even
        # below the floating integration engine's area guard. Shared edges
        # and vertices have zero measure and carry no independent potential.
        drop=max(range(3),key=lambda i:abs(n[i]))
        axes=[i for i in range(3) if i!=drop]
        def orient(a,b,c):
            return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        triangles=[[tuple(p[j][i] for i in axes) for j in ids] for _,ids in self.faces]
        triangles=[t if orient(*t)>0 else list(reversed(t)) for t in triangles]
        for i,first in enumerate(triangles):
            for second in triangles[i+1:]:
                poly=list(first)
                for a,b in zip(second,second[1:]+second[:1]):
                    if not poly: break
                    out=[];old=poly[-1];old_d=orient(a,b,old)
                    for current in poly:
                        d=orient(a,b,current)
                        if (old_d>=0)!=(d>=0):
                            fraction=old_d/(old_d-d)
                            out.append(tuple(x+fraction*(y-x) for x,y in zip(old,current)))
                        if d>=0: out.append(current)
                        old,old_d=current,d
                    poly=out
                if len(poly)>2 and sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(poly,poly[1:]+poly[:1]))>0:
                    raise ValueError('rigid triangles have intersecting positive-area interiors')
        return result
