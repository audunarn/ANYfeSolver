"""Explicit spatial-axis nodal couples for the private native beam driver.

Virtual work is m dot delta_theta for spatial multiplicative rotations.
A constant spatial couple is not a globally conservative SO(3) load. It has
zero derivative in spatial residual components, not a fictitious -m dot log Q
potential. No existing beam, shell or public load route is changed here.
"""
from dataclasses import dataclass
from math import isfinite


POLICY = 'GE_BEAM3_CONSTANT_SPATIAL_NODAL_MOMENTS_NONCONSERVATIVE_V1'


@dataclass(frozen=True)
class SpatialNodalMoments:
    rows: tuple  # Sorted (node, Mx, My, Mz), force units times length.

    def __post_init__(self):
        if type(self.rows) is not tuple or not 1 <= len(self.rows) <= 64:
            raise ValueError('one to 64 explicit spatial nodal moment rows required')
        ids = []
        for row in self.rows:
            if (type(row) is not tuple or len(row) != 4 or type(row[0]) is not int or row[0] <= 0
                    or any(type(x) is not float or not isfinite(x) for x in row[1:])):
                raise ValueError('positive node and three finite binary64 moment components required')
            ids.append(row[0])
        if ids != sorted(set(ids)) or not any(x != 0. for row in self.rows for x in row[1:]):
            raise ValueError('unique ascending nodes and a nonzero moment pattern required')

    def descriptor(self):
        return dict(policy=POLICY, rows=self.rows, axes='FIXED_SPATIAL',
            virtual_work='M_DOT_SPATIAL_MULTIPLICATIVE_VIRTUAL_ROTATION',
            conservative_potential=False, spatial_residual_derivative='ZERO',
            conservative_modal_buckling_authorized=False, production_qualified=False)

    def potential(self, *args, **kwargs):
        raise ValueError('constant spatial nodal moments have no general conservative SO(3) potential')
