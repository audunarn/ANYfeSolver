"""Prospective scalar preparation for 21 explicit whole-member ratios.

Reference development only: no native operator, beta, uniform qualification or
F03 execution. Each invocation owns one fresh JSON/NPZ pair and one fixed case.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
from types import MappingProxyType

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_info

from scripts.r8_bl5_projected_auxiliary_norm_dev import integrated_p1, lower_gram
from scripts.b3_e4_bl5_riesz_controls_dev import shell_scalar
from scripts.r8_bl5_projected_reference_action_dev import integrated_projected_weights
from scripts.r8_bl5_external_affine_reference_dev import shape_weights
from scripts.r8_bl5_projected_modal_envelope_dev import projected_modal_bracket

IDENTITY = 'BL5_PROJECTED_WHOLE_MEMBER_ENVELOPE_DEV_V1'
UNIFORM_IDENTITY = 'BL5_PROJECTED_WHOLE_MEMBER_UNIFORM_REFERENCE_DEV_V2'
# Retained v1 registrations bind this archived, unchanged preparation source.
LEGACY_SOURCE_SHA256 = 'd2f4bf24f68f280a2d3951c0ca1a2dc1a69cbddd14a8ce78821568c8088948d4'
RATIOS = (F(1, 8), F(1, 4), F(1, 2), F(1), F(2), F(4), F(8))
MODES = (4096, 8192, 16384)
LENGTH, WIDTH, ELL = F(1), F(1, 5), F(2, 5)
CRITERIA = MappingProxyType(dict(reference_error=1e-3, inverse_eigen_symmetry=1e-10,
    condition_budget=1e4, arithmetic_beta_margin=1e-6, variational_ceiling=(1+1e-6)**2))
EXPECTED_HELPERS = MappingProxyType({
    'r8_bl5_projected_auxiliary_norm_dev.py': '906ebf3d821a85b6d8851f2dc86fc6e0cb2e80d19eb4f70976bad0b22f5e2bbf',
    'b3_e4_bl5_riesz_controls_dev.py': 'da2ed22be805ce6901cfdfe034fe52a80089617e83849f71375948c64f724a4f',
    'r8_bl5_projected_reference_action_dev.py': 'b199b5273381009a21815184c2b665f1339ede0bc81d2f89d462fc085617202b',
    'r8_bl5_external_affine_reference_dev.py': '04c1b930258d5911c500ab3430b196e9488324ae8cafaebbfa30584420a26b08',
    'r8_bl5_projected_modal_envelope_dev.py': '6d534e8319c3b86b4660edc9a9857f35bd1a67f9a87975e1722212d31045387b',
    'b3_e4_bl5_modal_reference_dev.py': 'e1553e83596d5d371a7b2e196c000a1d9b551e361abf0942e352d54018b5712b',
    'r8_bl5_projected_ratio_norm_dev.py': '2424d4ad6f5d6c0287dbbe7321b70419ade9ac2ee0908028f3c85dc86a880a48',
})


@dataclass(frozen=True, slots=True)
class EnvelopeCase:
    nx: int
    member_ratio: F

    def __post_init__(self):
        if type(self.nx) is not int or self.nx not in (16, 32, 64):
            raise ValueError('explicit shell level 16, 32 or 64 required')
        if not isinstance(self.member_ratio, F) or self.member_ratio not in RATIOS:
            raise ValueError('one registered exact whole-member Fraction ratio required')

    @property
    def ny(self): return self.nx//4

    @property
    def halfcells(self): return int(2*self.nx*self.member_ratio)

    @property
    def span(self): return int(4*max(F(1), F(self.halfcells, self.nx)))

    @property
    def supports(self): return self.halfcells//self.span

    @property
    def fixture_case(self):
        ratio = str(self.member_ratio).replace('/', '_')
        return f'ns{self.nx}_b{ratio}'

    @property
    def fixture(self):
        return dict(nx=self.nx, ny=self.ny, halfcells=self.halfcells, supports=self.supports)

    @property
    def fixed_count(self): return 6*(self.ny+1)+5

    @property
    def counts(self):
        shell = (self.nx+1)*(self.ny+1)
        nodal = 6*(shell+self.halfcells+1)
        cells, resultants = 3*self.halfcells, 9*self.halfcells
        physical_free = nodal+cells-self.fixed_count
        return dict(shell_nodes=shell, beam_nodes=self.halfcells+1,
            physical_dofs=nodal, members=self.halfcells//2,
            retained_cell_rotations=cells, retained_resultants=resultants,
            retained_dofs=cells+resultants, full_q=nodal+cells+resultants,
            multipliers=10*self.supports, physical_free=physical_free,
            kernel=physical_free-10*self.supports)

    @property
    def supported_kkt_count(self):
        return self.counts['full_q']-self.fixed_count+self.counts['multipliers']


def all_cases():
    return tuple(EnvelopeCase(nx, ratio) for nx in (16, 32, 64) for ratio in RATIOS)


def validate_campaign(cases):
    if not isinstance(cases, tuple) or cases != all_cases():
        raise ValueError('complete immutable registered case ordering required')
    return cases


def case_from_id(fixture_case):
    for case in all_cases():
        if type(fixture_case) is str and case.fixture_case == fixture_case:
            return case
    raise ValueError('unknown envelope fixture case')


def _case(case):
    if type(case) is not EnvelopeCase or case not in all_cases():
        raise ValueError('explicit immutable registered EnvelopeCase required')
    return case


def projection(case):
    case = _case(case)
    p = np.zeros((case.halfcells, 2*case.supports))
    for cell in range(case.halfcells):
        support = cell//case.span
        midpoint = F(2*cell+1, 2*case.halfcells)
        p[cell, 2*support] = 1.
        p[cell, 2*support+1] = float(2*(midpoint-F(support, case.supports))*case.supports-1)
    return p


def density_mass_diagonal(case):
    case = _case(case)
    return np.tile((float(F(1, case.supports)),
                    float((1-F(1, case.span**2))/(3*case.supports))), case.supports)


def quarter_witness(case):
    case = _case(case)
    intervals = F(case.nx, case.supports)
    if (intervals/4).denominator != 1:
        raise ValueError('quarter witness not representable on shell trace')
    return tuple((int(j*intervals+intervals/4), int(j*intervals+3*intervals/4))
                 for j in range(case.supports))


def exact_boundary(case):
    case = _case(case)
    exact = np.zeros((case.nx+1, 2*case.supports))
    for support in range(case.supports):
        weights, _ = integrated_projected_weights(F(support, case.supports),
            F(support+1, case.supports), case.nx, case.halfcells)
        for degree, row in enumerate(weights):
            for node, value in row.items():
                exact[node, 2*support+degree] = float(value)
    return exact


def cell_action(case):
    """Complete P0 to shell-P1 action without an Nhalf squared identity."""
    case = _case(case)
    out = np.zeros((case.nx+1, case.halfcells))
    cuts = sorted({F(i, case.nx) for i in range(case.nx+1)} |
                  {F(i, case.halfcells) for i in range(case.halfcells+1)})
    for left, right in zip(cuts[:-1], cuts[1:]):
        cell = int((left+right)*case.halfcells/2)
        a, b = shape_weights(left, case.nx, 0), shape_weights(right, case.nx, 0)
        for node in set(a) | set(b):
            out[node, cell] += float((right-left)*(a.get(node, F(0))+b.get(node, F(0)))/2)
    return out


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _failure_json(value):
    """Keep nonfinite diagnostics explicit; their exact arrays stay in NPZ."""
    if isinstance(value, dict):
        return {key: _failure_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_failure_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return dict(nonfinite=repr(value))
    return value


def helpers(case, *, reference='original'):
    _case(case)
    directory = Path(__file__).resolve().parent
    actual = {name: sha(directory/name) for name in EXPECTED_HELPERS}
    if actual != dict(EXPECTED_HELPERS):
        raise ValueError('scalar preparation helper identity changed')
    for function in (integrated_p1, lower_gram, shell_scalar,
                     integrated_projected_weights, shape_weights, projected_modal_bracket):
        path = Path(function.__code__.co_filename).resolve()
        if path != directory/path.name or sha(path) != actual.get(path.name):
            raise ValueError('scalar preparation imported helper path differs')
    if reference not in ('original', 'exact_uniform'):
        raise ValueError('explicit admitted reference bound required')
    if reference == 'exact_uniform':
        from scripts.r8_bl5_projected_uniform_modal_dev import uniform_projected_modal_bracket
        path = Path(uniform_projected_modal_bracket.__code__.co_filename).resolve()
        if path != directory/'r8_bl5_projected_uniform_modal_dev.py':
            raise ValueError('uniform reference imported helper path differs')
        actual[path.name] = sha(path)
    return actual


def register(case, output, *, reference='original'):
    case = _case(case)
    output = Path(output)
    arrays_path = output.with_suffix('.npz')
    if output.suffix != '.json' or output.exists() or arrays_path.exists():
        raise ValueError('fresh .json and sibling .npz outputs required')
    binding = helpers(case, reference=reference)
    modal = projected_modal_bracket
    if reference == 'exact_uniform':
        from scripts.r8_bl5_projected_uniform_modal_dev import uniform_projected_modal_bracket
        modal = uniform_projected_modal_bracket
    if any(name.split('.')[0] == 'anyfesolver' for name in sys.modules):
        raise ValueError('scalar registration requires a product-free process')
    pools = threadpool_info()
    if not pools or any(pool['num_threads'] != 1 for pool in pools):
        raise ValueError('effective scalar numerical pools must each have one thread')
    output.parent.mkdir(parents=True, exist_ok=True)
    record = dict(status='STARTED_ENVELOPE_DEVELOPMENT_PREPARATION',
        identity=IDENTITY if reference == 'original' else UNIFORM_IDENTITY,
        reference_bound=reference,
        fixture_case=case.fixture_case, fixture=case.fixture, counts=case.counts,
        fixed_count=case.fixed_count, supported_kkt_count=case.supported_kkt_count,
        coordinate_norm=dict(length_m=float(LENGTH), width_m=float(WIDTH), ell_m=float(ELL)),
        physical_ratio=dict(member_shell=str(case.member_ratio), halfcell_shell=str(F(case.halfcells, case.nx)),
            shell_intervals_per_support=case.nx//case.supports, halfcells_per_support=case.span),
        fixed_criteria=dict(CRITERIA), source_sha256=sha(__file__), helper_sha256=binding,
        runtime=dict(numpy=np.__version__, numerical_pools=pools),
        limitations='one prospective scalar development bound; no native beta, uniform envelope acceptance or F03 execution')
    with output.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    raw = {}

    def save_raw():
        # This pair was exclusively claimed above; updates retain partial raw evidence.
        with arrays_path.open('wb') as handle:
            np.savez(handle, **raw)

    try:
        with arrays_path.open('xb') as handle:
            np.savez(handle)
        length, width, ell = map(float, (LENGTH, WIDTH, ELL))
        p = projection(case)
        knots = np.linspace(0., length, case.halfcells+1)
        raw.update(projection=p, knots=knots)
        save_raw()
        xs, _, h, active = shell_scalar(length, width, ell, case.nx, case.ny)
        boundary = integrated_p1(xs, p, knots)
        full = np.zeros((case.counts['shell_nodes'], 2*case.supports))
        full[:case.nx+1] = boundary
        action = full[active]
        raw.update(scalar_action=action, scalar_boundary=boundary, scalar_active=active,
                   H_data=h.data, H_indices=h.indices, H_indptr=h.indptr, H_shape=np.array(h.shape))
        save_raw()
        g, residual, skew = lower_gram(h, action)
        expected = density_mass_diagonal(case)
        mass = p.T@p/case.halfcells
        mass_error = float(np.max(np.abs(mass-np.diag(expected))))
        d = np.diag(1/np.sqrt(expected))
        raw.update(scalar_lower=g, density_mass=mass, density_L2_scale=d)
        save_raw()
        preparation = []
        for modes in MODES:
            lower, upper, info = modal(knots, p, length, width, ell, modes)
            raw.update(reference_lower=lower, reference_upper=upper)
            save_raw()
            error = info['relative_loewner_tail_bound']
            preparation.append(dict(modes=modes, projected_relative_tail=error, modal=info))
            record['reference_preparation'] = preparation
            if error is None:
                raise ArithmeticError('candidate projected reference is not numerically positive/full rank')
            if error <= CRITERIA['reference_error']:
                break
        else:
            raise ArithmeticError('preset independent shell reference budget exhausted')
        gd, md = d@g@d, d@upper@d
        values, vectors = linalg.eigh(md, gd)
        eigen_residual = float(np.linalg.norm(md@vectors-(gd@vectors)*values)/np.linalg.norm(md@vectors))
        cs2 = float(values[-1])
        rho = np.hypot(.05, .1/2)/ell
        product2 = max(2*cs2, max(2., 1+2*rho*rho)*max(2*cs2, 1+rho*rho))
        safe = float(1/np.sqrt(product2)/np.sqrt(1.001)-CRITERIA['arithmetic_beta_margin'])
        conditions = dict(discrete=float(np.linalg.cond(gd)), reference=float(np.linalg.cond(md)))
        raw.update(scalar_eigenvalues=values, scalar_eigenvectors=vectors,
                   scaled_scalar_lower=gd, scaled_reference_upper=md)
        save_raw()
        exact = exact_boundary(case)
        endpoint_error = float(np.max(np.abs(boundary-exact)))
        witness = quarter_witness(case)
        interior_present = all(exact[left, 2*j+1] != 0 and exact[right, 2*j+1] != 0
                               for j, (left, right) in enumerate(witness))
        checker_cells = 4*case.nx
        checker_knots = np.linspace(0., length, checker_cells+1)
        checker = np.tile((1., -3., 3., -1.), case.nx)
        checker_action = integrated_p1(xs, checker[:, None], checker_knots)[:, 0]
        checker_defect = float(np.max(np.abs(checker_action[1:])))
        checker_lower, checker_upper, checker_info = modal(
            checker_knots, checker[:, None], length, width, ell, modes)
        checker_reference = float(checker_lower[0, 0])
        cells_action = cell_action(case)
        native_rank = int(np.linalg.matrix_rank(cells_action[1:]))
        duplicate_rank = int(np.linalg.matrix_rank(np.c_[action, action[:, 0]]))
        raw.update(independent_boundary=exact, cell_action=cells_action,
            checker_density=checker, checker_action=checker_action,
            checker_reference_lower=checker_lower, checker_reference_upper=checker_upper)
        save_raw()
        change = np.diag(np.geomspace(.2, 5., 2*case.supports))
        transformed = linalg.eigvalsh(change@upper@change, change@g@change)
        basis_error = float(np.max(np.abs(transformed-values))/max(1., cs2))
        unit = 1000.
        xu, _, hu, au = shell_scalar(unit*length, unit*width, unit*ell, case.nx, case.ny)
        fullu = np.zeros_like(full)
        fullu[:case.nx+1] = integrated_p1(xu, p, unit*knots)
        gu, unit_residual, unit_skew = lower_gram(hu, fullu[au])
        raw.update(unit_scalar_lower=gu, unit_scalar_action=fullu[au],
                   basis_transformed_eigenvalues=transformed)
        save_raw()
        _, unit_upper, unit_info = modal(unit*knots, p,
            unit*length, unit*width, unit*ell, modes)
        unitvalues = linalg.eigvalsh(unit_upper, gu)
        unit_error = float(np.max(np.abs(unitvalues-values))/max(1., cs2))
        raw.update(independent_boundary=exact, cell_action=cells_action,
            scalar_eigenvalues=values, scalar_eigenvectors=vectors,
            checker_density=checker, checker_action=checker_action,
            checker_reference_lower=checker_lower, checker_reference_upper=checker_upper,
            basis_transformed_eigenvalues=transformed, unit_scalar_lower=gu,
            unit_reference_upper=unit_upper, unit_eigenvalues=unitvalues)
        save_raw()  # Every computed array precedes the registration predicates.
        predicates = dict(full_scalar_positive=bool(np.all(values > 0)),
            reference_error=bool(error <= CRITERIA['reference_error']),
            inverse_eigen_symmetry=max(residual, eigen_residual, skew, unit_residual, unit_skew) <= CRITERIA['inverse_eigen_symmetry'],
            physical_L2_condition=max(conditions.values()) <= CRITERIA['condition_budget'],
            physical_L2_mass=mass_error <= 1e-15,
            independent_work=endpoint_error <= 1e-13 and interior_present,
            checkerboard=checker_defect <= 1e-13 and checker_reference > 0.,
            duplicate_deficiency=duplicate_rank == 2*case.supports,
            basis_and_units=max(basis_error, unit_error) <= 1e-10,
            positive_safe_floor=bool(np.isfinite(safe) and safe > 0.))
        record.update(status='REGISTERED_ENVELOPE_DEVELOPMENT_BOUND' if all(predicates.values()) else 'UNRESOLVED_ENVELOPE_BOUND',
            predicates=predicates, safe_case_floor=safe, cs_squared_upper=cs2,
            conditions=conditions, shell_modes=modes, quarter_witness=[list(pair) for pair in witness],
            all_scalar_eigenvalues=values.tolist(), physical_L2_basis_scale=np.diag(d).tolist(),
            numerical=dict(reference_error=error, inverse_residual=residual, eigen_residual=eigen_residual,
                gram_skew=skew, endpoint_error=endpoint_error, mass_error=mass_error,
                checker_defect=checker_defect, checker_reference=checker_reference, checker_cells=checker_cells,
                native_action_rank=native_rank, native_action_columns=case.halfcells,
                basis_error=basis_error, unit_error=unit_error, unit_inverse_residual=unit_residual,
                unit_gram_skew=unit_skew), checker_modal=checker_info, unit_modal=unit_info,
            arrays_sha256=sha(arrays_path))
        output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        return record
    except BaseException as error:
        record.update(status='FAILED_ENVELOPE_DEVELOPMENT_PREPARATION',
                      failure=dict(type=type(error).__name__, message=str(error)))
        try:
            if arrays_path.exists():
                record['arrays_sha256'] = sha(arrays_path)
            output.write_text(json.dumps(_failure_json(record), indent=2, allow_nan=False)+'\n', encoding='utf-8')
        except BaseException as recording_error:
            error.add_note('Failure summary recording also failed: '+repr(recording_error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-case', choices=[case.fixture_case for case in all_cases()], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', choices=('original', 'exact_uniform'), default='original')
    args = parser.parse_args()
    result = register(case_from_id(args.fixture_case), args.output, reference=args.reference)
    print(json.dumps(dict(status=result['status'], fixture_case=result['fixture_case'])))
