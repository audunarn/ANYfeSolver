"""Read-only rational source-equation witness; no production imports or runs."""
import ast
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINDINGS = {
    'src/anyfesolver/elements.py': 'f8c59792947a3c9b84416c61a4d9db98e42926d1bf21e74e736afbf6a9a88b37',
    'src/anyfesolver/_ge_beam3_g3c_recovery.py': 'a2100556f3ed682d6880b0387d5f7c2131ac6a869aa2d8b60a2844fa22d6691a',
    'docs/GE_BEAM3_G3C_LOCAL_RECOVERY_EQUATIONS.md': '0f81a43804b4874402d7ff09dc158de95888bc4f7e3c734e76548688f7235e67',
    'docs/GE_BEAM3_G3C_MATRIX_SHELL_CONTRACT.md': '59a09dc7db6256a145520ae7b4f2d325dee1abca5d5b5c381a7ada57a2ad92fd',
}


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def sources(root=ROOT):
    answer = {}
    for name, digest in BINDINGS.items():
        data = (root / name).read_bytes().replace(b'\r\n', b'\n')
        if sha256(data).hexdigest() != digest:
            raise ValueError('source identity mismatch: ' + name)
        answer[name] = data.decode('utf-8')
    return answer


def rational(node, source, values):
    """Tiny fail-closed interpreter of arithmetic AST, never eval/exec."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return F(ast.get_source_segment(source, node))
    if isinstance(node, (ast.Name, ast.Attribute)):
        return values[ast.unparse(node)]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -rational(node.operand, source, values)
    if isinstance(node, ast.BinOp):
        a, b = rational(node.left, source, values), rational(node.right, source, values)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.Div): return a / b
        if isinstance(node.op, ast.Pow) and b.denominator == 1: return a ** int(b)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'max' and len(node.args) == 2 and not node.keywords):
        return max(rational(arg, source, values) for arg in node.args)
    raise ValueError('unsupported source expression')


def expression(tree, target):
    matches = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and any(ast.unparse(t) == target for t in node.targets)]
    if len(matches) != 1: raise ValueError('source target is not unique: ' + target)
    return matches[0]


def witness(source, area=F('1e-14'), axis='v'):
    if axis not in ('v', 'w') or area <= 0: raise ValueError('witness input')
    tree = ast.parse(source)
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name == '_local_linear_stiffness']
    if len(methods) != 1: raise ValueError('source method identity')
    method = methods[0]
    small = rational(expression(tree, '_SMALL'), source, {})
    L, E, G, inertia, a = F(1), F(1), F(1, 2), F('1e-12'), F(1, 100)
    values = {'L': L, 'EIy': E*inertia, 'EIz': E*inertia, 'G12': G,
              'G13': G, 'self._A': area, 'self._ky': F(1), 'self._kz': F(1), '_SMALL': small}
    phi = rational(expression(method, 'phi_' + axis), source, values)
    values['phi_' + axis] = phi
    k_target, m_target = ('K[1, 1]', 'K[1, 5]') if axis == 'v' else ('K[2, 2]', 'K[2, 4]')
    k = rational(expression(method, k_target), source, values)
    # Magnitude of either end moment for a transverse displacement a at end2.
    m = abs(rational(expression(method, m_target), source, values) * a)
    shear = 2*m/L
    ga = G*area
    effective_ga = max(ga, small/L**2)
    operator_energy = k*a*a/2
    # Equilibrated M(x)=m*(1-2*x/L); integrate its square analytically.
    bending_energy = m*m*L/(6*E*inertia)
    physical_energy = bending_energy + shear*shear*L/(2*ga)
    effective_energy = bending_energy + shear*shear*L/(2*effective_ga)
    defect = physical_energy - operator_energy
    predicted_defect = shear*shear*L/2*(1/ga - 1/effective_ga)
    if effective_energy != operator_energy or defect != predicted_defect:
        raise ValueError('source-equation reconstruction disagreement')
    return dict(axis=axis, area=str(area), phi=str(phi), displacement=str(a),
                axial_displacement=str(-a*a/(2*L)), physical_shear_rigidity=str(ga),
                effective_shear_rigidity=str(effective_ga), source_floor=str(small),
                operator_energy=str(operator_energy), physical_energy=str(physical_energy),
                energy_defect=str(defect), relative_energy_defect=str(defect/operator_energy),
                physical_energy_identity=(defect == 0), clamp_active=(ga < effective_ga))


def audit():
    bound = sources()
    source = bound['src/anyfesolver/elements.py']
    clamped = [witness(source, axis=axis) for axis in ('v', 'w')]
    controls = [witness(source, area=area, axis=axis)
                for area in (F('2e-12'), F('1e-10')) for axis in ('v', 'w')]
    if any(x['physical_energy_identity'] for x in clamped): raise ValueError('missing contradiction')
    if not all(x['physical_energy_identity'] for x in controls): raise ValueError('control contradiction')
    if sources() != bound: raise ValueError('source changed during audit')
    return dict(schema='G3C_MO16_RATIONAL_SOURCE_OBSTRUCTION_V1', sources=BINDINGS,
                arithmetic='EXACT_RATIONAL_DECIMAL_SOURCE_EQUATIONS_NOT_BINARY64_EXECUTION',
                clamped_witnesses=clamped, unclamped_controls=controls,
                disposition='BLOCKED_G3C_B2_PHYSICAL_RECOVERY_SHEAR_CLAMP',
                full_g3c_qualified=False, production_activation_authorized=False,
                q4_single_field_recovery='UNRESOLVED_NOT_PROVED_IMPOSSIBLE_BY_THIS_WITNESS')


if __name__ == '__main__':
    print(canonical(audit()).decode('ascii'), end='')
