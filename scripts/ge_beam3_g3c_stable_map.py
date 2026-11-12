"""Inert source-map audit/preview; never imports or executes copied mechanics."""
import ast
from hashlib import sha256
from importlib.util import resolve_name
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MAP = 'docs/reference_cases/ge_beam3_g3c_stable_source_map_v1.json'


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode('ascii')


def strict(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError('duplicate key')
            result[key] = value
        return result
    value = json.loads(raw.decode('ascii'), object_pairs_hook=pairs)
    if canonical(value) != raw:
        raise ValueError('noncanonical map')
    return value


def fingerprint(raw):
    return dict(bytes=len(raw), sha256=sha256(raw).hexdigest())


def normalized(path):
    return path.read_bytes().replace(b'\r\n', b'\n')


def imports(source, original_module, mapping):
    """Replace only statically resolved import AST spans, never execute code."""
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1]+len(line))
    edits = []
    package = original_module.rsplit('.', 1)[0]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import) and any(a.name in mapping for a in node.names):
            raise ValueError('mapped bare import requires explicit source-map support')
        if not isinstance(node, ast.ImportFrom):
            continue
        absolute = resolve_name('.'*node.level+(node.module or ''), package) if node.level else node.module
        made = []
        for alias in node.names:
            whole = absolute+'.'+alias.name
            if node.module is None or whole in mapping:
                made.append('import '+mapping.get(whole, whole)+' as '+(alias.asname or alias.name))
            else:
                module = mapping.get(absolute, absolute)
                if absolute == 'anyfesolver._ge_beam3_mixed_ad' and alias.name in ('so3_exp', 'so3_log'):
                    module = 'anyfesolver._ge_beam3_g3c_so3_numerics'
                made.append('from '+module+' import '+alias.name+(' as '+alias.asname if alias.asname else ''))
        # External absolute imports retain their exact original spelling.
        if node.level == 0 and absolute not in mapping and absolute != 'anyfesolver._ge_beam3_mixed_ad' and not any(absolute+'.'+a.name in mapping for a in node.names):
            continue
        start = offsets[node.lineno-1]+node.col_offset
        end = offsets[node.end_lineno-1]+node.end_col_offset
        edits.append((start, end, ('\n'+' '*node.col_offset).join(made)))
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start]+replacement+source[end:]
    return source


def generated(row, mapping, root=ROOT):
    expected = {'source', 'source_module', 'destination', 'mode', 'source_fingerprint',
                'source_blob', 'replacements', 'generated_fingerprint'}
    if row['mode'] == 'definitions':
        expected |= {'definitions', 'header'}
    if set(row) != expected:
        raise ValueError('copy row schema')
    if Path(row['source']).is_absolute() or '..' in Path(row['source']).parts:
        raise ValueError('source outside root')
    raw = normalized(root/row['source'])
    if fingerprint(raw) != row['source_fingerprint']:
        raise ValueError('changed original source '+row['source'])
    source = raw.decode('utf-8')
    if row['mode'] == 'definitions':
        nodes = {n.name: n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        lines = source.splitlines(keepends=True)
        pieces = []
        for name in row['definitions']:
            node = nodes[name]
            start = min([node.lineno]+[d.lineno for d in node.decorator_list])-1
            pieces.append(''.join(lines[start:node.end_lineno]).rstrip()+'\n')
        source = row['header']+'\n\n'+'\n\n'.join(pieces)
    elif row['mode'] != 'module':
        raise ValueError('unregistered extraction mode')
    source = imports(source, row['source_module'], mapping)
    for change in row['replacements']:
        if source.count(change['old']) != 1:
            raise ValueError('ambiguous declared replacement')
        source = source.replace(change['old'], change['new'])
    ast.parse(source)
    return source.encode('utf-8')


def audit(contract, root=ROOT, *, implemented=False):
    if contract['schema'] != 'GE_BEAM3_G3C_STABLE_SOURCE_MAP_V1' or contract['stage'] != 'DESIGN_ONLY':
        raise ValueError('source-map schema/stage')
    if len(contract['copies']) != 13 or len(contract['module_map']) != 12:
        raise ValueError('source-map inventory')
    if contract['implementation_authorized'] is not False or contract['execution_authorized'] is not False:
        raise ValueError('design cannot authorize implementation or execution')
    environment = os.environ.copy()
    environment.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='NUL',
                       GIT_CONFIG_SYSTEM='NUL', GIT_NO_REPLACE_OBJECTS='1')
    def git(*args):
        return subprocess.check_output(['git', '-c', 'safe.directory='+str(root),
                                        '-c', 'core.autocrlf=true', '-c', 'core.eol=crlf',
                                        *args], cwd=root, env=environment, timeout=10)
    base = contract['base_commit']
    if git('rev-parse', base+'^{tree}').decode().strip() != contract['base_tree']:
        raise ValueError('base tree mismatch')
    tree = {}
    for entry in git('ls-tree', '-r', '-z', base).split(b'\0'):
        if entry:
            metadata, path = entry.split(b'\t', 1)
            tree[path.decode('utf-8')] = metadata.split()[2].decode('ascii')
    destinations = set()
    for row in contract['copies']:
        destination = row['destination']
        if destination in destinations or not destination.startswith(('src/anyfesolver/_ge_beam3_g3c_stable/', 'tests/test_ge_beam3_g3c_stable_')) or '..' in Path(destination).parts:
            raise ValueError('duplicate or out-of-scope destination')
        destinations.add(destination)
        if tree.get(row['source']) != row['source_blob']:
            raise ValueError('original source blob mismatch')
        made = generated(row, contract['module_map'], root)
        if fingerprint(made) != row['generated_fingerprint']:
            raise ValueError('generated body/import mapping changed '+destination)
        if implemented and normalized(root/destination) != made:
            raise ValueError('implementation differs from frozen mapping '+destination)
    for row in contract['bindings']:
        if tree.get(row['path']) != row['source_blob']:
            raise ValueError('authority blob mismatch')
        if fingerprint(normalized(root/row['path'])) != row['fingerprint']:
            raise ValueError('bound authority changed '+row['path'])
    return dict(kind='STATIC_SOURCE_MAP_ONLY', copies=len(destinations), implementation_checked=implemented)


if __name__ == '__main__':
    print(canonical(audit(strict(normalized(ROOT/MAP)))).decode(), end='')
