"""Fresh-process public-module compatibility for the ordinary Beam import route."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("access", ("attribute", "from_import", "submodule"))
def test_stiffened_shell_lazy_public_identity(access):
    # Use the actual already-selected installed/source artifact in a clean child.
    # A repository test must not fall back to a historical checkout.
    import anyfesolver
    source = str(Path(anyfesolver.__file__).resolve().parent.parent)
    code = '''
import importlib, sys, types
sys.path.insert(0, SOURCE)
import anyfesolver
assert 'stiffened_shell' in anyfesolver.__all__
assert 'stiffened_shell' in dir(anyfesolver)
assert 'stiffened_shell' not in vars(anyfesolver)
assert 'anyfesolver.stiffened_shell' not in sys.modules
assert 'scipy.spatial.transform' not in sys.modules
try:
    anyfesolver._not_a_public_export
except AttributeError:
    pass
else:
    raise AssertionError('unknown export accepted')
if ACCESS == 'attribute':
    module = anyfesolver.stiffened_shell
elif ACCESS == 'from_import':
    from anyfesolver import stiffened_shell as module
else:
    module = importlib.import_module('anyfesolver.stiffened_shell')
assert isinstance(module, types.ModuleType)
assert module is anyfesolver.stiffened_shell
assert module is sys.modules['anyfesolver.stiffened_shell']
from anyfesolver import stiffened_shell
assert stiffened_shell is module
assert importlib.import_module('anyfesolver.stiffened_shell') is module
assert vars(anyfesolver)['stiffened_shell'] is module
assert 'stiffened_shell' in dir(anyfesolver)
assert module.GEBeamLineAction is importlib.import_module('anyfesolver._stiffened_shell_ge_distributed_load').GEBeamLineAction
'''
    code = 'SOURCE = ' + repr(source) + '\nACCESS = ' + repr(access) + '\n' + code
    env = os.environ.copy()
    env['PYTHONPATH'] = ''
    completed = subprocess.run([sys.executable, '-c', code], env=env,
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr
