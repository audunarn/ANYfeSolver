"""Optional CP311 Windows guard; unsupported hosts retain the Python package."""
import hashlib
import os
from pathlib import Path
import platform
import sys
from setuptools import Extension,setup

def native_guard_enabled(environ=os.environ):
    """New name wins; the pre-rename name is honoured only when it is absent."""
    for name in ('ANYFESOLVER_BUILD_NATIVE_GUARD','ANYSOLVER_BUILD_NATIVE_GUARD'):
        if name in environ:
            return environ[name]!='0'
    return True

extensions=[]
if (sys.platform=='win32' and sys.version_info[:2]==(3,11)
        and platform.machine().upper() in ('AMD64','X86_64')
        and native_guard_enabled()):
    source=Path('src/anyfesolver/_native_guard.cpp')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    # The unchanged C++ source reads ANYSOLVER_GUARD_SOURCE_SHA; that is its macro name.
    extensions=[Extension('anyfesolver._native_guard',[str(source)],language='c++',optional=True,
        define_macros=[('ANYSOLVER_GUARD_SOURCE_SHA','"'+digest+'"')],
        extra_compile_args=['/std:c++17','/O2','/EHsc'])]
setup(ext_modules=extensions)
