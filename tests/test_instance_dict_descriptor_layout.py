"""The guards' notion of a class's own ``__dict__`` descriptor (Python 3.14 and 3.15).

Up to 3.14 each class owns its ``__dict__`` getset descriptor; from 3.15 all
classes share one descriptor owned by ``object``. The guards accept exactly the
running interpreter's own layout and still refuse a replaced descriptor.
"""

import sys
import types

from anyfesolver.boundary import _SHARED_INSTANCE_DICT, _builtin_instance_dict_descriptor


class _Plain:
    pass


class _Other:
    pass


def _own(cls):
    return type.__getattribute__(cls, "__dict__")["__dict__"]


def test_shared_descriptor_is_captured_only_on_the_shared_layout():
    if sys.version_info >= (3, 15):
        assert type(_SHARED_INSTANCE_DICT) is types.GetSetDescriptorType
        assert _SHARED_INSTANCE_DICT.__objclass__ is object
        assert _own(_Plain) is _SHARED_INSTANCE_DICT
    else:
        assert _SHARED_INSTANCE_DICT is None
        assert _own(_Plain).__objclass__ is _Plain


def test_a_class_own_descriptor_is_accepted():
    for cls in (_Plain, _Other):
        assert _builtin_instance_dict_descriptor(_own(cls), cls)


def test_a_foreign_or_replaced_descriptor_is_refused():
    fake = property(lambda self: {})
    assert not _builtin_instance_dict_descriptor(fake, _Plain)
    # A builtin getset descriptor with another name.
    weakref_descriptor = type.__getattribute__(_Plain, "__dict__")["__weakref__"]
    assert not _builtin_instance_dict_descriptor(weakref_descriptor, _Plain)
    # Another builtin type's __dict__ descriptor (owned by neither _Plain nor object).
    function_dict = type.__getattribute__(types.FunctionType, "__dict__")["__dict__"]
    assert type(function_dict) is types.GetSetDescriptorType
    assert not _builtin_instance_dict_descriptor(function_dict, _Plain)
    if sys.version_info < (3, 15):
        # Per-class layout: another class's descriptor is not this class's own.
        assert not _builtin_instance_dict_descriptor(_own(_Other), _Plain)
