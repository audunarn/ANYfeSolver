"""Inert orchestration checks; no numerical owner is constructed."""
import ast
from hashlib import sha256
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'),str(ROOT/'tests')]
import ge_beam3_g3c_history_owner as history
import test_ge_beam3_g3c_restart_preflight as specimens


class WrapperStaticTests(unittest.TestCase):
    def test_import_is_inert_and_runtime_deterministic(self):
        self.assertNotIn('numpy',sys.modules)
        self.assertNotIn('anyfesolver',sys.modules)
        self.assertEqual(history.runtime_identity(),history.runtime_identity())

    def test_malformed_packet_never_reaches_constructor(self):
        value=specimens.syntax_packet(); value['runtime_sha256']=history.runtime_identity()
        value['final_state']['native_rows'][0]['payload']['response']['full']['residual']=[0.]*41
        specimens.rehash(value); raw=history.packet.canonical(value)
        with patch.object(history,'HistoryOwner',side_effect=AssertionError('construction')) as factory:
            with self.assertRaisesRegex(ValueError,'array shape'):
                history.resume(raw,sha256(raw).hexdigest(),expected_runtime_sha256=history.runtime_identity())
            factory.assert_not_called()

    def test_well_formed_commitments_require_genuine_construction(self):
        value=specimens.syntax_packet(); value['runtime_sha256']=history.runtime_identity()
        raw=history.packet.canonical(value)
        with patch.object(history,'HistoryOwner',side_effect=RuntimeError('genuine construction required')) as factory:
            with self.assertRaisesRegex(RuntimeError,'genuine construction required'):
                history.resume(raw,sha256(raw).hexdigest(),expected_runtime_sha256=history.runtime_identity())
            factory.assert_called_once_with('J_B2_PAIR','BASE','NONE')

    def test_no_accepted_generation_write_or_cached_replay(self):
        text=(ROOT/'scripts/ge_beam3_g3c_history_owner.py').read_text()
        tree=ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='__setattr__':
                self.assertNotEqual(getattr(node.args[1],'value',None),'_published')
        self.assertIn("owner.solve(entry['command'])",text)
        self.assertIn("owner.checkpoint_bytes() != raw",text)
        self.assertIn("packet.preflight(raw, expected_sha256",text)
        self.assertNotIn('numpy',sys.modules)

    def test_guard_code_and_keyword_defaults_cannot_bypass_publication_check(self):
        # Orchestration stub only, not a numerical owner or scientific proof.
        class StubInner:
            def __init__(self): self.publications=0
            def solve(self,command,*,hook):
                hook('before_publish'); self.publications+=1
        for mutation in ('code','kwdefaults'):
            inner=StubInner(); wrapper=object.__new__(history.HistoryOwner); lock=threading.Lock()
            for name,value in dict(_owner=inner,_owned_owner=inner,_lock=lock,_owned_lock=lock,
                    _dispatch=history.dispatch(),_runtime=history.runtime_identity(),_poisoned=False).items():
                object.__setattr__(wrapper,name,value)
            guard=wrapper._guard
            for name,value in dict(_owned_guard=guard,_guard_call=guard,
                                  _guard_signature=history.guard_signature(guard)).items():
                object.__setattr__(wrapper,name,value)
            code=guard.__func__.__code__; keywords=guard.__func__.__kwdefaults__
            def bypass(self,**kwargs): return None
            def mutate(stage):
                if mutation=='code': guard.__func__.__code__=bypass.__code__
                else: guard.__func__.__kwdefaults__={**keywords,'_dispatch_fn':lambda:wrapper._dispatch}
            try:
                with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError,'guard code/default'):
                    wrapper.solve(dict(kind='LOAD_STAGE',root_stage=0,load_factor=0.,force_scale=.01),hook=mutate)
                self.assertEqual(inner.publications,0)
                self.assertTrue(wrapper._poisoned)
                self.assertFalse(lock.locked())
            finally:
                guard.__func__.__code__=code; guard.__func__.__kwdefaults__=keywords
        self.assertNotIn('numpy',sys.modules)


if __name__ == '__main__': unittest.main()
