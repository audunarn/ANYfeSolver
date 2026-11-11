"""Shared bounded beam gate runner; only explicitly registered gates execute."""
import argparse
import ast
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import threading
import time
import uuid

_JOB_TYPE_LOCK=threading.Lock()

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import ge_beam3_g3b_environment as environment
import ge_beam3_g3c_proof_compressed as proof_compressed

SCOPE='GE_BEAM3_BOUNDED_REGISTERED_GATE_V2'
BASE='8767dbaaf4003daa24369628a1e6e119a642e0bf'
CONTRACT='docs/reference_cases/ge_beam3_g3c_b2_physical_contract_v1.json'
CONTRACT_SHA='6d202e2c25bc62c0987f090a7b5c092bc788318bbb37e27c1261e67af592996b'
DESIGN_REVIEW='docs/reference_cases/ge_beam3_g3c_b2_physical_contract_review_v1.json'
DESIGN_SHA='ba784eb40010a765fed25f891a0b0ccadf68c5beadabcf20b5161f2da5c204a4'
CAPSULE=Path('C:/Users/AudunArnesenNyhus/AppData/Local/ANYrelease/ge-beam3-g3b-rehearsal-history-20260912-0fe7f8f/capsule/environment.json')
CAPSULE_SHA='2ce154b866565e1d03019651b1ae7fdd070e5554f38d72adf21d8080558b6756'
JOB='docs/reference_cases/e4_pl_s3_v2_bounded_process.py'
JOB_SHA='c5b192c9c3f6ee2c68a42ab4a0cfbcdbe81581381b800c13aacce0bb219a3383'
TEST='tests/test_ge_beam3_g3c_b2_physical_core.py'
TESTS={'b2-core':(TEST,'numeric_core'),
       'b2-adapter':('tests/test_ge_beam3_g3c_b2_physical_adapter.py','adapter'),
       'q4-audit':('tests/test_ge_beam3_q4_recovery_coefficient_audit.py','q4_audit'),
       'q4-affine-exact':('tests/test_ge_beam3_q4_affine_recovery.py','q4_affine'),
       'q4-affine-numerical':('tests/test_ge_beam3_q4_affine_numerical_recovery.py','q4_affine_numerical'),
       'g3c-physical':('tests/test_ge_beam3_g3c_physical_owner.py','physical_owner'),
       'g4a-station':('tests/test_ge_beam3_g4a_station_transaction.py','g4a_station'),
       'g4-completion':('tests/test_ge_beam3_g4_completion.py','g4_completion'),
       'g5-completion':('tests/test_ge_beam3_g5_completion.py','g5_completion')}
G4A_BASE='6cb1aad7ea564a0acb7af3e44ac92728548d7025'
G4A_BASE_TREE='625b375968e83ae905807ae41b044555c3b9438d'
G4A_PLAN='docs/GE_BEAM3_G4A_STATION_TRANSACTION_CONTRACT.md'
G4A_PLAN_SHA='c01f2661dc023a65a68b4c5488c6a798111aedf621504ea635940314c5724a8e'
G4A_CONTRACT='docs/reference_cases/ge_beam3_g4a_station_transaction_contract_v1.json'
G4A_CONTRACT_SHA='8426af2cb56ac9277ee6dcdc3c47e5239e1d9b3db4a475fe6ae0aa95e0f783bc'
G4A_DESIGN_REVIEW='docs/reference_cases/ge_beam3_g4a_station_transaction_contract_review_v1.json'
G4A_DESIGN_REVIEW_SHA='24b0f428701a69980dc95590474bf5425600bd93c5b11a9e63c23ecc99c7fdee'
G4A_TESTS=('test_six_stage_history_independent_replay_and_capabilities',
    'test_trial_failure_is_nonpublishing','test_prepare_failure_discards_whole_proposal',
    'test_before_and_after_publication_boundaries','test_foreign_stale_replayed_and_reordered_inputs',
    'test_mutated_response_history_and_law_fail_closed','test_concurrent_writer_and_immutable_outputs',
    'test_tangent_and_incremental_work_directional_agreement','test_checker_does_not_import_transaction_owner')
G4A_NODES=(G4A_TESTS[0],
    *(G4A_TESTS[1]+'['+str(index)+']' for index in range(3)),
    *(G4A_TESTS[2]+'['+str(index)+']' for index in range(3)),*G4A_TESTS[3:])
G4A_IMPLEMENTATION_PATHS={
    'src/anyfesolver/_ge_beam3_g4a_station_transaction.py',
    'docs/reference_cases/ge_beam3_g4a_station_transaction_checker.py',
    'tests/test_ge_beam3_g4a_station_transaction.py',
    'scripts/run_ge_beam3_qualification.py','tests/test_ge_beam3_qualification_runner.py'}
G4_BASE='5d7966bd1ea22cd5aaee7ad4481d93dae575659a'
G4_BASE_TREE='6a27b4c8d23243237e42f39780951a535e4a39e2'
G4_PLAN='docs/GE_BEAM3_G4_COMPLETION_CONTRACT.md'
G4_PLAN_SHA='631742fd615da0a831ade8e37d91e098955e90a8ee35aad0b6bec69b761bcdc2'
G4_CONTRACT='docs/reference_cases/ge_beam3_g4_completion_contract_v1.json'
G4_CONTRACT_SHA='ef77ccd2b976cf428c09d6a318d508430666c1b7c7987defa1a094d77390ec85'
G4_DESIGN_REVIEW='docs/reference_cases/ge_beam3_g4_completion_contract_review_v1.json'
G4_DESIGN_REVIEW_SHA='1057114f2356746487f8b2ce20631886e6a061b11d574f064f583da0a8ccef15'
G4_TESTS=('test_complete_material_state_control_and_recovery',
    'test_resealed_checkpoint_mutations_fail_before_publication',
    'test_untrusted_restart_bytes_and_foreign_authority_fail_closed',
    'test_atomic_failure_keeps_last_accepted_epoch',
    'test_actual_cancellation_keeps_fibre_accepted_prefix',
    'test_missing_record_and_noncanonical_bytes_are_rejected')
G4_NODES=(G4_TESTS[0],
    *(G4_TESTS[1]+'['+name+']' for name in ('schema','owner','graph','formulation','section','constraint',
      'load','control','provenance','epoch','previous','state','recovery','record_count','production','extra')),
    *(G4_TESTS[2]+'['+name+']' for name in ('duplicate','nonfinite','external_hash','cross_family')),
    *(G4_TESTS[3]+'['+name+']' for name in ('trial','prepare','before_publication')),
    *G4_TESTS[4:])
G4_IMPLEMENTATION_PATHS={'src/anyfesolver/_ge_beam3_g4_completion.py',
    'tests/test_ge_beam3_g4_completion.py','scripts/run_ge_beam3_qualification.py',
    'tests/test_ge_beam3_qualification_runner.py'}
G5_BASE='0373b8bc0a7c5bbfc4c991cd62d59d718734e24c'
G5_BASE_TREE='8905b4286c8df5a77b53fbb6721e865838339a91'
G5_PLAN='docs/GE_BEAM3_G5_INTEGRATION_CLOSURE_CONTRACT.md'
G5_PLAN_SHA='ff1bf641452987571d2d2e6c55ea349067723e454be1a1cae53c524db993e495'
G5_TESTS=('test_actual_legacy_b2_reference_routes',
    'test_actual_legacy_b3_reference_routes',
    'test_ge_reference_load_inertia_and_combinations',
    'test_ge_reference_modal_buckling_and_linear_transient',
    'test_explicit_native_static_restart_recovery_and_provenance',
    'test_unsupported_routes_fail_closed_before_state_mutation',
    'test_selector_and_provenance_boundary_remains_additive')
G5_IMPLEMENTATION_PATHS={'src/anyfesolver/_ge_beam3_g5_completion.py',
    'src/anyfesolver/ge_beam3_element.py','tests/test_ge_beam3_g5_completion.py',
    'scripts/run_ge_beam3_g5_package.py','tests/test_ge_beam3_g5_package_gate.py',
    'tests/test_ge_beam3_mixed_p3_optin.py',
    'scripts/run_ge_beam3_qualification.py','tests/test_ge_beam3_qualification_runner.py'}
PHYSICAL_BASE='ba4f4d793bb9494f3c116030c99577a1b3ebefe6'
PHYSICAL_BASE_TREE='47c29d6e5d60dc133544cedade3a11a78ee13b3c'
PHYSICAL_PLAN='docs/GE_BEAM3_G3C_PHYSICAL_MIXED_OWNER_CONTRACT.md'
PHYSICAL_PLAN_SHA='f6638863e6e222c795e980194c9a71cf42eb83bdcba6af2a8c409164ad4d957b'
PHYSICAL_DESIGN_REVIEW='docs/reference_cases/ge_beam3_g3c_physical_owner_contract_review_v1.json'
PHYSICAL_DESIGN_SHA='47829c5c1c896ed2eb28131ffedadd25cc83bc5bed79bc2517cfe308f7dc8988'
PHYSICAL_PARTITION_ADDENDUM='docs/GE_BEAM3_G3C_PHYSICAL_REHEARSAL_PARTITION_ADDENDUM.md'
PHYSICAL_PARTITION_ADDENDUM_SHA='449a37b98bdc2fcbaf81a9a51d83712ac63988020fb632eab904595f10cf8e27'
PHYSICAL_PARTITION_REVIEW='docs/reference_cases/ge_beam3_g3c_physical_rehearsal_partition_review_v1.json'
PHYSICAL_PARTITION_REVIEW_SHA='3868f1480cfdb8cb4883dc115ba6afd1e2a4069126245893fa4a9cf2b38671ab'
PHYSICAL_FORMAL_ADDENDUM='docs/GE_BEAM3_G3C_PHYSICAL_FORMAL_EXECUTION_ADDENDUM.md'
PHYSICAL_FORMAL_ADDENDUM_SHA='03289611222de86c5887abe53dc74c3bfdc700c0ee1bb548878918c9c2d3f746'
PHYSICAL_FORMAL_DESIGN_REVIEW='docs/reference_cases/ge_beam3_g3c_physical_formal_execution_design_review_v1.json'
PHYSICAL_FORMAL_DESIGN_REVIEW_SHA='55cbdfdf18242fc113caeb3e12b9cca3fa4c3b1b7c69d55875bfa60ddcc76d1d'
PHYSICAL_FORMAL_HISTORY_INVENTORY_SHA='c8e934143fd896d3d1c072c55819e35f05b8a2b1592c75b2f7207b543cba9fb9'
PHYSICAL_FORMAL_ASSIGNMENT_INVENTORY_SHA='94fb5e294a802fe6b2515408b680b908330b66578453d2cfe93a35827c5c247e'
PHYSICAL_PARTITION_INVENTORY_SHA='a53103c9d3162dbfdc04e647c81fa36aaad6fa80f9a13b1cd4abcae124c96733'
PHYSICAL_PARTITION_MANIFEST_SHA='0b1beffb58ddbed1bb53891e11a7f5e0c8ad7b646cc539e4a908d7e4321ade86'
PHYSICAL_CORRECTION_ADDENDUM='docs/GE_BEAM3_G3C_PHYSICAL_CORRECTION_INHERITANCE_ADDENDUM.md'
PHYSICAL_CORRECTION_ADDENDUM_SHA='2c03f72a0e9c50c6a22fe5b7f47fd66f786aa3eadc23bedeb4dab1f2045a8696'
PHYSICAL_CORRECTION_REVIEW='docs/reference_cases/ge_beam3_g3c_physical_correction_inheritance_review_v1.json'
PHYSICAL_CORRECTION_REVIEW_SHA='0fbaedb533e501a5136407d5dc039b5393b4bbf53064c273c7e974917ba9fec9'
PHYSICAL_CORRECTION_REVISION='docs/GE_BEAM3_G3C_PHYSICAL_CORRECTION_INHERITANCE_V2.md'
PHYSICAL_CORRECTION_REVISION_SHA='dae0dd3081ff45ab66a266b2706bdee6049e556d01c09003de2ba1749ce0be76'
PHYSICAL_CORRECTION_REVISION_REVIEW='docs/reference_cases/ge_beam3_g3c_physical_correction_inheritance_v2_review_v2.json'
PHYSICAL_CORRECTION_REVISION_REVIEW_SHA='339367400bc48eeb07eb09eca353097297e7dae2907f23ae53b53dab5199c26b'
PHYSICAL_CORRECTION_SUPERSEDED_REVIEW='docs/reference_cases/ge_beam3_g3c_physical_correction_inheritance_v2_review.json'
PHYSICAL_CORRECTION_RECOVERY='docs/GE_BEAM3_G3C_PHYSICAL_CORRECTION_PARTITION_RECOVERY_V3.md'
PHYSICAL_CORRECTION_RECOVERY_SHA='faabb5d96fd7e2fbd74f115a46e050dcb27df6bf05db9f7d5ce91d8ee5017bdd'
PHYSICAL_CORRECTION_RECOVERY_INITIAL_REVIEW='docs/reference_cases/ge_beam3_g3c_physical_correction_partition_recovery_review_v3_initial.json'
PHYSICAL_CORRECTION_RECOVERY_INITIAL_REVIEW_SHA='55d018012f19aec71ffdcf30b695ae815eabb2c443621756be2936accf7f844f'
PHYSICAL_CORRECTION_RECOVERY_SUPERSEDED_REVIEW='docs/reference_cases/ge_beam3_g3c_physical_correction_partition_recovery_review_v3.json'
PHYSICAL_CORRECTION_RECOVERY_SUPERSEDED_REVIEW_SHA='42240ff0a21dc322f97703edf8a5c1e483e3d3ee0d8579bf4b5c99cf4db8656f'
PHYSICAL_CORRECTION_RECOVERY_REVIEW='docs/reference_cases/ge_beam3_g3c_physical_correction_partition_recovery_review_v3_correction.json'
PHYSICAL_CORRECTION_RECOVERY_REVIEW_SHA='f20539245dae880b332905a5e86cde3a42a840dbaa6a7b14731b7946a0b5498b'
PHYSICAL_CORRECTION_RECOVERY_IMPL_INITIAL_REVIEW='docs/reference_cases/ge_beam3_g3c_physical_correction_partition_implementation_review_v3_initial.json'
PHYSICAL_CORRECTION_RECOVERY_IMPL_INITIAL_REVIEW_SHA='61be7f2377a20ad9d68fa389cd6cd35db5d98cc66f4f676b331bf69825706b0a'
PHYSICAL_CORRECTION_UNCHANGED_INPUTS_SHA='34a2f26c67aa71d7d443f514ca143657dc8a82fd0077108c53a922fb07230d1c'
PHYSICAL_PREDECESSOR={'commit':'f6a62518be52a414604aa5e1beddd4601093faca',
    'tree':'1230eea2b64ca6e585ad23b389e514f1d5c493c4'}
PHYSICAL_PREDECESSOR_REVIEW_SHA='52f8df02bd22c635bf828bf93190a34ec1463b20268a1a822e4e0f3c6c042eaf'
PHYSICAL_PREDECESSOR_RUNTIME_SHA='41a0dddda886672479294953871e83be3073ed38e129e12f3fa210bfbf3ce87c'
PHYSICAL_PARTITIONS=(
    ('R-HISTORY',tuple(range(0,10)),'654af72fd3dddca92c736a138e179df0e4ba97da7cd25e4ae97316403fbb5d27'),
    ('R-PREFIX-A',tuple(range(10,30)),'7a3390c1df7c7391feb2bea111bd9ee314db0b90c411145f42fbe0844170eae5'),
    ('R-PREFIX-B',tuple(range(30,50)),'b3e168ff12baee60cb5c88eed5e2337aafa9ec7b860704cf92ba0626ff4314bc'),
    ('R-PREFIX-C',tuple(range(50,70)),'a91be5d24950c85c45bdaf4efd8e1ce23515c1f4adf2f3db472b973dbe8d8313'),
    ('R-PREFIX-D',tuple(range(70,90)),'10d5cd0ca3a3529605f02f12d0b821c801cd4cce564f85d74830c158b3ad741e'),
    ('R-GUARDS',tuple(range(90,234)),'e597616542d6bf5a323b1cd12b7e3c0517d726eab812118b8ec63548c5429827'))
PHYSICAL_PARTITION_IDS=tuple(row[0] for row in PHYSICAL_PARTITIONS)
PHYSICAL_CORRECTION_GUARD_SEGMENTS=(
    ('R-GUARDS-A',tuple(range(90,114)),'6981dc38e504841a158b7f9e747e33cf0e94d7c9736fdee60c276fb12792fd98'),
    ('R-GUARDS-B',tuple(range(114,138)),'761a4c222e150bccc9f501ebcc64c150855d8d63bee2c30e03c4239355bc95af'),
    ('R-GUARDS-C',tuple(range(138,162)),'c6f2002fec928892800c1d61b60ea2822e5c8c838a700ce8d6b4dd96a015b518'),
    ('R-GUARDS-D',tuple(range(162,186)),'1eac165b1ad6a6dae98c5302949738a387d476497b62c3f7482e4408eaaa4b1c'),
    ('R-GUARDS-E',tuple(range(186,210)),'8046ab0073bf7128568cb62a33e7286432edf18a8ea52f0e0b1940260b059c63'),
    ('R-GUARDS-F',tuple(range(210,234)),'2bece23cb08bafb2bdee82f97c7245dc03429b4c1e6a65293907bfc0ad69b37c'))
PHYSICAL_CORRECTION_GUARD_SEGMENT_IDS=tuple(row[0] for row in PHYSICAL_CORRECTION_GUARD_SEGMENTS)
PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA='8ae0c6560960ef9bf9035838abb5daae056372d7aae0f9426945330a898bf477'
PHYSICAL_OWNER_TEST='tests/test_ge_beam3_g3c_physical_owner.py'
PHYSICAL_HISTORY_TEST='tests/test_ge_beam3_g3c_physical_history_restart.py'
PHYSICAL_CORRECTION_TEST='tests/test_ge_beam3_g3c_physical_correction_guards.py'
PHYSICAL_FORMAL_TEST='tests/test_ge_beam3_g3c_physical_formal_case.py'
PHYSICAL_FORMAL_INITIAL_IMPLEMENTATION_REVIEW='docs/reference_cases/ge_beam3_g3c_physical_formal_shard_implementation_review_v1_initial.json'
PHYSICAL_FORMAL_INITIAL_IMPLEMENTATION_REVIEW_SHA='351aec568c44a02b0b78d57bb40568753f6657a6c8d51e959a651dfaf63f0d68'
PHYSICAL_FORMAL_CORRECTION1_IMPLEMENTATION_REVIEW='docs/reference_cases/ge_beam3_g3c_physical_formal_shard_implementation_review_v1_correction1.json'
PHYSICAL_FORMAL_CORRECTION1_IMPLEMENTATION_REVIEW_SHA='0348afe2033e85f6ec423f2f18708b4235deed52eea8091b2760d031d9ee6f8c'
PHYSICAL_FORMAL_MEASUREMENT_CASE_ORDINALS=(0,224,374)
PHYSICAL_PROOF_COMPRESSED_PLAN='docs/GE_BEAM3_G3C_PROOF_COMPRESSED_COMPLETION_V2.md'
PHYSICAL_PROOF_COMPRESSED_PLAN_SHA='e6e1932dd0bba2b465306330796e7b64444c13e40c43727875ea9e9fe2dcf8b7'
PHYSICAL_PROOF_COMPRESSED_TOOL='scripts/ge_beam3_g3c_proof_compressed.py'
PHYSICAL_PROOF_COMPRESSED_TOOL_SHA='1b36c7a89a98a75fa17372d09e9540f531e201304a42ab135c99e05d524795c3'
PHYSICAL_PROOF_COMPRESSED_CHECKER='docs/reference_cases/ge_beam3_g3c_proof_compressed_checker.py'
PHYSICAL_PROOF_COMPRESSED_CHECKER_SHA='7045637080093566335a29b68263007ff86e05776aa62ef353aed6dcfc111076'
PHYSICAL_PROOF_COMPRESSED_INITIAL_REVIEW='docs/reference_cases/ge_beam3_g3c_proof_compressed_implementation_review_v2.json'
PHYSICAL_PROOF_COMPRESSED_INITIAL_REVIEW_SHA='6d84a93df5d1fe99407ed4b4461868f6d360bb5f1498c3dfc16f02f2cffac96b'
PHYSICAL_PROOF_COMPRESSED_MEASUREMENT='docs/reference_cases/ge_beam3_g3c_proof_compressed_measurement_v2.json'
PHYSICAL_PROOF_COMPRESSED_MEASUREMENT_SHA='52f2d7769b08559d6d880695dc2d73a438a9e3544d7ac3d5ef4dd824e8780967'
PHYSICAL_PROOF_COMPRESSED_PARTITION_SHA='3b41ea2e2e8b4b8feb308261b9d87b9e4ca0283106304f298706705ae41dd3bd'
PHYSICAL_PROOF_COMPRESSED_CHECKER_INITIAL_REVIEW='docs/reference_cases/ge_beam3_g3c_proof_compressed_checker_review_v2_initial.json'
PHYSICAL_PROOF_COMPRESSED_CHECKER_INITIAL_REVIEW_SHA='a892fd8b0b5ac7459632e1be1c7f7bdd353907241718c469c5dc06cd95fcfec8'
PHYSICAL_PROOF_COMPRESSED_FAILED_CYCLE='docs/reference_cases/ge_beam3_g3c_proof_compressed_cycle1_failed_v2.json'
PHYSICAL_PROOF_COMPRESSED_FAILED_CYCLE_SHA='fa11ebc773a63920f613cbe974bf2d13bd417c9c7fc559fc2fbe0f7d5ed8e74b'
PHYSICAL_PROOF_COMPRESSED_FAILED_REVIEW='docs/reference_cases/ge_beam3_g3c_proof_compressed_implementation_review_v2_final.json'
PHYSICAL_PROOF_COMPRESSED_FAILED_REVIEW_SHA='0fc05033d2cd96d5190942126734ca9a1d22ae362d41443206b2873a56fa2302'
PHYSICAL_PROOF_COMPRESSED_SUCCESSOR_REVIEW='docs/reference_cases/ge_beam3_g3c_proof_compressed_implementation_review_v2_successor.json'
PHYSICAL_PROOF_COMPRESSED_SUCCESSOR_REVIEW_SHA='ad2196bdde31b9f8c3519dcd883ca54097fd26b9d1fbac6bba9fc5fe5554b2a2'
PHYSICAL_IMPLEMENTATION_PATHS={PHYSICAL_PLAN,PHYSICAL_DESIGN_REVIEW,
    PHYSICAL_PARTITION_ADDENDUM,PHYSICAL_PARTITION_REVIEW,
    'src/anyfesolver/_ge_beam3_g3c_physical_owner.py',
    'src/anyfesolver/_ge_beam3_g3c_physical_authority.py',
    'scripts/ge_beam3_g3c_physical_history_owner.py',
    'scripts/ge_beam3_g3c_physical_restart_preflight.py',
    PHYSICAL_OWNER_TEST,PHYSICAL_HISTORY_TEST,
    PHYSICAL_FORMAL_ADDENDUM,PHYSICAL_FORMAL_DESIGN_REVIEW,PHYSICAL_FORMAL_TEST,
    PHYSICAL_FORMAL_INITIAL_IMPLEMENTATION_REVIEW,PHYSICAL_FORMAL_CORRECTION1_IMPLEMENTATION_REVIEW,
    'tests/test_ge_beam3_qualification_runner.py',
    'scripts/run_ge_beam3_qualification.py',
    'docs/GE_BEAM3_QUALIFICATION_COMPLETION_REGISTER.md',
    'scripts/ge_beam3_g3c_rehearsal_mutations.py',
    'tests/test_ge_beam3_g3c_rehearsal_mutations_static.py',
    PHYSICAL_CORRECTION_ADDENDUM,PHYSICAL_CORRECTION_REVIEW,
    PHYSICAL_CORRECTION_REVISION,PHYSICAL_CORRECTION_REVISION_REVIEW,PHYSICAL_CORRECTION_SUPERSEDED_REVIEW,
    PHYSICAL_CORRECTION_RECOVERY,PHYSICAL_CORRECTION_RECOVERY_INITIAL_REVIEW,
    PHYSICAL_CORRECTION_RECOVERY_SUPERSEDED_REVIEW,PHYSICAL_CORRECTION_RECOVERY_REVIEW,
    PHYSICAL_CORRECTION_RECOVERY_IMPL_INITIAL_REVIEW,
    'scripts/ge_beam3_g3c_correction_lease_binding.py',
    'scripts/ge_beam3_g3c_physical_history_owner.py',PHYSICAL_CORRECTION_TEST}
PHYSICAL_PROOF_COMPRESSED_PATHS={PHYSICAL_PROOF_COMPRESSED_PLAN,
    PHYSICAL_PROOF_COMPRESSED_TOOL,PHYSICAL_PROOF_COMPRESSED_CHECKER,
    PHYSICAL_PROOF_COMPRESSED_INITIAL_REVIEW,PHYSICAL_PROOF_COMPRESSED_MEASUREMENT,
    PHYSICAL_PROOF_COMPRESSED_CHECKER_INITIAL_REVIEW,
    PHYSICAL_PROOF_COMPRESSED_FAILED_CYCLE,PHYSICAL_PROOF_COMPRESSED_FAILED_REVIEW,
    PHYSICAL_PROOF_COMPRESSED_SUCCESSOR_REVIEW,
    'tests/test_ge_beam3_g3c_proof_compressed.py'}
PHYSICAL_TIMING_GATE_PATHS={
    'docs/reference_cases/ge_beam3_g3c_physical_formal_measurement_review_access_incident_v1.json',
    'docs/reference_cases/ge_beam3_g3c_physical_formal_measurement_review_v1.json',
    'docs/reference_cases/ge_beam3_g3c_physical_formal_shard_implementation_review_v1.json'}
PHYSICAL_TIMING_GATE_HASHES={
    'docs/reference_cases/ge_beam3_g3c_physical_formal_measurement_review_access_incident_v1.json':'61e1b7b77248c4ac0196319cc7e25e2e58ca247d947ea691b13a42ec29a41ddc',
    'docs/reference_cases/ge_beam3_g3c_physical_formal_measurement_review_v1.json':'e336032fa338b1f5f727949136aef961dbadc997efc9b289a254f5fcc596cb9e',
    'docs/reference_cases/ge_beam3_g3c_physical_formal_shard_implementation_review_v1.json':'19dbd054c3b96d5f39b6b79c5dc5c557856b451ab2fd15587438393126bcde15'}
PHYSICAL_CORRECTION_CHANGED_PATHS={
    'scripts/ge_beam3_g3c_rehearsal_mutations.py',
    'tests/test_ge_beam3_g3c_rehearsal_mutations_static.py',
    PHYSICAL_CORRECTION_ADDENDUM,PHYSICAL_CORRECTION_REVIEW,
    PHYSICAL_CORRECTION_REVISION,PHYSICAL_CORRECTION_REVISION_REVIEW,PHYSICAL_CORRECTION_SUPERSEDED_REVIEW,
    PHYSICAL_CORRECTION_RECOVERY,PHYSICAL_CORRECTION_RECOVERY_INITIAL_REVIEW,
    PHYSICAL_CORRECTION_RECOVERY_SUPERSEDED_REVIEW,PHYSICAL_CORRECTION_RECOVERY_REVIEW,
    PHYSICAL_CORRECTION_RECOVERY_IMPL_INITIAL_REVIEW,'docs/GE_BEAM3_QUALIFICATION_COMPLETION_REGISTER.md',
    'scripts/ge_beam3_g3c_correction_lease_binding.py',
    'scripts/ge_beam3_g3c_physical_history_owner.py',PHYSICAL_CORRECTION_TEST,
    'scripts/run_ge_beam3_qualification.py','tests/test_ge_beam3_qualification_runner.py'}
PHYSICAL_INHERITED_FINGERPRINTS={
 'smoke':(('scientific.json',6645,'dff755e3dc50d8b793ac43a5b2febea832275ace0dac18ff8e5a6c588b541616'),('receipt.json',3405,'118b06cc65ea098b782393f8e92983a033de95ceea9a9c2f6a5c99528034cbf6'),('process.json',7091,'4fc4e8f4a84468405573a4cbf13bf1b648ca3cb7c96dc6eafe2a136beac60721')),
 'local':(('scientific.json',84699,'ecb695f6c7c30ace594870766afe7e9cdbb682e5df1b6bd563a95443da7150ae'),('receipt.json',14536,'a1e62f7253bdbecc3d89647537ef551384d0c1dbaadece1acdbd5edb6454956e'),('process.json',27638,'e912f52a513cc174e0f829c6a4d710f09a1e93ab0ca0adf991088e23f2edd328')),
 'R-HISTORY':(('scientific.json',19393,'b2e71ef49767df5f3ae585faa5e0b278def5f8b7c3b3c84fa6f6da87af1d4209'),('receipt.json',7173,'44445deeea4597759041ec279507a025e859ce043f1db67821ce2117520cbf90'),('process.json',19607,'09c7736c03fe74f606676e4fe65f992b796123ebed1fdcd01238b4ff831df5d5')),
 'R-PREFIX-A':(('scientific.json',18922,'84a39ec74c6ea13b02f8ef2293a555b8a4a9692b4fa0b31fa4d14617968f0ff6'),('receipt.json',12965,'9b4112683739675f9cec7acdd8966af888280062ce4af0b70a522f7628c199f4'),('process.json',21418,'ecbb49444361d6703a7b9dfb8e75540a52fd86934c2c6a24be3d262c164973bc')),
 'R-PREFIX-B':(('scientific.json',18916,'ae2504a125350c154339644265fd7099f2c25c93f79693a32f24aba09f8e8e3f'),('receipt.json',13479,'f0f8e8a8bc1727ca80af8c73649d1380ac73a8d709a7891264d2739360a4e470'),('process.json',21422,'63cc1a5c7536bf5ec81c03f29247bb212bcdfc1870cebad63e4b5f591dba443b')),
 'R-PREFIX-C':(('scientific.json',18910,'756fdff456cafd5edcdb74cfeb580fd2248cd14c6a63cc3f609692ead744a8e7'),('receipt.json',13993,'22a7fe753f730d8436ef008e39d23becc4c1b26c03b3b6bd8a2ae78f332b0e9a'),('process.json',21424,'6092e155a02a439b29f480fece584eabb994b08fe934066aafeae009f4b389cc')),
 'R-PREFIX-D':(('scientific.json',19342,'b1ddc241a233d9e4b93748351401f72e168623dd13067159ab8c69b9a1f06414'),('receipt.json',14507,'c7d42b1321d58c1f787278b86595e1429679bd67ed97aa158fa23c4c215e2b64'),('process.json',21433,'6b04c405a87761384bb8ccdb81a847b83a52f1f49baae86ffe859ff5b16797c7'))}
PHYSICAL_FAILED_GUARDS_FINGERPRINT={'bytes':75379,'sha256':'7fd9a5ce935a5018c9e0656f3ddc56c279fe077399d0891a46025da79dd22903'}
PHYSICAL_INTERRUPTED_GUARDS_MANIFEST={'files':1039,'bytes':67930103,
    'sha256':'cdaaf2feeb38ba4e771c55ef66d9ff8ad32536fd9fea98fc2dff812910518ba6'}
PHYSICAL_CORRECTION_COMPATIBILITY_NEGATIVE_COUNT=12
NUMERICAL_PLAN='docs/GE_BEAM3_Q4_AFFINE_NUMERICAL_RECOVERY_CONTRACT.md'
NUMERICAL_PLAN_SHA='21cec54b2469e291c8f90a2c042e3cf693a14f7bc10efd5111ea842ca65a218c'
NUMERICAL_REVIEW='docs/reference_cases/ge_beam3_q4_affine_numerical_design_review_v1.json'
NUMERICAL_REVIEW_SHA='e2a28209165d72046093ec08d06b8243399f820ab8db0b6fb7beeeab0e96055f'
CHART_ID='GE_BEAM3_Q4_AFFINE_INCREMENT_RESOLVED_CHART_NUMERICS_V1'
INCREMENT_ADDENDUM='docs/GE_BEAM3_Q4_AFFINE_CANCELLATION_SAFE_CHART_ADDENDUM.md'
INCREMENT_ADDENDUM_SHA='7e637ff69453715e5251614d76e08d2be7c7fd8ac4932c5fd7a4cc939023caf0'
INCREMENT_REVIEW='docs/reference_cases/ge_beam3_q4_affine_increment_chart_design_review_v1.json'
INCREMENT_REVIEW_SHA='04bda4789cc3529fbae2fd52f2881e8b8f7a6e48447cb6ec91c8275581ff6655'
STATION_ASSOCIATION_ID='GE_BEAM3_Q4_NATURAL_COORDINATE_BIJECTION_V1'
CHART_ADDENDUM='docs/GE_BEAM3_Q4_AFFINE_STABLE_CHART_ADDENDUM.md'
CHART_ADDENDUM_SHA='1eb29b8814e6555e7d16569e2c8d30c828a60458dea4a2801c10f4d58d2c2176'
CHART_REVIEW='docs/reference_cases/ge_beam3_q4_affine_stable_chart_design_review_v1.json'
CHART_REVIEW_SHA='ca6de27e79186c9e0b37bf21e765322aa8e46dc323eb858db039951fec7efeab'
CHART_SOURCES={
 'src/anyfesolver/_ge_beam3_g3c_affine_q4_chart.py':'2a26742ec7680a5e75934fb32ee8601d3877cd0befb250489f991f1b9d198a9f',
 'docs/reference_cases/ge_beam3_q4_affine_numerical_chart.py':'b7cc92f70d35e3b8f2ab10202cb2f3ad1116e92775425888e482293b323c23ac',
 'src/anyfesolver/_ge_beam3_g3c_so3_numerics.py':'588ace39570658f1a83c0110d3a019ebad770d6091b271ff4bbc40b97f96e12e',
 'docs/GE_BEAM3_G3C_SO3_NUMERICS_CONTRACT.md':'269bcd4e1fda151e7c22c396a70c43498e8450b222c263dd2655b9ca45998496',
 'docs/reference_cases/ge_beam3_g3c_so3_numerics_implementation_review_v1.json':'4debde89b44c990478487dda4852d7734b852ca01dfb2d3626e782c2328756f1',
 'docs/reference_cases/ge_beam3_g3c_so3_numerics_confirmation_receipt_v1.json':'aa7225cae347662bf92b3c02f8a1ff541d40e67f13db21900d0ad3269f88b91f',
 'docs/reference_cases/ge_beam3_g3c_so3_numerics_result_v1.json':'c8b938bbdbcce329aa83afc00037905d158142d7d2e417acee159bab95aba6fd',
 'src/anyfesolver/_ge_beam3_g3c_affine_q4_registry.py':'5f208b716d89ba778193d1261fb5975db4ece68c93fa2efce97e5c129c56acf1'}
NUMERICAL_TESTS=['test_affine_recovery_'+suffix for suffix in (
    'definition_and_source_identity','zero_and_station_constitutive','independent_material_fields',
    '64_stationarity_and_schur','actual_chart_work_hessian','directional_derivatives_all_steps',
    'six_rigid_modes_and_common_motion','d4_and_director_transports','passive_and_same_pose_rebase',
    'all_graph_q4_reference_variants','tiny_physical_energy_and_numerical_separation',
    'definition_observation_races','immutable_detached_results_and_reentry',
    'unsupported_routes_and_cancellation','actual_mutation_rejection')]
NUMERICAL_SMOKE='test_affine_recovery_smoke_square_station_work'
NUMERICAL_TABLES=[{'definitions':19,'source_graph':1,'extension_lemma':1,'fingerprint':1,'chart_authority':1,'station_join':1},
    {'station':54},{'independent':54},{'schur':54},{'work':54},{'directional':27},
    {'rigid':3,'common_motion':12},{'d4':24,'director':6},{'passive':3,'rebase':3},
    {'graph':20},{'tiny':6,'channels':54},{'races':20},{'immutability':9},{'rejections':19},{'mutations':42,'chart_mutations':9,'eigen_derivatives':1}]
NUMERICAL_SHAPES=('SQUARE','RECTANGLE','RHOMBUS')
NUMERICAL_BASE_IDS=[s+'::'+v for s in NUMERICAL_SHAPES for v in ('0.01','1','10')]
NUMERICAL_GRAPH_IDS=[g+'::'+v for g in ('J_Q4_PAIR','J_MULTIFAMILY_LOOP') for v in
    ('BASE','SHUFFLED_INSERTION','RENUMBERED','CONNECTIVITY_REVERSED','PROPER_GLOBAL_TRANSFORM')]
NUMERICAL_CONTEXT_IDS=[s+'::'+p for s in NUMERICAL_BASE_IDS for p in ('ZERO','MEMBRANE','BENDING','SHEAR','CHECKERBOARD','MIXED')]
NUMERICAL_TABLE_IDS=[
    {'definitions':NUMERICAL_BASE_IDS+NUMERICAL_GRAPH_IDS,'source_graph':['source_graph'],'extension_lemma':['ideal_recipe_extension'],'fingerprint':['typed_payload_encoding'],'chart_authority':['stable_chart_bindings'],'station_join':['exact_coordinate_bijection']},
    {'station':NUMERICAL_CONTEXT_IDS},{'independent':NUMERICAL_CONTEXT_IDS},{'schur':NUMERICAL_CONTEXT_IDS},{'work':NUMERICAL_CONTEXT_IDS},
    {'directional':[i+'::h='+h for i in NUMERICAL_BASE_IDS for h in ('0.0001','1e-05','1e-06')]},
    {'rigid':[s+'::1' for s in NUMERICAL_SHAPES],'common_motion':[s+'::1::motion='+str(i) for s in NUMERICAL_SHAPES for i in range(4)]},
    {'d4':[s+'::1::D4:'+str(i) for s in NUMERICAL_SHAPES for i in range(8)],
     'director':[s+'::1::DIRECTOR:'+str(i) for s in NUMERICAL_SHAPES for i in (-1,1)]},
    {'passive':[s+'::1::PASSIVE' for s in NUMERICAL_SHAPES],'rebase':[s+'::1' for s in NUMERICAL_SHAPES]},
    {'graph':[g+'::'+p for g in NUMERICAL_GRAPH_IDS for p in ('ZERO','MIXED')]},
    {'tiny':[s+'::1::amplitude='+a for s in NUMERICAL_SHAPES for a in ('1e-06','0.001')],'channels':NUMERICAL_CONTEXT_IDS},
    {'races':['descriptor','displacement_array','accepted_array','cancellation','material_descriptor','cache_array','cache_cancellation',
              'preentry_E','preentry_coordinates','preentry_material_direction','preentry_policy','preentry_cached_definition',
              'preentry_chart_missing','preentry_chart_old','preentry_chart_wrong','preentry_chart_cache',
              'preentry_station_missing','preentry_station_old','preentry_station_wrong','preentry_station_cache']},
    {'immutability':['all_detached_arrays','caller_arrays_preserved','same_input_repeat','reentry','changed_accepted_matrix','concurrent_evaluation','operator_cache_tamper','nested_candidate_bytes','old_chart_entrypoints_intercepted']},
    {'rejections':['nonaffine','director','material_direction','node_ids','generalized_section','history_section','offset','initial_fields','foreign_policy',
       'GE_BEAM3_G3C_MATRIX_POSE_SHELL_PULLBACK_V1','OLD_RETAINED_35_VARIABLE_SYSTEM','FOREIGN',
       'nonfinite_q','nonfinite_accepted','wrong_q_shape','wrong_rotation_shape','before_work','before_publication','invalid_callback']},
    {'mutations':[s+'::'+m for s in NUMERICAL_SHAPES for m in ('station_order','weight','frame','M','resultant','n','Dn','Hn',
       'force_weighted_Hessian','chart_second','coupling_sign','inverse','numerical_energy_leak','nonlinear_point_join')],
     'chart_mutations':['missing_delta_covariance','naive_reference_subtraction','omitted_delta_derivatives',
                        'wrong_davenport_branch','wrong_polar_branch','davenport_nonconvergence','polar_nonconvergence',
                        'eigen_first_derivative','eigen_second_derivative'],
     'eigen_derivatives':['SQUARE::1::amplitude=1e-06']}]
NUMERICAL_SOURCE_HASHES={
 'src/anyfesolver/e4_pl_element.py':'7fc46a18046e043512a5fb2c3f61ed0b95b834e4dde764f63c500a885e87ea38',
 'src/anyfesolver/elements.py':'f8c59792947a3c9b84416c61a4d9db98e42926d1bf21e74e736afbf6a9a88b37',
 'src/anyfesolver/_ge_beam3_g3c_local_shell.py':'69d9f27a96ed7e9a5959a42a081eeafbbe72021de6e4da1f91355c5fb3850f10',
 'src/anyfesolver/_ge_beam3_variational_shell.py':'b0db7a83633f4a835c06e940a6f0de36ab958f7e816c37a23c6ebc16259a2a60',
 'src/anyfesolver/_ge_beam3_mixed_ad.py':'b299ff765cd2eaae8b33a2ba1d05069f6fe8c39209e1eac75df356a2afb1fe36',
 'src/anyfesolver/_ge_beam3_pose_joint.py':'69cfb0a71761ab7cad0d62080d81511949e7c920f7f70162bcaba0497b402657',
 'src/anyfesolver/_ge_beam3_g3c_definition.py':'4b46b870fec010e3378df83c374d763f858d8f25820c14f9704dc6f0a656f0c2',
 'src/anyfesolver/_ge_beam3_g3c_owner.py':'18b9565d56192e23f192b1a3fbae3cb12ed7ede958c0e4b3e46279a799d1afc5',
 'docs/reference_cases/ge_beam3_g3c_fixtures_v1.json':'d5fecc80c3203ac7d191b4031d64bf35c56d2900066cb01fc1dfa4276d24c006'}
PHYSICAL_TABLES=frozenset(('work','common_motion','d4','director','passive','rebase','graph','tiny','smoke'))
PHYSICAL_SHAPES={'physical_energy':[],'physical_force':[24],'physical_hessian':[24,24],
    'source_physical_energy':[],'source_physical_force':[24],'source_physical_hessian':[24,24]}
OBSERVATION_MANIFEST='docs/reference_cases/ge_beam3_q4_affine_observation_manifest_v1.json'
OBSERVATION_MANIFEST_SHA='a2b19f46cdb41bbf7981d824525c6f4f0553815d0aaf00e2faf37e5e0d59f53a'
AFFINE_TESTS=['test_affine_exact_arithmetic_and_schema','test_affine_source_and_representation_boundaries',
              'test_affine_square_chart_polynomial','test_affine_rectangle_chart_polynomial',
              'test_affine_rhombus_chart_polynomial','test_affine_stationary_schur_and_mutations']
AFFINE_FIXTURES=('AFFINE_Q4_SQUARE','AFFINE_Q4_RECTANGLE','AFFINE_Q4_RHOMBUS')
AFFINE_PLAN='docs/GE_BEAM3_Q4_AFFINE_CHART_RECOVERY_PLAN.md'
AFFINE_PLAN_SHA='42abd8f6cae8131be9bc62869329e1000244675f005b9bef9660a54e3e239fc0'
AFFINE_REVIEW='docs/reference_cases/ge_beam3_q4_affine_design_review_v1.json'
AFFINE_REVIEW_SHA='0d7e631ed3aad3630aa6bd296670735d93f24a4fa8160e4a17a4eed0164b66f5'
Q4_TESTS=['test_exact_field_laws_and_schema','test_registered_source_boundary',
          'test_square_coefficient_proof','test_rhombus_coefficient_proof','test_assembly_and_proof_mutations']
Q4_FIXTURES=('MO16_SQUARE_EXACT','MO16_AFFINE_RHOMBUS_EXACT')
Q4_PLAN_SHA='65f51816fe678acd38dffa9578f7cf2864d7cea518fc0e16e659899632cc3ce8'
ALLOWED={
    'scripts/run_ge_beam3_qualification.py','tests/test_ge_beam3_qualification_runner.py',
    'docs/GE_BEAM3_QUALIFICATION_COMPLETION_REGISTER.md',
    'src/anyfesolver/_ge_beam3_g3c_affine_q4_recovery.py',
    'src/anyfesolver/_ge_beam3_g3c_affine_q4_chart.py',CHART_ADDENDUM,CHART_REVIEW,
    'src/anyfesolver/_ge_beam3_g3c_affine_q4_increment_chart.py',INCREMENT_ADDENDUM,INCREMENT_REVIEW,
    'docs/reference_cases/ge_beam3_q4_affine_increment_chart.py',
    'src/anyfesolver/_ge_beam3_g3c_affine_q4_registry.py',
    'docs/reference_cases/ge_beam3_q4_affine_numerical_checker.py',
    'docs/reference_cases/ge_beam3_q4_affine_numerical_chart.py',
    'docs/GE_BEAM3_Q4_AFFINE_RECOVERY_EXTENSION_LEMMA.md',
    'docs/reference_cases/ge_beam3_q4_affine_extension_lemma_review_v1.json',
    OBSERVATION_MANIFEST,
    TESTS['q4-affine-numerical'][0],
}
INTEGRATION_REVIEW='docs/reference_cases/ge_beam3_8073635_integration_contract_review.json'
INTEGRATION_SHA='3be7021fd63a259cca5c48d0e6b47e6a261805f10b4909374f5a9610c17a7b59'
MEMORY=24*1024**3
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','NUMBA_NUM_THREADS','TBB_NUM_THREADS')

def canonical(v):
    return (json.dumps(v,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode('ascii')

def exact_json(actual,expected):
    """JSON equality that never treats bool/int/float as interchangeable."""
    if type(actual)is not type(expected):return False
    if type(actual)is dict:
        return set(actual)==set(expected) and all(exact_json(actual[k],expected[k]) for k in actual)
    if type(actual)is list:
        return len(actual)==len(expected) and all(exact_json(a,b) for a,b in zip(actual,expected))
    return actual==expected

def write(path,value):
    with path.open('xb') as stream:stream.write(canonical(value))

def read(path):return environment.regular(path).read_bytes()
def fingerprint(raw):return dict(bytes=len(raw),sha256=sha256(raw).hexdigest())

def exact_fingerprint(actual,raw):
    return (type(actual)is dict and set(actual)=={'bytes','sha256'}
            and type(actual['bytes'])is int and actual['bytes']>=0
            and type(actual['sha256'])is str and exact_json(actual,fingerprint(raw)))

def exact_assignment_index(value,expected):
    if type(value)is not int or value!=expected:raise ValueError('exact assignment index')
    return value

def git(*args):
    return subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),
        '-c','core.autocrlf=true','-c','core.eol=crlf',*args],cwd=ROOT,
        env=environment.git_env(),timeout=30).decode().strip()

def inputs():
    names=git('ls-files','-z').split('\0')
    return {name:fingerprint(read(ROOT/name).replace(b'\r\n',b'\n')) for name in sorted(names) if name}

PHYSICAL_OWNER_STATIC_NODES=(
    'test_physical_inventory_and_obligation_map',
    'test_physical_inert_schema_guards')
PHYSICAL_OWNER_NUMERICAL_NODES=(
    'test_physical_family_work','test_physical_directional','test_physical_atomicity',
    'test_physical_token_and_definition_guards','test_physical_independent_joint_work_transport',
    'test_physical_observation_and_cache_guards','test_physical_recovery_witness_guards')
PHYSICAL_HISTORY_NODE='test_physical_history_assignment'
PHYSICAL_PREFLIGHT_NODE='test_physical_preflight_guards'
PHYSICAL_MUTATION_NODE='test_physical_mutation_assignment'
PHYSICAL_OBLIGATIONS={
 'MO01':'test_physical_inventory_and_obligation_map',
 'MO02':'test_physical_family_work','MO03':'test_physical_family_work',
 'MO04':'test_physical_family_work','MO05':'test_physical_directional',
 'MO06':'test_physical_independent_joint_work_transport','MO07':'test_physical_history_assignment',
 'MO08':'DEFERRED_FULL_OPERATOR_HISTORY_TRANSPORT_PARTITION_ADDENDUM','MO09':'test_physical_atomicity',
 'MO10':'test_physical_atomicity','MO11':'test_physical_observation_and_cache_guards',
 'MO12':'test_physical_observation_and_cache_guards','MO13':'test_physical_preflight_guards',
 'MO14':'test_physical_history_assignment','MO15':'test_physical_mutation_assignment',
 'MO16':'test_physical_recovery_witness_guards','MO17':'SHARED_RUNNER_PROCESS_INVENTORY',
 'MO18':'SHARED_RUNNER_TWO_FULL_CYCLE_AND_INDEPENDENT_REVIEW'}

def physical_support():
    """Inert inventory module only; it imports no anyfesolver or numerical package."""
    import ge_beam3_g3c_physical_history_owner as support
    return support

def physical_test_nodes(path):
    tree=ast.parse(read(ROOT/path))
    return [n.name for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]

def physical_inventory(lane):
    support=physical_support()
    owner_nodes=physical_test_nodes(PHYSICAL_OWNER_TEST)
    history_nodes=physical_test_nodes(PHYSICAL_HISTORY_TEST)
    expected_owner=list(PHYSICAL_OWNER_STATIC_NODES+PHYSICAL_OWNER_NUMERICAL_NODES)
    expected_history=[PHYSICAL_HISTORY_NODE,PHYSICAL_PREFLIGHT_NODE,PHYSICAL_MUTATION_NODE]
    if owner_nodes!=expected_owner or history_nodes!=expected_history:
        raise ValueError('registered physical test inventory changed')
    if lane=='local':
        rows=[dict(kind='owner-static',nodes=[PHYSICAL_OWNER_TEST+'::'+n for n in PHYSICAL_OWNER_STATIC_NODES])]
        rows += [dict(kind='owner',assignment=dict(graph=graph,variant=variant),
                      nodes=[PHYSICAL_OWNER_TEST+'::'+n for n in PHYSICAL_OWNER_NUMERICAL_NODES])
                 for graph in support.GRAPHS for variant in support.VARIANTS]
        return rows
    if lane in ('smoke','formal'):
        return support.work_inventory(lane)
    if lane!='rehearsal':raise ValueError('unregistered physical lane')
    rows=list(support.work_inventory('rehearsal'))
    origins=[]
    for item in support.mutation_inventory():
        origin=item['origin']
        if origin not in origins:origins.append(origin)
    rows += [dict(kind='preflight',origin=o) for o in origins]
    for item in support.mutation_inventory():
        probe={k:v for k,v in item.items() if k!='executor'}
        rows.append(dict(kind='authority-mutation' if item['executor']=='authority' else 'mutation',probe=probe))
    return rows

def physical_formal_case_associations():
    """Exact join from375 case work units to the unchanged3825 assignments."""
    support=physical_support();cases=support.history_matrix();assignments=physical_inventory('formal')
    if (len(cases)!=375 or sha256(canonical(cases)).hexdigest()!=PHYSICAL_FORMAL_HISTORY_INVENTORY_SHA
        or len(assignments)!=3825
        or sha256(canonical(assignments)).hexdigest()!=PHYSICAL_FORMAL_ASSIGNMENT_INVENTORY_SHA):
        raise ValueError('formal physical inventory authority')
    cursor=len(cases);rows=[]
    for ordinal,case in enumerate(cases):
        history_assignment=dict(kind='history',case=case,stages=case['accepted_stages'])
        if not exact_json(assignments[ordinal],history_assignment):raise ValueError('formal history association')
        prefix_indexes=list(range(cursor,cursor+case['accepted_stages']+1))
        for prefix,index in enumerate(prefix_indexes):
            if not exact_json(assignments[index],dict(kind='prefix',case=case,prefix=prefix)):
                raise ValueError('formal prefix association')
        rows.append(dict(case_ordinal=ordinal,case=case,history_assignment_index=ordinal,
                         prefix_assignment_indexes=prefix_indexes))
        cursor+=case['accepted_stages']+1
    if cursor!=len(assignments):raise ValueError('formal assignment association coverage')
    return rows

def _physical_formal_shard(associations,case_ordinal,kind,prefix_start=None,prefix_stop=None):
    if type(case_ordinal)is not int or type(case_ordinal)is bool or not 0<=case_ordinal<len(associations):
        raise ValueError('formal case ordinal')
    row=associations[case_ordinal]
    if kind=='history-producer':
        if prefix_start is not None or prefix_stop is not None:raise ValueError('producer prefix range forbidden')
        return dict(kind=kind,case_ordinal=case_ordinal,case=row['case'],
                    history_assignment_index=row['history_assignment_index'])
    if kind!='prefix-range':raise ValueError('formal shard kind')
    if (type(prefix_start)is not int or type(prefix_start)is bool
        or type(prefix_stop)is not int or type(prefix_stop)is bool
        or not 0<=prefix_start<prefix_stop<=len(row['prefix_assignment_indexes'])):
        raise ValueError('formal prefix shard range')
    return dict(kind=kind,case_ordinal=case_ordinal,case=row['case'],
        prefix_start=prefix_start,prefix_stop=prefix_stop,
        assignment_indexes=row['prefix_assignment_indexes'][prefix_start:prefix_stop])

def physical_formal_shard(case_ordinal,kind,prefix_start=None,prefix_stop=None):
    return _physical_formal_shard(physical_formal_case_associations(),case_ordinal,kind,prefix_start,prefix_stop)

def physical_formal_partition(shards,mode='measurement'):
    """Validate a closed measured work partition without authorizing execution."""
    if mode not in ('measurement','formal','compressed'):raise ValueError('formal partition mode')
    if type(shards)is not list or not shards:raise ValueError('formal partition shards')
    made=[];seen_history=set();seen_prefix=set();associations=physical_formal_case_associations()
    for ordinal,item in enumerate(shards):
        if type(item)is not dict:raise ValueError('formal partition row')
        shard=_physical_formal_shard(associations,item.get('case_ordinal'),item.get('kind'),
            item.get('prefix_start'),item.get('prefix_stop'))
        if not exact_json(item,shard):raise ValueError('formal partition row authority')
        if shard['kind']=='history-producer':
            if shard['case_ordinal'] in seen_history:raise ValueError('duplicate formal producer')
            seen_history.add(shard['case_ordinal'])
        else:
            for index in shard['assignment_indexes']:
                if index in seen_prefix:raise ValueError('duplicate formal prefix')
                seen_prefix.add(index)
        made.append(dict(shard_id=f'{mode}-{ordinal:04d}',assignment=shard,
                         assignment_sha256=sha256(canonical(shard)).hexdigest()))
    if mode=='formal':
        if seen_history!=set(range(375)):raise ValueError('formal producer coverage')
        if seen_prefix!={index for row in associations for index in row['prefix_assignment_indexes']}:
            raise ValueError('formal prefix coverage')
    if mode=='compressed':
        case_index={row['case']['case_id']:row for row in associations}
        expected_history={case_index[case_id]['case_ordinal']
                          for case_id in proof_compressed.executed_case_ids()}
        expected_prefix={case_index[row['case_id']]['prefix_assignment_indexes'][prefix]
                         for row in proof_compressed.restart_plan() for prefix in row['prefixes']}
        if seen_history!=expected_history or seen_prefix!=expected_prefix:
            raise ValueError('compressed executed-basis coverage')
    body=dict(schema='GE_BEAM3_G3C_PHYSICAL_FORMAL_PARTITION_V1',mode=mode,shards=made)
    if mode=='compressed':
        body['proof_manifest_sha256']=sha256(canonical(proof_compressed.manifest())).hexdigest()
    return dict(body=body,self_sha256=sha256(canonical(body)).hexdigest())

def physical_formal_measurement_partition():
    """Frozen nonclassifying timing sample: three cases and 1/2/3 prefixes."""
    shards=[]
    for case_ordinal in PHYSICAL_FORMAL_MEASUREMENT_CASE_ORDINALS:
        shards.append(physical_formal_shard(case_ordinal,'history-producer'))
        for start,stop in ((0,1),(1,3),(3,6)):
            shards.append(physical_formal_shard(case_ordinal,'prefix-range',start,stop))
    return physical_formal_partition(shards,'measurement')

def physical_proof_compressed_partition():
    """Exact 25-history/120-restart successor basis; mechanics still separate."""
    associations=physical_formal_case_associations()
    by_id={row['case']['case_id']:row for row in associations};shards=[]
    restart={row['case_id']:row['prefixes'] for row in proof_compressed.restart_plan()}
    for case_id in proof_compressed.executed_case_ids():
        ordinal=by_id[case_id]['case_ordinal']
        shards.append(physical_formal_shard(ordinal,'history-producer'))
        prefixes=restart[case_id];start=prefixes[0];previous=start;length=1
        for prefix in prefixes[1:]:
            if prefix!=previous+1 or length==2:
                shards.append(physical_formal_shard(ordinal,'prefix-range',start,previous+1));start=prefix
                length=1
            else:length+=1
            previous=prefix
        shards.append(physical_formal_shard(ordinal,'prefix-range',start,previous+1))
    return physical_formal_partition(shards,'compressed')

def physical_formal_common_manifest(candidate,inputs,implementation_review,runtime_sha,partition,cycle=0):
    if (type(candidate)is not dict or set(candidate)!={'commit','tree'}
        or any(type(candidate[k])is not str or len(candidate[k])!=40 for k in candidate)
        or type(inputs)is not dict or type(implementation_review)is not dict):
        raise ValueError('formal common authority')
    for value in candidate.values():
        if any(c not in '0123456789abcdef' for c in value):raise ValueError('formal git identity')
    sha_value(runtime_sha)
    if type(cycle)is not int or type(cycle)is bool or cycle not in (0,1,2):raise ValueError('formal cycle')
    if type(partition)is not dict or partition.get('body',{}).get('mode') not in ('measurement','formal','compressed'):
        raise ValueError('formal partition manifest')
    expected_partition=physical_formal_partition(
        [row['assignment'] for row in partition['body'].get('shards',[])],partition['body'].get('mode'))
    if not exact_json(partition,expected_partition):raise ValueError('formal partition changed')
    mode=partition['body']['mode']
    if (mode=='measurement')!=(cycle==0):raise ValueError('formal cycle and mode')
    if mode=='compressed' and cycle not in (1,2):raise ValueError('compressed cycle')
    cases=physical_support().history_matrix();assignments=physical_inventory('formal')
    body=dict(kind='G3C_PHYSICAL_PRIVATE_DEVELOPMENT',
        schema='GE_BEAM3_G3C_PHYSICAL_FORMAL_COMMON_MANIFEST_V1',mode=mode,cycle=cycle,
        candidate=candidate,inputs=inputs,implementation_review=implementation_review,
        review_sha256=sha256(canonical(implementation_review)).hexdigest(),
        runtime_sha256=runtime_sha,contract_sha256=PHYSICAL_PLAN_SHA,
        formal_addendum_sha256=PHYSICAL_FORMAL_ADDENDUM_SHA,
        formal_design_review_sha256=PHYSICAL_FORMAL_DESIGN_REVIEW_SHA,
        history_inventory_sha256=PHYSICAL_FORMAL_HISTORY_INVENTORY_SHA,
        assignment_inventory_sha256=PHYSICAL_FORMAL_ASSIGNMENT_INVENTORY_SHA,
        cases=cases,assignments=assignments,associations=physical_formal_case_associations(),partition=partition,
        limits=dict(workers=3,threads=1,memory_bytes=MEMORY,child_seconds=600,
                    inactivity_seconds=120,wave_seconds=1800),
        full_g3c_qualified=False,production_qualified=False)
    return dict(body=body,self_sha256=sha256(canonical(body)).hexdigest())

def physical_validate_formal_common(value):
    if type(value)is not dict or set(value)!={'body','self_sha256'}:raise ValueError('formal common schema')
    body=value['body']
    if type(body)is not dict or set(body)!={'kind','schema','mode','cycle','candidate','inputs',
        'implementation_review','review_sha256','runtime_sha256','contract_sha256','formal_addendum_sha256',
        'formal_design_review_sha256','history_inventory_sha256','assignment_inventory_sha256','cases',
        'assignments','associations','partition','limits','full_g3c_qualified','production_qualified'}:
        raise ValueError('formal common body schema')
    made=physical_formal_common_manifest(body['candidate'],body['inputs'],body['implementation_review'],
                                         body['runtime_sha256'],body['partition'],body['cycle'])
    if not exact_json(value,made):raise ValueError('formal common content')
    return body

def physical_formal_manifest_descriptor(path):
    path=Path(path).resolve();raw=read(path)
    physical_validate_formal_common(environment.strict(raw))
    return dict(path=str(path),bytes=len(raw),sha256=sha256(raw).hexdigest())

def physical_formal_producer_receipt(out,lease,common):
    assignment=lease['assignment'];case=assignment['case']
    if assignment['kind']!='history-producer':raise ValueError('formal producer receipt kind')
    packets={f'prefix-{prefix:02d}':packet_descriptor(out/f'prefix-{prefix:02d}.json')
             for prefix in range(case['accepted_stages']+1)}
    body=dict(schema='GE_BEAM3_G3C_PHYSICAL_FORMAL_PRODUCER_RECEIPT_V1',mode=common['mode'],
        cycle=common['cycle'],common_manifest_sha256=lease['common_manifest']['sha256'],
        producer_shard_id=lease['shard_id'],case_ordinal=assignment['case_ordinal'],case_id=case['case_id'],
        science=fingerprint(read(out/'scientific.shard.json')),completion=fingerprint(read(out/'completion.json')),
        packets=packets,passed=True)
    return dict(body=body,self_sha256=sha256(canonical(body)).hexdigest())

def physical_validate_formal_producer_receipt(path,common):
    path=Path(path).resolve();raw=read(path);value=environment.strict(raw)
    if type(value)is not dict or set(value)!={'body','self_sha256'}:raise ValueError('formal producer receipt schema')
    body=value['body']
    if (type(body)is not dict or set(body)!={'schema','mode','cycle','common_manifest_sha256','producer_shard_id',
        'case_ordinal','case_id','science','completion','packets','passed'}
        or body['schema']!='GE_BEAM3_G3C_PHYSICAL_FORMAL_PRODUCER_RECEIPT_V1'
        or not exact_json(body['mode'],common['mode']) or not exact_json(body['cycle'],common['cycle'])
        or body['passed']is not True
        or value['self_sha256']!=sha256(canonical(body)).hexdigest()):raise ValueError('formal producer receipt content')
    ordinal=body['case_ordinal']
    if type(ordinal)is not int or type(ordinal)is bool or not 0<=ordinal<375:raise ValueError('formal producer receipt ordinal')
    expected=physical_formal_shard(ordinal,'history-producer');case=expected['case']
    if body['case_id']!=case['case_id']:raise ValueError('formal producer receipt case')
    producer_dir=path.parent;producer_lease=environment.strict(read(producer_dir/'lease.json'))
    if (producer_lease.get('shard_id')!=body['producer_shard_id']
        or not exact_json(producer_lease.get('assignment'),expected)
        or body['common_manifest_sha256']!=producer_lease.get('common_manifest',{}).get('sha256')):
        raise ValueError('formal producer receipt lease')
    physical_validate_formal_lease(producer_lease,common)
    if (not exact_fingerprint(body['science'],read(producer_dir/'scientific.shard.json'))
        or not exact_fingerprint(body['completion'],read(producer_dir/'completion.json'))):
        raise ValueError('formal producer receipt science')
    expected_packets={f'prefix-{prefix:02d}':packet_descriptor(producer_dir/f'prefix-{prefix:02d}.json')
                      for prefix in range(case['accepted_stages']+1)}
    if not exact_json(body['packets'],expected_packets):raise ValueError('formal producer receipt packets')
    physical_validate_formal_process(producer_dir,producer_lease)
    return body

def physical_validate_formal_lease(lease,common):
    assignment=lease.get('assignment') if type(lease)is dict else None
    extra={'producer_receipt'} if type(assignment)is dict and assignment.get('kind')=='prefix-range' else set()
    if (type(lease)is not dict or set(lease)!=({'kind','schema','run_id','mode','cycle','common_manifest',
        'shard_id','assignment','input_packets'}|extra)
        or lease['kind']!='G3C_PHYSICAL_PRIVATE_DEVELOPMENT'
        or lease['schema']!='GE_BEAM3_G3C_PHYSICAL_FORMAL_SHARD_LEASE_V1'
        or not exact_json(lease['mode'],common['mode']) or not exact_json(lease['cycle'],common['cycle'])):
        raise ValueError('formal shard lease schema')
    uuid.UUID(lease['run_id'])
    descriptor=lease['common_manifest'];raw=validate_packet_descriptor(descriptor)
    if not exact_json(physical_validate_formal_common(environment.strict(raw)),common):
        raise ValueError('formal common descriptor')
    match=[row for row in common['partition']['body']['shards'] if row['shard_id']==lease['shard_id']]
    if len(match)!=1 or not exact_json(lease['assignment'],match[0]['assignment']):
        raise ValueError('formal leased shard')
    assignment=lease['assignment'];packets=lease['input_packets']
    if assignment['kind']=='history-producer':expected=set()
    else:expected={f'prefix-{i:02d}' for i in range(assignment['prefix_start'],assignment['prefix_stop'])}|{'final'}
    if type(packets)is not dict or set(packets)!=expected:raise ValueError('formal shard packet set')
    for row in packets.values():validate_packet_descriptor(row)
    if assignment['kind']=='prefix-range':
        descriptor=lease['producer_receipt'];receipt_raw=validate_packet_descriptor(descriptor)
        receipt=physical_validate_formal_producer_receipt(descriptor['path'],common)
        if fingerprint(receipt_raw)!={k:descriptor[k] for k in ('bytes','sha256')}:
            raise ValueError('formal producer receipt descriptor')
        if (receipt['case_ordinal']!=assignment['case_ordinal']
            or receipt['case_id']!=assignment['case']['case_id']):
            raise ValueError('formal replay producer case lineage')
        expected_packets={f'prefix-{prefix:02d}':receipt['packets'][f'prefix-{prefix:02d}']
            for prefix in range(assignment['prefix_start'],assignment['prefix_stop'])}
        expected_packets['final']=receipt['packets'][f"prefix-{assignment['case']['accepted_stages']:02d}"]
        if not exact_json(packets,expected_packets):raise ValueError('formal replay producer lineage')
    return assignment

def physical_formal_expected_files(assignment,process=True):
    names={'lease.json','review.json','worker-attempt.json','stdout.log','stderr.log',
           'scientific.shard.json','completion.json'}
    if assignment['kind']=='history-producer':
        names.update(('accepted-diagnostic.json','producer.receipt.json'))
        names.update(f'prefix-{index:02d}.json' for index in range(assignment['case']['accepted_stages']+1))
    if process:names.add('process.json')
    return names

def physical_validate_formal_shard_science(out,lease,common):
    assignment=physical_validate_formal_lease(lease,common)
    raw=read(out/'scientific.shard.json');science=environment.strict(raw)
    completion=environment.strict(read(out/'completion.json'))
    common_sha=lease['common_manifest']['sha256']
    if (type(science)is not dict or set(science)!={'schema','mode','cycle','candidate','common_manifest_sha256',
        'shard_id','assignment','record','full_g3c_qualified','production_qualified'}
        or science['schema']!='GE_BEAM3_G3C_PHYSICAL_FORMAL_SHARD_SCIENCE_V1'
        or not exact_json(science['mode'],common['mode']) or not exact_json(science['cycle'],common['cycle'])
        or not exact_json(science['candidate'],common['candidate'])
        or science['common_manifest_sha256']!=common_sha or science['shard_id']!=lease['shard_id']
        or not exact_json(science['assignment'],assignment)
        or science['full_g3c_qualified']is not False or science['production_qualified']is not False):
        raise ValueError('formal shard science schema')
    record=science['record'];case=assignment['case'];case_id=case['case_id']
    if assignment['kind']=='history-producer':
        if type(record)is not dict or set(record)!={'kind','case_ordinal','history_assignment_index','history','transport','passed'}:
            raise ValueError('formal producer record schema')
        history_row=record['history'];transport=record['transport']
        if (record['kind']!='history-producer'
            or type(record['case_ordinal'])is not int or record['case_ordinal']!=assignment['case_ordinal']
            or type(record['history_assignment_index'])is not int
            or record['history_assignment_index']!=assignment['history_assignment_index'] or record['passed']is not True
            or type(history_row)is not dict or set(history_row)!={'kind','case_id','events','directional_errors','packets','passed'}
            or history_row['kind']!='history' or history_row['case_id']!=case_id
            or type(history_row['events'])is not int or history_row['events']!=case['accepted_stages']
            or history_row['passed']is not True
            or type(history_row['directional_errors'])is not list or len(history_row['directional_errors'])!=3
            or any(type(v)is not float or not math.isfinite(v) or not 0<=v<=1e-7 for v in history_row['directional_errors'])
            or type(history_row['packets'])is not list or len(history_row['packets'])!=case['accepted_stages']+1):
            raise ValueError('formal producer history')
        for prefix,row in enumerate(history_row['packets']):
            if (type(row)is not dict or set(row)!={'name','bytes','sha256','prefix'}
                or row['name']!=f'prefix-{prefix:02d}.json' or type(row['prefix'])is not int or row['prefix']!=prefix
                or type(row['bytes'])is not int or row['bytes']<=0):raise ValueError('formal producer packet row')
            sha_value(row['sha256']);actual=packet_descriptor(out/row['name'])
            if actual['bytes']!=row['bytes'] or actual['sha256']!=row['sha256']:
                raise ValueError('formal producer packet bytes')
        if (type(transport)is not dict or set(transport)!={'accepted_diagnostic_sha256','definition_sha256',
            'expanded_sha256','programs_sha256','final_state_sha256','case_id','graph','variant','force_scale',
            'common_motion','immutable','passed'} or transport['case_id']!=case_id
            or transport['graph']!=case['graph'] or transport['variant']!=case['variant']
            or transport['force_scale']!=case['force_scale'] or transport['common_motion']!=case['common_motion']
            or transport['immutable']is not True
            or transport['passed']is not True):raise ValueError('formal transport record')
        for name in ('accepted_diagnostic_sha256','definition_sha256','expanded_sha256','programs_sha256','final_state_sha256'):
            sha_value(transport[name])
        if sha256(read(out/'accepted-diagnostic.json')).hexdigest()!=transport['accepted_diagnostic_sha256']:
            raise ValueError('formal diagnostic bytes')
    else:
        if (type(record)is not dict or set(record)!={'kind','case_ordinal','prefix_start','prefix_stop',
            'assignment_indexes','records','fresh_owners','passed'} or record['kind']!='prefix-range'
            or type(record['case_ordinal'])is not int or record['case_ordinal']!=assignment['case_ordinal']
            or type(record['prefix_start'])is not int or record['prefix_start']!=assignment['prefix_start']
            or type(record['prefix_stop'])is not int or record['prefix_stop']!=assignment['prefix_stop']
            or not exact_json(record['assignment_indexes'],assignment['assignment_indexes'])
            or type(record['fresh_owners'])is not int or record['fresh_owners']!=assignment['prefix_stop']-assignment['prefix_start']
            or type(record['records'])is not list or len(record['records'])!=record['fresh_owners']
            or record['passed']is not True):raise ValueError('formal prefix shard record')
        for offset,row in enumerate(record['records']):
            prefix=assignment['prefix_start']+offset;index=assignment['assignment_indexes'][offset]
            expected_record=dict(kind='prefix',case_id=case_id,prefix=prefix,
                input_sha256=lease['input_packets'][f'prefix-{prefix:02d}']['sha256'],
                final_sha256=lease['input_packets']['final']['sha256'],passed=True)
            if (type(row)is not dict or set(row)!={'assignment_index','record'}
                or type(row['assignment_index'])is not int or row['assignment_index']!=index
                or not exact_json(row['record'],expected_record)):
                raise ValueError('formal prefix record association')
    expected_completion=dict(shard_id=lease['shard_id'],assignment=assignment,
        scientific=fingerprint(raw),passed=True)
    if not exact_json(completion,expected_completion):raise ValueError('formal shard completion')
    return science

def physical_validate_formal_process(out,lease):
    expected_files=physical_formal_expected_files(lease['assignment'])
    expected_directories={'pytest'} if (out/'pytest').is_dir() else set()
    physical_closed_world(out,expected_files,expected_directories)
    value=environment.strict(read(out/'process.json'))
    if (type(value)is not dict or set(value)!={'status','active_processes','peak_tree_bytes','drained',
        'elapsed_seconds','kind','shard_id','assignment_sha256','returncode','files'}
        or value['status']!='PASSED' or type(value['active_processes'])is not int or value['active_processes']!=0
        or value['drained']is not True or type(value['peak_tree_bytes'])is not int
        or not 0<=value['peak_tree_bytes']<=MEMORY or type(value['elapsed_seconds'])is not float
        or not math.isfinite(value['elapsed_seconds']) or not 0<=value['elapsed_seconds']<600
        or value['kind']!='GE_BEAM3_G3C_PHYSICAL_FORMAL_SHARD_PROCESS_V1'
        or value['shard_id']!=lease['shard_id']
        or value['assignment_sha256']!=sha256(canonical(lease['assignment'])).hexdigest()
        or type(value['returncode'])is not int or value['returncode']!=0 or type(value['files'])is not dict):
        raise ValueError('formal accepted process')
    actual={path.name:fingerprint(read(path)) for path in sorted(out.iterdir())
            if path.is_file() and path.name!='process.json'}
    if not exact_json(value['files'],actual):raise ValueError('formal process file DAG')
    return value

def physical_formal_child(root,common_path,shard_id,review,packets,watchdog,deadline,producer_receipt=None):
    if time.monotonic()>=deadline:raise ValueError('formal measurement wave deadline')
    common_raw=read(common_path);common_value=environment.strict(common_raw)
    common=physical_validate_formal_common(common_value)
    selected=[row for row in common['partition']['body']['shards'] if row['shard_id']==shard_id]
    if len(selected)!=1:raise ValueError('formal child shard selection')
    out=Path(root)/('shard-'+shard_id);out.mkdir()
    with (out/'review.json').open('xb')as stream:stream.write(review)
    lease=dict(kind='G3C_PHYSICAL_PRIVATE_DEVELOPMENT',
        schema='GE_BEAM3_G3C_PHYSICAL_FORMAL_SHARD_LEASE_V1',run_id=str(uuid.uuid4()),
        mode=common['mode'],cycle=common['cycle'],common_manifest=physical_formal_manifest_descriptor(common_path),
        shard_id=shard_id,assignment=selected[0]['assignment'],input_packets=packets)
    if lease['assignment']['kind']=='prefix-range':
        if producer_receipt is None:raise ValueError('formal replay producer receipt required')
        lease['producer_receipt']=producer_receipt
    elif producer_receipt is not None:raise ValueError('formal producer cannot inherit receipt')
    physical_validate_formal_lease(lease,common);write(out/'lease.json',lease)
    lease_sha=sha256(read(out/'lease.json')).hexdigest();job=job_type()(MEMORY);process=None
    record=dict(status='FAILED');watchdog.attach(job)
    try:
        env=dict(os.environ,**{key:'1' for key in THREADS})
        env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTEST_ADDOPTS='',PYTHONDONTWRITEBYTECODE='1')
        with (out/'stdout.log').open('xb')as stdout,(out/'stderr.log').open('xb')as stderr:
            started=time.monotonic()
            process=job.launch([sys.executable,'-I','-S','-B','-u',str(Path(__file__).resolve()),
                '--physical-formal-worker',str(out),lease_sha],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
            record=monitor(job,process,lambda:((out/'stdout.log').stat().st_size,(out/'stderr.log').stat().st_size),start=started)
        if time.monotonic()>=deadline:record['status']='RESOURCE_BLOCKED'
        if record['status']=='PASSED':
            physical_validate_formal_shard_science(out,lease,common)
            if lease['assignment']['kind']=='history-producer':
                write(out/'producer.receipt.json',physical_formal_producer_receipt(out,lease,common))
    except BaseException as exc:
        record.update(status='FAILED_EVIDENCE',exception=type(exc).__name__)
    finally:
        if job.accounting()[1]:job.terminate()
        record['active_processes']=job.accounting()[1]
        if record['active_processes']:record['status']='FAILED_TO_DRAIN'
        job.close();watchdog.detach(job)
        record.update(kind='GE_BEAM3_G3C_PHYSICAL_FORMAL_SHARD_PROCESS_V1',shard_id=shard_id,
            assignment_sha256=sha256(canonical(lease['assignment'])).hexdigest(),
            returncode=None if process is None else process.poll(),
            files={path.name:fingerprint(path.read_bytes()) for path in out.iterdir() if path.is_file()})
        write(out/'process.json',record)
    if record.get('status')=='PASSED':physical_validate_formal_process(out,lease)
    return record,out

def physical_formal_measurement_batch(root,common_path,specs,review,packet_bindings,watchdog,deadline):
    """Run one bounded group; all launched children reach a terminal state."""
    if type(specs)is not list or not 1<=len(specs)<=3:raise ValueError('formal measurement batch size')
    if time.monotonic()>=deadline:raise ValueError('formal measurement wave deadline')
    results={}
    with ThreadPoolExecutor(max_workers=len(specs))as pool:
        futures={pool.submit(physical_formal_child,root,common_path,spec['shard_id'],review,
            packet_bindings[spec['shard_id']]['packets'],watchdog,deadline,
            packet_bindings[spec['shard_id']].get('producer_receipt')):spec for spec in specs}
        for future,spec in futures.items():
            try:record,out=future.result();results[spec['shard_id']]=(record,out)
            except BaseException as exc:
                results[spec['shard_id']]=(dict(status='FAILED_EVIDENCE',exception=type(exc).__name__),None)
    return [results[spec['shard_id']] for spec in specs]

def execute_physical_formal_measurement(args,watchdog,expected):
    """Disposable timing rehearsal; it cannot publish or classify formal science."""
    started=time.monotonic();deadline=started+1760
    partition=physical_formal_measurement_partition();runtime=physical_support().runtime_identity()
    common_value=physical_formal_common_manifest(expected[0],expected[1],environment.strict(expected[2]),
        runtime,partition,cycle=0)
    root=Path(tempfile.mkdtemp(prefix='anyfesolver-g3c-physical-formal-measurement-'))
    print('DIAGNOSTICS '+str(root),flush=True)
    common_path=root/'common.json';write(common_path,common_value)
    common=physical_validate_formal_common(environment.strict(read(common_path)))
    specs=common['partition']['body']['shards'];results={};passed=True;validation_sha=None
    try:
        producers=[spec for spec in specs if spec['assignment']['kind']=='history-producer']
        bindings={spec['shard_id']:dict(packets={}) for spec in producers}
        for offset in range(0,len(producers),3):
            batch=producers[offset:offset+3]
            for spec,result in zip(batch,physical_formal_measurement_batch(root,common_path,batch,
                    expected[2],bindings,watchdog,deadline)):
                results[spec['shard_id']]=result
            watchdog.check()
        if any(record.get('status')!='PASSED' or out is None for record,out in results.values()):
            passed=False
        producer_receipts={}
        if passed:
            for spec in producers:
                out=results[spec['shard_id']][1];descriptor=packet_descriptor(out/'producer.receipt.json')
                body=physical_validate_formal_producer_receipt(out/'producer.receipt.json',common)
                producer_receipts[spec['assignment']['case_ordinal']]=(descriptor,body)
        replays=[spec for spec in specs if spec['assignment']['kind']=='prefix-range']
        replay_bindings={}
        if passed:
            for spec in replays:
                assignment=spec['assignment'];receipt_descriptor,receipt=producer_receipts[assignment['case_ordinal']]
                packets={f'prefix-{prefix:02d}':receipt['packets'][f'prefix-{prefix:02d}']
                    for prefix in range(assignment['prefix_start'],assignment['prefix_stop'])}
                packets['final']=receipt['packets'][f"prefix-{assignment['case']['accepted_stages']:02d}"]
                replay_bindings[spec['shard_id']]=dict(packets=packets,producer_receipt=receipt_descriptor)
            for offset in range(0,len(replays),3):
                batch=replays[offset:offset+3]
                for spec,result in zip(batch,physical_formal_measurement_batch(root,common_path,batch,
                        expected[2],replay_bindings,watchdog,deadline)):
                    results[spec['shard_id']]=result
                watchdog.check()
                if any(results[spec['shard_id']][0].get('status')!='PASSED' for spec in batch):
                    passed=False;break
        if passed and len(results)==len(specs):
            sciences=[]
            for spec in specs:
                out=results[spec['shard_id']][1]
                lease=environment.strict(read(out/'lease.json'))
                physical_validate_formal_process(out,lease)
                sciences.append(physical_validate_formal_shard_science(out,lease,common))
            validation=physical_formal_scientific_union(sciences,common)
            validation_sha=validation['self_sha256']
            if authority(args.review,args.review_sha256,'g3c-physical')!=expected:
                raise ValueError('formal measurement final authority')
            if time.monotonic()>=deadline:raise ValueError('formal measurement wave deadline')
        else:passed=False
    except BaseException as exc:
        passed=False;failure=type(exc).__name__
    summaries=[]
    for spec in specs:
        record,out=results.get(spec['shard_id'],(dict(status='NOT_LAUNCHED'),None))
        summaries.append(dict(shard_id=spec['shard_id'],assignment_sha256=spec['assignment_sha256'],
            status=record.get('status'),elapsed_seconds=record.get('elapsed_seconds'),
            peak_tree_bytes=record.get('peak_tree_bytes'),active_processes=record.get('active_processes'),
            output=None if out is None else str(out)))
    process=dict(schema='GE_BEAM3_G3C_PHYSICAL_FORMAL_MEASUREMENT_PROCESS_V1',candidate=expected[0],
        mode='measurement',classification='NONCLASSIFYING',partition_sha256=partition['self_sha256'],
        required_shards=len(specs),terminal_shards=sum(row['status']!='NOT_LAUNCHED' for row in summaries),
        validation_sha256=validation_sha,passed=passed,elapsed_seconds=time.monotonic()-started,
        active_processes=sum((row['active_processes'] or 0) for row in summaries),shards=summaries,
        formal_execution_authorized=False,
        full_g3c_qualified=False,production_qualified=False)
    if not passed:process['failure']=locals().get('failure','SHARD_FAILURE')
    write(root/'measurement-process.json',process);print(canonical(process).decode(),flush=True)
    return int(not passed)

def physical_proof_checker_worker(root,aggregate_sha):
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise ValueError('isolated compressed checker required')
    path=root/'aggregate.pending.json';raw=read(path)
    if sha256(raw).hexdigest()!=aggregate_sha or canonical(environment.strict(raw))!=raw:
        raise ValueError('compressed checker aggregate authority')
    checker_path=ROOT/PHYSICAL_PROOF_COMPRESSED_CHECKER
    if sha256(read(checker_path).replace(b'\r\n',b'\n')).hexdigest()!=PHYSICAL_PROOF_COMPRESSED_CHECKER_SHA:
        raise ValueError('compressed checker source authority')
    spec=importlib.util.spec_from_file_location('ge_beam3_g3c_proof_compressed_independent_checker',checker_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.verify(environment.strict(raw))
    if read(path)!=raw:raise ValueError('compressed aggregate changed during check')
    sys.stdout.buffer.write(canonical(result));sys.stdout.buffer.flush();return 0

def physical_proof_checker_child(root,replica,aggregate_sha,watchdog):
    out=root/f'checker-{replica}';out.mkdir();job=job_type()(MEMORY);watchdog.attach(job);process=None
    try:
        with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
            started=time.monotonic();process=job.launch([sys.executable,'-I','-S','-B','-u',
                str(Path(__file__).resolve()),'--physical-proof-checker',str(root),aggregate_sha],
                cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=stdout,stderr=stderr)
            record=monitor(job,process,lambda:((out/'stdout.log').stat().st_size,
                (out/'stderr.log').stat().st_size),start=started)
        if record['status']=='PASSED':
            raw=read(out/'stdout.log');value=environment.strict(raw)
            if canonical(value)!=raw or value.get('independently_verified') is not True:
                raise ValueError('compressed checker output')
            with (root/f'checker-{replica}.json').open('xb') as stream:stream.write(raw)
    except BaseException as exc:
        record=dict(status='FAILED_EVIDENCE',exception=type(exc).__name__,active_processes=job.accounting()[1])
    finally:
        if job.accounting()[1]:job.terminate()
        record['active_processes']=job.accounting()[1]
        if record['active_processes']:record['status']='FAILED_TO_DRAIN'
        job.close();watchdog.detach(job)
    return record

def execute_physical_proof_compressed(args,watchdog,expected):
    """Execute one bounded successor cycle; each three-child batch is a wave."""
    started=time.monotonic();partition=physical_proof_compressed_partition()
    runtime=physical_support().runtime_identity()
    common_value=physical_formal_common_manifest(expected[0],expected[1],environment.strict(expected[2]),
        runtime,partition,cycle=args.proof_cycle)
    root=Path(tempfile.mkdtemp(prefix=f'anyfesolver-g3c-proof-compressed-cycle{args.proof_cycle}-'))
    print('DIAGNOSTICS '+str(root),flush=True);common_path=root/'common.json';write(common_path,common_value)
    common=physical_validate_formal_common(environment.strict(read(common_path)))
    specs=common['partition']['body']['shards'];results={};passed=True;failure=None
    try:
        producers=[spec for spec in specs if spec['assignment']['kind']=='history-producer']
        bindings={spec['shard_id']:dict(packets={}) for spec in producers}
        for offset in range(0,len(producers),3):
            watchdog.renew_wave();batch=producers[offset:offset+3];deadline=time.monotonic()+1760
            for spec,result in zip(batch,physical_formal_measurement_batch(root,common_path,batch,
                    expected[2],bindings,watchdog,deadline)):
                results[spec['shard_id']]=result
            if any(results[spec['shard_id']][0].get('status')!='PASSED' for spec in batch):
                passed=False;break
        receipts={}
        if passed:
            for spec in producers:
                out=results[spec['shard_id']][1]
                body=physical_validate_formal_producer_receipt(out/'producer.receipt.json',common)
                receipts[spec['assignment']['case_ordinal']]=(packet_descriptor(out/'producer.receipt.json'),body)
        replays=[spec for spec in specs if spec['assignment']['kind']=='prefix-range']
        replay_bindings={}
        if passed:
            for spec in replays:
                assignment=spec['assignment'];descriptor,receipt=receipts[assignment['case_ordinal']]
                packets={f'prefix-{prefix:02d}':receipt['packets'][f'prefix-{prefix:02d}']
                    for prefix in range(assignment['prefix_start'],assignment['prefix_stop'])}
                packets['final']=receipt['packets'][f"prefix-{assignment['case']['accepted_stages']:02d}"]
                replay_bindings[spec['shard_id']]=dict(packets=packets,producer_receipt=descriptor)
            for offset in range(0,len(replays),3):
                watchdog.renew_wave();batch=replays[offset:offset+3];deadline=time.monotonic()+1760
                for spec,result in zip(batch,physical_formal_measurement_batch(root,common_path,batch,
                        expected[2],replay_bindings,watchdog,deadline)):
                    results[spec['shard_id']]=result
                if any(results[spec['shard_id']][0].get('status')!='PASSED' for spec in batch):
                    passed=False;break
        if passed and len(results)==len(specs):
            sciences=[]
            for spec in specs:
                out=results[spec['shard_id']][1];lease=environment.strict(read(out/'lease.json'))
                physical_validate_formal_process(out,lease)
                sciences.append(physical_validate_formal_shard_science(out,lease,common))
            union=physical_formal_scientific_union(sciences,common)
            aggregate=physical_proof_compressed_aggregate(union,common)
            write(root/'aggregate.pending.json',aggregate);aggregate_sha=sha256(read(root/'aggregate.pending.json')).hexdigest()
            watchdog.renew_wave()
            with ThreadPoolExecutor(max_workers=2) as pool:
                checks=list(pool.map(lambda replica:physical_proof_checker_child(
                    root,replica,aggregate_sha,watchdog),(1,2)))
            if any(row.get('status')!='PASSED' for row in checks):raise ValueError('compressed checker process')
            if read(root/'checker-1.json')!=read(root/'checker-2.json'):
                raise ValueError('compressed checker disagreement')
            if authority(args.review,args.review_sha256,'g3c-physical')!=expected:
                raise ValueError('compressed final authority')
            os.rename(root/'aggregate.pending.json',root/'scientific.json')
        else:passed=False
    except BaseException as exc:
        passed=False;failure=type(exc).__name__
    summaries=[]
    for spec in specs:
        record,out=results.get(spec['shard_id'],(dict(status='NOT_LAUNCHED'),None))
        summaries.append(dict(shard_id=spec['shard_id'],status=record.get('status'),
            elapsed_seconds=record.get('elapsed_seconds'),peak_tree_bytes=record.get('peak_tree_bytes'),
            active_processes=record.get('active_processes'),output=None if out is None else str(out)))
    process=dict(schema='GE_BEAM3_G3C_PROOF_COMPRESSED_PROCESS_V2',cycle=args.proof_cycle,
        candidate=expected[0],partition_sha256=partition['self_sha256'],required_shards=len(specs),
        terminal_shards=sum(row['status']!='NOT_LAUNCHED' for row in summaries),passed=passed,
        failure=failure,elapsed_seconds=time.monotonic()-started,
        active_processes=sum((row['active_processes'] or 0) for row in summaries),shards=summaries,
        full_g3c_qualified=False,production_qualified=False)
    write(root/'process.json',process);print(canonical(process).decode(),flush=True);return int(not passed)

def physical_formal_scientific_union(shard_sciences,common):
    """Stdlib-only, order-independent flattening back to assignment records."""
    if type(shard_sciences)is not list or not shard_sciences:raise ValueError('formal shard science inventory')
    by_id={row.get('shard_id'):row for row in shard_sciences if type(row)is dict}
    expected_shards=common['partition']['body']['shards']
    if len(by_id)!=len(shard_sciences) or set(by_id)!={row['shard_id'] for row in expected_shards}:
        raise ValueError('formal shard science coverage')
    common_value=dict(body=common,self_sha256=sha256(canonical(common)).hexdigest())
    common_sha=sha256(canonical(common_value)).hexdigest()
    assignments=common['assignments'];records={};transports=[]
    for spec in expected_shards:
        science=by_id[spec['shard_id']];assignment=spec['assignment']
        if (set(science)!={'schema','mode','cycle','candidate','common_manifest_sha256','shard_id','assignment',
            'record','full_g3c_qualified','production_qualified'}
            or science['schema']!='GE_BEAM3_G3C_PHYSICAL_FORMAL_SHARD_SCIENCE_V1'
            or not exact_json(science['mode'],common['mode']) or not exact_json(science['cycle'],common['cycle'])
            or not exact_json(science['candidate'],common['candidate'])
            or science['common_manifest_sha256']!=common_sha
            or science['shard_id']!=spec['shard_id'] or not exact_json(science['assignment'],assignment)
            or science['full_g3c_qualified']is not False or science['production_qualified']is not False):
            raise ValueError('formal shard union schema')
        row=science['record']
        if assignment['kind']=='history-producer':
            if (type(row)is not dict or set(row)!={'kind','case_ordinal','history_assignment_index','history','transport','passed'}
                or row['kind']!='history-producer' or type(row['case_ordinal'])is not int
                or row['case_ordinal']!=assignment['case_ordinal']
                or type(row['history_assignment_index'])is not int
                or row['history_assignment_index']!=assignment['history_assignment_index'] or row['passed']is not True):
                raise ValueError('formal producer union record')
            index=assignment['history_assignment_index'];payload=row['history'];transport=row['transport']
            if (type(payload)is not dict or set(payload)!={'kind','case_id','events','directional_errors','packets','passed'}
                or payload['kind']!='history' or payload['case_id']!=assignment['case']['case_id']
                or type(payload['events'])is not int or payload['events']!=assignment['case']['accepted_stages']
                or type(payload['directional_errors'])is not list or len(payload['directional_errors'])!=3
                or any(type(v)is not float or not math.isfinite(v) or not 0<=v<=1e-7 for v in payload['directional_errors'])
                or type(payload['packets'])is not list or len(payload['packets'])!=payload['events']+1
                or payload['passed']is not True):raise ValueError('formal producer union history')
            for prefix,packet_row in enumerate(payload['packets']):
                if (type(packet_row)is not dict or set(packet_row)!={'name','bytes','sha256','prefix'}
                    or packet_row['name']!=f'prefix-{prefix:02d}.json' or type(packet_row['bytes'])is not int
                    or packet_row['bytes']<=0 or type(packet_row['prefix'])is not int or packet_row['prefix']!=prefix):
                    raise ValueError('formal producer union packet')
                sha_value(packet_row['sha256'])
            if (type(transport)is not dict or set(transport)!={'accepted_diagnostic_sha256','definition_sha256',
                'expanded_sha256','programs_sha256','final_state_sha256','case_id','graph','variant','force_scale',
                'common_motion','immutable','passed'} or transport['case_id']!=assignment['case']['case_id']
                or transport['graph']!=assignment['case']['graph'] or transport['variant']!=assignment['case']['variant']
                or transport['force_scale']!=assignment['case']['force_scale']
                or transport['common_motion']!=assignment['case']['common_motion']
                or transport['immutable']is not True or transport['passed']is not True):
                raise ValueError('formal producer union transport')
            for name in ('accepted_diagnostic_sha256','definition_sha256','expanded_sha256','programs_sha256','final_state_sha256'):
                sha_value(transport[name])
            transports.append(transport)
            if index in records:raise ValueError('duplicate formal history record')
            records[index]=payload
        else:
            if (type(row)is not dict or set(row)!={'kind','case_ordinal','prefix_start','prefix_stop',
                'assignment_indexes','records','fresh_owners','passed'} or row['kind']!='prefix-range'
                or type(row['case_ordinal'])is not int or row['case_ordinal']!=assignment['case_ordinal']
                or type(row['prefix_start'])is not int or row['prefix_start']!=assignment['prefix_start']
                or type(row['prefix_stop'])is not int or row['prefix_stop']!=assignment['prefix_stop']
                or not exact_json(row['assignment_indexes'],assignment['assignment_indexes'])
                or type(row['fresh_owners'])is not int or row['fresh_owners']!=row['prefix_stop']-row['prefix_start']
                or type(row['records'])is not list or len(row['records'])!=row['fresh_owners']
                or row['passed']is not True):raise ValueError('formal prefix union record')
            for item in row['records']:
                if type(item)is not dict or set(item)!={'assignment_index','record'} or type(item['assignment_index'])is not int:
                    raise ValueError('formal prefix union item')
                index=item['assignment_index']
                prefix_assignment=assignments[index] if 0<=index<len(assignments) else None;payload=item['record']
                if (type(prefix_assignment)is not dict or prefix_assignment.get('kind')!='prefix'
                    or type(payload)is not dict or set(payload)!={'kind','case_id','prefix','input_sha256','final_sha256','passed'}
                    or payload['kind']!='prefix' or payload['case_id']!=prefix_assignment['case']['case_id']
                    or type(payload['prefix'])is not int or payload['prefix']!=prefix_assignment['prefix']
                    or payload['passed']is not True):raise ValueError('formal prefix union payload')
                sha_value(payload['input_sha256']);sha_value(payload['final_sha256'])
                if index in records:raise ValueError('duplicate formal prefix record')
                records[index]=payload
    expected_indexes=({spec['assignment']['history_assignment_index'] for spec in expected_shards
                       if spec['assignment']['kind']=='history-producer'}|
        {index for spec in expected_shards if spec['assignment']['kind']=='prefix-range'
         for index in spec['assignment']['assignment_indexes']})
    if set(records)!=expected_indexes:raise ValueError('formal union record coverage')
    if common['mode']=='formal' and expected_indexes!=set(range(3825)):
        raise ValueError('formal complete assignment coverage')
    made=[]
    for index in sorted(records):
        assignment=assignments[index];record=records[index]
        expected_kind=assignment['kind']
        if (expected_kind=='history' and (record.get('kind')!='history' or record.get('case_id')!=assignment['case']['case_id'])):
            raise ValueError('formal history union association')
        if (expected_kind=='prefix' and (record.get('kind')!='prefix' or record.get('case_id')!=assignment['case']['case_id']
            or record.get('prefix')!=assignment['prefix'])):raise ValueError('formal prefix union association')
        science=dict(schema='GE_BEAM3_G3C_PHYSICAL_NODE_SCIENCE_V1',candidate=common['candidate'],lane='formal',
            assignment_index=index,assignment=assignment,records=[record],
            full_g3c_qualified=False,production_qualified=False)
        made.append(dict(assignment_index=index,science_sha256=sha256(canonical(science)).hexdigest(),science=science))
    order={row['case_id']:index for index,row in enumerate(common['cases'])}
    if len(transports)!=len({row['case_id'] for row in transports}):raise ValueError('duplicate formal transport record')
    transports.sort(key=lambda row:order[row['case_id']])
    value=dict(schema='GE_BEAM3_G3C_PHYSICAL_FORMAL_AGGREGATE_V1',candidate=common['candidate'],
        mode=common['mode'],assignment_inventory_sha256=PHYSICAL_FORMAL_ASSIGNMENT_INVENTORY_SHA,
        records=made,transport_records=transports,passed=True,
        terminal=('COMPLETE_GE_BEAM3_G3C_PHYSICAL_FORMAL_'+common['mode'].upper()+'_ONLY'),
        full_g3c_qualified=False,production_qualified=False)
    value['self_sha256']=sha256(canonical(value)).hexdigest();return value

def physical_proof_compressed_aggregate(formal_union,common):
    """Convert only the exact executed basis into truthful derived evidence."""
    if (common.get('mode')!='compressed' or common.get('cycle') not in (1,2)
        or formal_union.get('mode')!='compressed' or formal_union.get('passed') is not True):
        raise ValueError('compressed aggregate authority')
    records={row['assignment_index']:row['science']['records'][0]
             for row in formal_union['records']}
    associations={row['case']['case_id']:row for row in common['associations']}
    transports={row['case_id']:row for row in formal_union['transport_records']}
    histories=[]
    for case_id in proof_compressed.executed_case_ids():
        association=associations[case_id];payload=records[association['history_assignment_index']]
        transport=transports.get(case_id)
        if transport is None:raise ValueError('compressed transport coverage')
        histories.append(dict(case_id=case_id,events=association['case']['accepted_stages'],
            history_sha256=sha256(canonical(payload)).hexdigest(),
            final_sha256=payload['packets'][-1]['sha256'],
            transport_sha256=sha256(canonical(transport)).hexdigest(),passed=True))
    restarts=[]
    for row in proof_compressed.restart_plan():
        association=associations[row['case_id']]
        for prefix in row['prefixes']:
            payload=records[association['prefix_assignment_indexes'][prefix]]
            restarts.append(dict(case_id=row['case_id'],prefix=prefix,
                input_sha256=payload['input_sha256'],final_sha256=payload['final_sha256'],passed=True))
    execution=dict(schema='GE_BEAM3_G3C_PROOF_COMPRESSED_EXECUTION_V2',cycle=common['cycle'],
        candidate=common['candidate'],
        manifest_sha256=sha256(canonical(proof_compressed.manifest())).hexdigest(),
        histories=histories,restarts=restarts,
        prerequisite_receipts=[dict(name=name,sha256=digest)
            for name,digest in proof_compressed.PREREQUISITES])
    return proof_compressed.aggregate(execution)

def physical_formal_worker(out,lease_sha):
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise ValueError('isolated formal worker required')
    lease_raw=read(out/'lease.json')
    if sha256(lease_raw).hexdigest()!=lease_sha:raise ValueError('formal lease hash')
    lease=environment.strict(lease_raw);common_raw=validate_packet_descriptor(lease['common_manifest'])
    common_value=environment.strict(common_raw);common=physical_validate_formal_common(common_value)
    assignment=physical_validate_formal_lease(lease,common)
    if ((common['mode']=='measurement' and common['cycle']!=0)
        or (common['mode']=='compressed' and common['cycle'] not in (1,2))
        or common['mode'] not in ('measurement','compressed')):
        raise ValueError('formal execution authorization not frozen')
    if common['runtime_sha256']!=physical_support().runtime_identity():
        raise ValueError('formal runtime authority')
    expected=authority(out/'review.json',common['review_sha256'],'g3c-physical')
    if (not exact_json(common['candidate'],expected[0]) or not exact_json(common['inputs'],expected[1])
        or not exact_json(common['implementation_review'],environment.strict(expected[2]))):
        raise ValueError('formal common implementation authority')
    claim_attempt(out,lease['run_id'])
    if any(os.environ.get(key)!='1' for key in THREADS):raise ValueError('formal numerical thread environment')
    print('BEAM CHECKPOINT formal authority complete',flush=True)
    sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(CAPSULE.parent/'site')]
    from anyfesolver import _ge_beam3_g3c_physical_authority as runtime_authority
    runtime_authority.capture(canonical(common))
    import pytest
    node=PHYSICAL_FORMAL_TEST+'::test_physical_formal_shard_assignment'
    class Context:
        def __init__(self):self.collected=[];self.passed=[];self.record=None
        def pytest_collection_modifyitems(self,items):
            self.collected=[item.nodeid for item in items]
            if self.collected!=[node]:raise ValueError('formal collected inventory')
            module=items[0].module;module.ASSIGNMENT=assignment;module.OUTPUT_DIRECTORY=str(out)
            module.INPUT_PACKETS=lease['input_packets'];module.EXPECTED_RUNTIME_SHA256=common['runtime_sha256']
            module.COMMON_MANIFEST_PATH=lease['common_manifest']['path']
            module.COMMON_MANIFEST_BYTES=lease['common_manifest']['bytes']
            module.COMMON_MANIFEST_SHA256=lease['common_manifest']['sha256']
            self.module=module
        def pytest_runtest_logreport(self,report):
            if report.when=='call' and report.passed:self.passed.append(report.nodeid)
        def pytest_sessionfinish(self,session,exitstatus):
            self.record=getattr(self.module,'FORMAL_SHARD_RECORD',None) if hasattr(self,'module') else None
    context=Context();code=pytest.main(['-vv','-s','-p','no:cacheprovider','--basetemp',str(out/'pytest'),node],plugins=[context])
    if code or context.collected!=[node] or context.passed!=[node] or type(context.record)is not dict:return 1
    if authority(out/'review.json',common['review_sha256'],'g3c-physical')!=expected:
        raise ValueError('formal final authority changed')
    descriptor=physical_formal_manifest_descriptor(Path(lease['common_manifest']['path']))
    if not exact_json(descriptor,lease['common_manifest']):raise ValueError('formal common changed')
    science=dict(schema='GE_BEAM3_G3C_PHYSICAL_FORMAL_SHARD_SCIENCE_V1',mode=common['mode'],
        cycle=common['cycle'],candidate=common['candidate'],common_manifest_sha256=descriptor['sha256'],
        shard_id=lease['shard_id'],assignment=assignment,record=context.record,
        full_g3c_qualified=False,production_qualified=False)
    write(out/'scientific.shard.json',science)
    write(out/'completion.json',dict(shard_id=lease['shard_id'],assignment=assignment,
        scientific=fingerprint(read(out/'scientific.shard.json')),passed=True))
    physical_validate_formal_shard_science(out,lease,common)
    print('BEAM CHECKPOINT formal shard complete',flush=True);return 0

def physical_partition_manifest(inventory_rows=None):
    """Return the exact design-reviewed partition of the unchanged rehearsal."""
    rows=physical_inventory('rehearsal') if inventory_rows is None else inventory_rows
    if sha256(canonical(rows)).hexdigest()!=PHYSICAL_PARTITION_INVENTORY_SHA:
        raise ValueError('physical rehearsal inventory authority')
    partitions=[];covered=[]
    for partition_id,indices,digest in PHYSICAL_PARTITIONS:
        selected=[rows[index] for index in indices]
        if sha256(canonical(selected)).hexdigest()!=digest:
            raise ValueError('physical partition assignment authority')
        partitions.append(dict(assignments_sha256=digest,indices=list(indices),partition_id=partition_id))
        covered.extend(indices)
    if covered!=list(range(len(rows))) or len(covered)!=len(set(covered)):
        raise ValueError('physical partition coverage')
    value=dict(lane='rehearsal',partitions=partitions,
        schema='GE_BEAM3_G3C_PHYSICAL_REHEARSAL_PARTITION_V1',
        whole_inventory_sha256=PHYSICAL_PARTITION_INVENTORY_SHA)
    if sha256(canonical(value)).hexdigest()!=PHYSICAL_PARTITION_MANIFEST_SHA:
        raise ValueError('physical partition manifest authority')
    return value

def physical_partition_spec(partition_id,inventory_rows=None):
    manifest=physical_partition_manifest(inventory_rows)
    for ordinal,row in enumerate(manifest['partitions']):
        if row['partition_id']==partition_id:return ordinal,row
    raise ValueError('unregistered physical rehearsal partition')

def physical_partition_prerequisite_ids(partition_id):
    ordinal,_=physical_partition_spec(partition_id)
    return list(PHYSICAL_PARTITION_IDS[:ordinal])

def physical_correction_guard_segment_manifest(inventory_rows=None):
    rows=physical_inventory('rehearsal') if inventory_rows is None else inventory_rows
    segments=[];covered=[]
    for segment_id,indices,digest in PHYSICAL_CORRECTION_GUARD_SEGMENTS:
        if sha256(canonical([rows[index] for index in indices])).hexdigest()!=digest:
            raise ValueError('correction guard segment assignment authority')
        segments.append(dict(assignment_sha256=digest,indices=list(indices),segment_id=segment_id))
        covered.extend(indices)
    if covered!=list(range(90,234)) or len(covered)!=len(set(covered)):
        raise ValueError('correction guard segment coverage')
    value=dict(schema='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARD_SEGMENTS_V1',
        whole_inventory_sha256=PHYSICAL_PARTITION_INVENTORY_SHA,
        partition_manifest_sha256=PHYSICAL_PARTITION_MANIFEST_SHA,segments=segments)
    if sha256(canonical(value)).hexdigest()!=PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA:
        raise ValueError('correction guard segment manifest authority')
    return value

def physical_correction_guard_segment_spec(segment_id,inventory_rows=None):
    manifest=physical_correction_guard_segment_manifest(inventory_rows)
    for ordinal,row in enumerate(manifest['segments']):
        if row['segment_id']==segment_id:return ordinal,row
    raise ValueError('unregistered correction guard segment')

def inventory(lane,gate='b2-core'):
    if gate not in TESTS:raise ValueError('unregistered gate')
    if gate=='g3c-physical':return physical_inventory(lane)
    test_path,inventory_key=TESTS[gate]
    if gate=='g4a-station':
        names=list(G4A_NODES)
        tree=ast.parse(read(ROOT/test_path))
        actual=[n.name for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]
        if actual!=list(G4A_TESTS):raise ValueError('registered G4a test inventory changed')
        if lane=='smoke':names=names[:1]
        elif lane not in ('core','rehearsal','formal'):raise ValueError('unregistered G4a lane')
        return [test_path+'::'+name for name in names]
    if gate=='g4-completion':
        names=list(G4_NODES)
        tree=ast.parse(read(ROOT/test_path))
        actual=[n.name for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]
        if actual!=list(G4_TESTS):raise ValueError('registered G4 completion test inventory changed')
        if lane=='smoke':names=names[:1]
        elif lane not in ('core','rehearsal','formal'):raise ValueError('unregistered G4 completion lane')
        return [test_path+'::'+name for name in names]
    if gate=='g5-completion':
        names=list(G5_TESTS)
        tree=ast.parse(read(ROOT/test_path))
        actual=[n.name for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]
        if actual!=list(G5_TESTS):raise ValueError('registered G5 completion test inventory changed')
        if lane=='smoke':names=names[:1]
        elif lane not in ('core','rehearsal','formal'):raise ValueError('unregistered G5 completion lane')
        return [test_path+'::'+name for name in names]
    contract=environment.strict(read(ROOT/CONTRACT).replace(b'\r\n',b'\n'))
    names=(NUMERICAL_TESTS if gate=='q4-affine-numerical' else AFFINE_TESTS if gate=='q4-affine-exact' else
           Q4_TESTS if gate=='q4-audit' else contract['test_nodes'][inventory_key])
    tree=ast.parse(read(ROOT/test_path))
    actual=[n.name for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]
    if actual!=names+([NUMERICAL_SMOKE] if gate=='q4-affine-numerical' else []):raise ValueError('registered test inventory changed')
    if lane=='smoke':
        names=([names[0],NUMERICAL_SMOKE] if gate=='q4-affine-numerical' else
               [names[2],names[7]] if gate=='b2-core' else (names[:2] if gate in ('q4-audit','q4-affine-exact') else [names[0]]))
    elif lane!='core':raise ValueError('unregistered lane')
    return [test_path+'::'+name for name in names]

def verify_review(raw,digest,candidate,rows,gate='b2-core'):
    if sha256(raw).hexdigest()!=digest:raise ValueError('review hash')
    r=environment.strict(raw)
    if gate=='g3c-physical':
        expected_scope={'scope_id':SCOPE,'gate':gate,'subject_tree':candidate['tree'],
            'inputs_sha256':sha256(canonical(rows)).hexdigest(),'contract_sha256':PHYSICAL_PLAN_SHA,
            'partition_addendum_sha256':PHYSICAL_PARTITION_ADDENDUM_SHA,
            'formal_addendum_sha256':PHYSICAL_FORMAL_ADDENDUM_SHA,
            'formal_design_review_sha256':PHYSICAL_FORMAL_DESIGN_REVIEW_SHA,
            'formal_measurement_authorized':True,'formal_execution_authorized':False,
            'proof_compressed_plan_sha256':PHYSICAL_PROOF_COMPRESSED_PLAN_SHA,
            'proof_compressed_tool_sha256':PHYSICAL_PROOF_COMPRESSED_TOOL_SHA,
            'proof_compressed_checker_sha256':PHYSICAL_PROOF_COMPRESSED_CHECKER_SHA,
            'proof_compressed_initial_review_sha256':PHYSICAL_PROOF_COMPRESSED_INITIAL_REVIEW_SHA,
            'proof_compressed_measurement_sha256':PHYSICAL_PROOF_COMPRESSED_MEASUREMENT_SHA,
            'proof_compressed_partition_sha256':PHYSICAL_PROOF_COMPRESSED_PARTITION_SHA,
            'proof_compressed_checker_initial_review_sha256':PHYSICAL_PROOF_COMPRESSED_CHECKER_INITIAL_REVIEW_SHA,
            'proof_compressed_cycle_authorized':True,
            'execution_authorized':True,'full_g3c_qualified':False,'production_qualified':False}
        if not exact_json(candidate,PHYSICAL_PREDECESSOR):
            expected_scope.update(correction_addendum_sha256=PHYSICAL_CORRECTION_ADDENDUM_SHA,
                correction_design_review_sha256=PHYSICAL_CORRECTION_REVIEW_SHA,
                correction_revision_sha256=PHYSICAL_CORRECTION_REVISION_SHA,
                correction_revision_review_sha256=PHYSICAL_CORRECTION_REVISION_REVIEW_SHA,
                correction_partition_recovery_sha256=PHYSICAL_CORRECTION_RECOVERY_SHA,
                correction_partition_recovery_review_sha256=PHYSICAL_CORRECTION_RECOVERY_REVIEW_SHA,
                correction_guard_segment_manifest_sha256=PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA,
                predecessor_commit=PHYSICAL_PREDECESSOR['commit'],
                predecessor_runtime_sha256=PHYSICAL_PREDECESSOR_RUNTIME_SHA,
                correction_inheritance_authorized=True,correction_partition_recovery_authorized=True)
        if (set(r)!={'decision','findings','reviewer','scope','subject_commit'}
            or r['decision']!='ACCEPTED_G3C_PHYSICAL_IMPLEMENTATION_FOR_BOUNDED_DEVELOPMENT'
            or r['findings'] or r['reviewer'].get('independent') is not True
            or r['subject_commit']!=candidate['commit'] or r['scope']!=expected_scope):
            raise ValueError('physical implementation review authority')
        return r
    if (set(r)!={'decision','findings','reviewer','scope','subject_commit'}
        or r['decision']!='ACCEPTED_GE_BEAM3_REGISTERED_GATE_FOR_BOUNDED_EXECUTION' or r['findings']
        or r['reviewer'].get('independent') is not True
        or r['subject_commit']!=candidate['commit']
        or r['scope']!={'scope_id':SCOPE,'gate':gate,'subject_tree':candidate['tree'],
                        'inputs_sha256':sha256(canonical(rows)).hexdigest(),
                        'contract_sha256':G5_PLAN_SHA if gate=='g5-completion' else G4_CONTRACT_SHA if gate=='g4-completion' else G4A_CONTRACT_SHA if gate=='g4a-station' else
                                         NUMERICAL_PLAN_SHA if gate=='q4-affine-numerical' else AFFINE_PLAN_SHA if gate=='q4-affine-exact' else
                                         Q4_PLAN_SHA if gate=='q4-audit' else CONTRACT_SHA}):
        raise ValueError('implementation review authority')
    return r

def authority(review_path,review_sha,gate='b2-core',*,observation_capture=None):
    if gate not in TESTS:raise ValueError('unregistered gate')
    if git('status','--porcelain','--untracked-files=all'):raise ValueError('dirty candidate')
    candidate=dict(commit=git('rev-parse','HEAD'),tree=git('rev-parse','HEAD^{tree}'))
    if gate=='g5-completion':
        if git('rev-parse',G5_BASE+'^{tree}')!=G5_BASE_TREE:raise ValueError('G5 base tree')
        git('merge-base','--is-ancestor',G5_BASE,'HEAD')
        changed=set(filter(None,git('diff','--name-only',G5_BASE,'HEAD').splitlines()))
        if changed!=G5_IMPLEMENTATION_PATHS:raise ValueError('G5 completion implementation extent changed')
        if sha256(read(ROOT/G5_PLAN).replace(b'\r\n',b'\n')).hexdigest()!=G5_PLAN_SHA:
            raise ValueError('G5 frozen contract changed')
        inventory('smoke',gate);inventory('core',gate);inventory('rehearsal',gate);inventory('formal',gate)
        rows=inputs();raw=read(review_path);verify_review(raw,review_sha,candidate,rows,gate)
        environment.verify(CAPSULE,CAPSULE_SHA)
        return candidate,rows,raw
    if gate=='g4-completion':
        if git('rev-parse',G4_BASE+'^{tree}')!=G4_BASE_TREE:raise ValueError('G4 base tree')
        git('merge-base','--is-ancestor',G4_BASE,'HEAD')
        changed=set(filter(None,git('diff','--name-only',G4_BASE,'HEAD').splitlines()))
        if changed!=G4_IMPLEMENTATION_PATHS:raise ValueError('G4 completion implementation extent changed')
        for path,digest in ((G4_PLAN,G4_PLAN_SHA),(G4_CONTRACT,G4_CONTRACT_SHA),
                            (G4_DESIGN_REVIEW,G4_DESIGN_REVIEW_SHA),(JOB,JOB_SHA)):
            if sha256(read(ROOT/path).replace(b'\r\n',b'\n')).hexdigest()!=digest:
                raise ValueError('G4 frozen authority input')
        design=environment.strict(read(ROOT/G4_DESIGN_REVIEW).replace(b'\r\n',b'\n'))
        if (set(design)!={'decision','findings','reviewer','scope','subject_commit'}
            or design['decision']!='ACCEPTED_GE_BEAM3_G4_COMPLETION_CONTRACT_FOR_PRIVATE_IMPLEMENTATION_ONLY'
            or design['findings'] or design['reviewer'].get('independent') is not True
            or design['subject_commit']!='f71d820447802fca0898e25834d9134769194c68'
            or design['scope'].get('contract_sha256')!=G4_CONTRACT_SHA
            or design['scope'].get('execution_authorized') is not False
            or design['scope'].get('production_qualified') is not False):
            raise ValueError('G4 completion design review authority')
        inventory('smoke',gate);inventory('core',gate);inventory('rehearsal',gate);inventory('formal',gate)
        rows=inputs();raw=read(review_path);verify_review(raw,review_sha,candidate,rows,gate)
        environment.verify(CAPSULE,CAPSULE_SHA)
        return candidate,rows,raw
    if gate=='g4a-station':
        if git('rev-parse',G4A_BASE+'^{tree}')!=G4A_BASE_TREE:raise ValueError('G4a base tree')
        git('merge-base','--is-ancestor',G4A_BASE,'HEAD')
        changed=set(filter(None,git('diff','--name-only',G4A_BASE,'HEAD').splitlines()))
        if changed!=G4A_IMPLEMENTATION_PATHS:raise ValueError('G4a implementation extent changed')
        for path,digest in ((G4A_PLAN,G4A_PLAN_SHA),(G4A_CONTRACT,G4A_CONTRACT_SHA),
                            (G4A_DESIGN_REVIEW,G4A_DESIGN_REVIEW_SHA),(JOB,JOB_SHA)):
            if sha256(read(ROOT/path).replace(b'\r\n',b'\n')).hexdigest()!=digest:
                raise ValueError('G4a frozen authority input')
        design=environment.strict(read(ROOT/G4A_DESIGN_REVIEW).replace(b'\r\n',b'\n'))
        if (set(design)!={'decision','findings','reviewer','scope','subject_commit'}
            or design['decision']!='ACCEPTED_GE_BEAM3_G4A_CONTRACT_FOR_PRIVATE_IMPLEMENTATION_ONLY'
            or design['findings'] or design['reviewer'].get('independent') is not True
            or design['subject_commit']!='d494d87c32fb53952e53392ca8d8a959510f70ef'
            or design['scope'].get('contract_sha256')!=G4A_CONTRACT_SHA
            or design['scope'].get('execution_authorized') is not False
            or design['scope'].get('production_qualified') is not False):
            raise ValueError('G4a design review authority')
        inventory('smoke',gate);inventory('core',gate);inventory('rehearsal',gate);inventory('formal',gate)
        rows=inputs();raw=read(review_path);verify_review(raw,review_sha,candidate,rows,gate)
        environment.verify(CAPSULE,CAPSULE_SHA)
        return candidate,rows,raw
    if gate=='g3c-physical':
        if git('rev-parse',PHYSICAL_BASE+'^{tree}')!=PHYSICAL_BASE_TREE:
            raise ValueError('physical successor base tree')
        git('merge-base','--is-ancestor',PHYSICAL_BASE,'HEAD')
        changed=set(filter(None,git('diff','--name-only',PHYSICAL_BASE,'HEAD').splitlines()))
        if changed!=PHYSICAL_IMPLEMENTATION_PATHS|PHYSICAL_PROOF_COMPRESSED_PATHS|PHYSICAL_TIMING_GATE_PATHS:
            raise ValueError('physical successor implementation extent changed')
        for path,digest in ((PHYSICAL_PLAN,PHYSICAL_PLAN_SHA),(PHYSICAL_DESIGN_REVIEW,PHYSICAL_DESIGN_SHA),
                            (PHYSICAL_PARTITION_ADDENDUM,PHYSICAL_PARTITION_ADDENDUM_SHA),
                            (PHYSICAL_PARTITION_REVIEW,PHYSICAL_PARTITION_REVIEW_SHA),
                            (PHYSICAL_FORMAL_ADDENDUM,PHYSICAL_FORMAL_ADDENDUM_SHA),
                            (PHYSICAL_FORMAL_DESIGN_REVIEW,PHYSICAL_FORMAL_DESIGN_REVIEW_SHA),
                            (PHYSICAL_FORMAL_INITIAL_IMPLEMENTATION_REVIEW,
                             PHYSICAL_FORMAL_INITIAL_IMPLEMENTATION_REVIEW_SHA),
                            (PHYSICAL_FORMAL_CORRECTION1_IMPLEMENTATION_REVIEW,
                             PHYSICAL_FORMAL_CORRECTION1_IMPLEMENTATION_REVIEW_SHA),
                            (PHYSICAL_PROOF_COMPRESSED_PLAN,PHYSICAL_PROOF_COMPRESSED_PLAN_SHA),
                            (PHYSICAL_PROOF_COMPRESSED_TOOL,PHYSICAL_PROOF_COMPRESSED_TOOL_SHA),
                            (PHYSICAL_PROOF_COMPRESSED_CHECKER,PHYSICAL_PROOF_COMPRESSED_CHECKER_SHA),
                            (PHYSICAL_PROOF_COMPRESSED_INITIAL_REVIEW,PHYSICAL_PROOF_COMPRESSED_INITIAL_REVIEW_SHA),
                            (PHYSICAL_PROOF_COMPRESSED_MEASUREMENT,PHYSICAL_PROOF_COMPRESSED_MEASUREMENT_SHA),
                            (PHYSICAL_PROOF_COMPRESSED_CHECKER_INITIAL_REVIEW,
                             PHYSICAL_PROOF_COMPRESSED_CHECKER_INITIAL_REVIEW_SHA),
                            (PHYSICAL_PROOF_COMPRESSED_FAILED_CYCLE,
                             PHYSICAL_PROOF_COMPRESSED_FAILED_CYCLE_SHA),
                            (PHYSICAL_PROOF_COMPRESSED_FAILED_REVIEW,
                             PHYSICAL_PROOF_COMPRESSED_FAILED_REVIEW_SHA),
                            (PHYSICAL_PROOF_COMPRESSED_SUCCESSOR_REVIEW,
                             PHYSICAL_PROOF_COMPRESSED_SUCCESSOR_REVIEW_SHA),
                            (PHYSICAL_CORRECTION_ADDENDUM,PHYSICAL_CORRECTION_ADDENDUM_SHA),
                            (PHYSICAL_CORRECTION_REVIEW,PHYSICAL_CORRECTION_REVIEW_SHA),(JOB,JOB_SHA)):
            if sha256(read(ROOT/path).replace(b'\r\n',b'\n')).hexdigest()!=digest:
                raise ValueError('physical frozen authority input')
        for path,digest in PHYSICAL_TIMING_GATE_HASHES.items():
            if sha256(read(ROOT/path).replace(b'\r\n',b'\n')).hexdigest()!=digest:
                raise ValueError('physical timing-gate input changed')
        design=environment.strict(read(ROOT/PHYSICAL_DESIGN_REVIEW).replace(b'\r\n',b'\n'))
        if (set(design)!={'decision','findings','reviewer','scope','subject_commit'}
            or design['decision']!='ACCEPTED_GE_BEAM3_G3C_PHYSICAL_OWNER_CONTRACT_DESIGN_ONLY'
            or design['findings'] or design['reviewer'].get('independent') is not True
            or design['subject_commit']!='d6f3b041b7b6f3ef8c2ccfa42711137799192cac'
            or design['scope'].get('contract_sha256')!=PHYSICAL_PLAN_SHA
            ):
            raise ValueError('physical design review authority')
        partition_review=environment.strict(read(ROOT/PHYSICAL_PARTITION_REVIEW).replace(b'\r\n',b'\n'))
        if (set(partition_review)!={'decision','findings','reviewer','scope','subject_commit'}
            or partition_review['decision']!='ACCEPTED_GE_BEAM3_G3C_PHYSICAL_REHEARSAL_PARTITION_DESIGN_ONLY'
            or partition_review['findings'] or partition_review['reviewer'].get('independent') is not True
            or partition_review['subject_commit']!='bf60da41fc0f5c7e96d9e298ed755150c28a531d'
            or partition_review['scope']!={'addendum_sha256':PHYSICAL_PARTITION_ADDENDUM_SHA,
                'execution_authorized':False,'full_g3c_qualified':False,
                'inventory_sha256':PHYSICAL_PARTITION_INVENTORY_SHA,
                'partition_manifest_sha256':PHYSICAL_PARTITION_MANIFEST_SHA,
                'production_qualified':False}):
                raise ValueError('physical partition design review authority')
        formal_review=environment.strict(read(ROOT/PHYSICAL_FORMAL_DESIGN_REVIEW).replace(b'\r\n',b'\n'))
        if (set(formal_review)!={'decision','findings','reviewer','scope','subject_commit'}
            or formal_review['decision']!='ACCEPTED_GE_BEAM3_G3C_PHYSICAL_FORMAL_EXECUTION_DESIGN_ONLY'
            or formal_review['findings'] or formal_review['reviewer'].get('independent') is not True
            or formal_review['subject_commit']!='509d77310959cb10a9529ed8162d72715b4c2e90'
            or formal_review['scope']!={'design_sha256':PHYSICAL_FORMAL_ADDENDUM_SHA,
                'formal_assignment_inventory_sha256':PHYSICAL_FORMAL_ASSIGNMENT_INVENTORY_SHA,
                'formal_execution_authorized':False,'full_g3c_qualified':False,
                'history_inventory_sha256':PHYSICAL_FORMAL_HISTORY_INVENTORY_SHA,
                'implementation_authorized':True,'nonclassifying_measurement_preparation_authorized':True,
                'production_qualified':False,'subject_tree':'f77b7c42d24186a2defacaa300015bddbd71eed1'}):
            raise ValueError('physical formal design review authority')
        for path,digest in ((PHYSICAL_CORRECTION_REVISION,PHYSICAL_CORRECTION_REVISION_SHA),
                            (PHYSICAL_CORRECTION_REVISION_REVIEW,PHYSICAL_CORRECTION_REVISION_REVIEW_SHA),
                            (PHYSICAL_CORRECTION_RECOVERY,PHYSICAL_CORRECTION_RECOVERY_SHA),
                            (PHYSICAL_CORRECTION_RECOVERY_INITIAL_REVIEW,PHYSICAL_CORRECTION_RECOVERY_INITIAL_REVIEW_SHA),
                            (PHYSICAL_CORRECTION_RECOVERY_SUPERSEDED_REVIEW,PHYSICAL_CORRECTION_RECOVERY_SUPERSEDED_REVIEW_SHA),
                            (PHYSICAL_CORRECTION_RECOVERY_REVIEW,PHYSICAL_CORRECTION_RECOVERY_REVIEW_SHA),
                            (PHYSICAL_CORRECTION_RECOVERY_IMPL_INITIAL_REVIEW,PHYSICAL_CORRECTION_RECOVERY_IMPL_INITIAL_REVIEW_SHA)):
            if sha256(read(ROOT/path).replace(b'\r\n',b'\n')).hexdigest()!=digest:
                raise ValueError('physical correction revision input changed')
        correction_review=environment.strict(read(ROOT/PHYSICAL_CORRECTION_REVIEW).replace(b'\r\n',b'\n'))
        if (set(correction_review)!={'decision','findings','reviewer','scope','subject_commit'}
            or correction_review['decision']!='ACCEPTED_GE_BEAM3_G3C_PHYSICAL_CORRECTION_INHERITANCE_DESIGN_ONLY'
            or correction_review['findings'] or correction_review['reviewer'].get('independent') is not True
            or correction_review['subject_commit']!='f00eb6336d1c6892345f0bf1a10ab8516779b000'
            or correction_review['scope']!={'addendum_sha256':PHYSICAL_CORRECTION_ADDENDUM_SHA,
                'execution_authorized':False,'full_g3c_qualified':False,
                'predecessor_commit':PHYSICAL_PREDECESSOR['commit'],
                'predecessor_runtime_sha256':PHYSICAL_PREDECESSOR_RUNTIME_SHA,
                'production_qualified':False,'subject_tree':'2c6550e276baed9945c86292c2c1067709490415'}):
                raise ValueError('physical correction design review authority')
        revision_review=environment.strict(read(ROOT/PHYSICAL_CORRECTION_REVISION_REVIEW).replace(b'\r\n',b'\n'))
        if (set(revision_review)!={'decision','findings','reviewer','scope','subject_commit'}
            or revision_review['decision']!='ACCEPTED_GE_BEAM3_G3C_PHYSICAL_CORRECTION_INHERITANCE_V2_DESIGN_ONLY'
            or revision_review['findings'] or revision_review['reviewer'].get('independent') is not True
            or revision_review['subject_commit']!='a862426ca446733ef77f9c6b27346f83d11878a2'
            or revision_review['scope']!={'execution_authorized':False,'full_g3c_qualified':False,
                'predecessor_addendum_sha256':PHYSICAL_CORRECTION_ADDENDUM_SHA,
                'production_qualified':False,'revision_sha256':PHYSICAL_CORRECTION_REVISION_SHA,
                'subject_tree':'45d6497608727c155a88d69c38c08cabce426f03'}):
            raise ValueError('physical correction revision review authority')
        recovery_review=environment.strict(read(ROOT/PHYSICAL_CORRECTION_RECOVERY_REVIEW).replace(b'\r\n',b'\n'))
        if (set(recovery_review)!={'decision','findings','reviewer','scope','subject_commit'}
            or recovery_review['decision']!='ACCEPTED_GE_BEAM3_G3C_PHYSICAL_CORRECTION_PARTITION_RECOVERY_DESIGN_ONLY'
            or recovery_review['findings'] or recovery_review['reviewer'].get('independent') is not True
            or recovery_review['subject_commit']!='2f3874f36fd88cecad871605641b6a3fd994153e'
            or recovery_review['scope']!={'design_sha256':PHYSICAL_CORRECTION_RECOVERY_SHA,
                'execution_authorized':False,'full_g3c_qualified':False,'production_qualified':False,
                'segment_manifest_sha256':PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA,
                'subject_tree':'c0213c726e324330970ac3271ffaaa69ddba53bb'}):
            raise ValueError('physical correction recovery review authority')
        physical_partition_manifest();physical_correction_guard_segment_manifest();physical_formal_case_associations()
        if physical_proof_compressed_partition()['self_sha256']!=PHYSICAL_PROOF_COMPRESSED_PARTITION_SHA:
            raise ValueError('proof-compressed partition authority')
        inventory('local',gate);inventory('smoke',gate);inventory('rehearsal',gate);inventory('formal',gate)
        rows=inputs();raw=read(review_path);verify_review(raw,review_sha,candidate,rows,gate)
        environment.verify(CAPSULE,CAPSULE_SHA)
        return candidate,rows,raw
    git('merge-base','--is-ancestor',BASE,'HEAD')
    changed=set(filter(None,git('diff','--name-only',BASE,'HEAD').splitlines()))
    if not changed<=ALLOWED:raise ValueError('production or unregistered extent changed')
    for path,digest in ((CONTRACT,CONTRACT_SHA),(DESIGN_REVIEW,DESIGN_SHA),(JOB,JOB_SHA),(INTEGRATION_REVIEW,INTEGRATION_SHA)):
        if sha256(read(ROOT/path).replace(b'\r\n',b'\n')).hexdigest()!=digest:raise ValueError('frozen authority input')
    contract=environment.strict(read(ROOT/CONTRACT).replace(b'\r\n',b'\n'))
    for key in ('source','proposal'):
        item=contract[key]
        if sha256(read(ROOT/item['path']).replace(b'\r\n',b'\n')).hexdigest()!=item['sha256']:raise ValueError('source equation changed')
    design=environment.strict(read(ROOT/INTEGRATION_REVIEW).replace(b'\r\n',b'\n'))
    for path,key in (('docs/GE_BEAM3_B2_PHYSICAL_ADAPTER_PLAN.md','b2_adapter_plan_sha256'),
                     ('docs/GE_BEAM3_Q4_RECOVERY_COEFFICIENT_AUDIT_PLAN.md','q4_audit_plan_sha256')):
        if sha256(read(ROOT/path).replace(b'\r\n',b'\n')).hexdigest()!=design['scope'][key]:raise ValueError('integration plan changed')
    if gate=='q4-affine-exact':
        for path,digest in ((AFFINE_PLAN,AFFINE_PLAN_SHA),(AFFINE_REVIEW,AFFINE_REVIEW_SHA)):
            if sha256(read(ROOT/path).replace(b'\r\n',b'\n')).hexdigest()!=digest:raise ValueError('affine design authority')
        affine=environment.strict(read(ROOT/AFFINE_REVIEW).replace(b'\r\n',b'\n'))
        if (set(affine)!={'decision','findings','reviewer','scope','subject_commit'}
            or affine['decision']!='ACCEPTED_GE_BEAM3_Q4_AFFINE_EXACT_GATE_DESIGN_ONLY' or affine['findings']
            or affine['reviewer'].get('independent') is not True
            or affine['scope'].get('gate')!='q4-affine-exact' or affine['scope'].get('plan_sha256')!=AFFINE_PLAN_SHA):
            raise ValueError('affine independent design acceptance')
    if gate=='q4-affine-numerical':
        for path,digest in ((INCREMENT_ADDENDUM,INCREMENT_ADDENDUM_SHA),(INCREMENT_REVIEW,INCREMENT_REVIEW_SHA)):
            if sha256(read(ROOT/path).replace(b'\r\n',b'\n')).hexdigest()!=digest:raise ValueError('increment chart authority')
        increment_review=environment.strict(read(ROOT/INCREMENT_REVIEW))
        if (set(increment_review)!={'decision','findings','reviewer','scope','subject_commit'} or increment_review['findings']
            or increment_review['decision']!='ACCEPTED_GE_BEAM3_Q4_AFFINE_INCREMENT_CHART_DESIGN_ONLY'
            or increment_review['reviewer'].get('independent') is not True
            or increment_review['scope'].get('addendum_sha256')!=INCREMENT_ADDENDUM_SHA
            or increment_review['scope'].get('original_contract_sha256')!=NUMERICAL_PLAN_SHA
            or increment_review['scope'].get('execution_authorized') is not False):raise ValueError('increment chart design review')
        for path,digest in {CHART_ADDENDUM:CHART_ADDENDUM_SHA,CHART_REVIEW:CHART_REVIEW_SHA,**CHART_SOURCES}.items():
            if sha256(read(ROOT/path).replace(b'\r\n',b'\n')).hexdigest()!=digest:raise ValueError('stable chart source authority')
        chart_review=environment.strict(read(ROOT/CHART_REVIEW))
        if (set(chart_review)!={'decision','findings','reviewer','scope','subject_commit'} or chart_review['findings']
            or chart_review['decision']!='ACCEPTED_GE_BEAM3_Q4_AFFINE_STABLE_CHART_DESIGN_ONLY'
            or chart_review['reviewer'].get('independent') is not True
            or chart_review['scope'].get('addendum_sha256')!=CHART_ADDENDUM_SHA
            or chart_review['scope'].get('original_contract_sha256')!=NUMERICAL_PLAN_SHA
            or chart_review['scope'].get('execution_authorized') is not False):raise ValueError('stable chart design review')
        for path,digest in ((NUMERICAL_PLAN,NUMERICAL_PLAN_SHA),(NUMERICAL_REVIEW,NUMERICAL_REVIEW_SHA),
            ('docs/GE_BEAM3_Q4_AFFINE_RECOVERY_EXTENSION_LEMMA.md','156d33ae5a621b953b5d04050218618f6416bac1042f94d3d3fc5c305c9f8762'),
            ('docs/reference_cases/ge_beam3_q4_affine_extension_lemma_review_v1.json','e28f184023ce1bba825a89079bcd2f9af99cf7861d701d7d918e2d30492b073e'),
            ('docs/reference_cases/ge_beam3_d49aacd_affine_evidence_review.json','a40277da9d95692e690f51e19dc069a7229455bcf76b84ea11f5f0dbd71dda7f'),
            ('docs/reference_cases/ge_beam3_d49aacd_affine_evidence_manifest.json','f5983d496cdac583c8ce4d13c49bcd882579aa0751864389bcc588d8b6e2e529')):
            if sha256(read(ROOT/path).replace(b'\r\n',b'\n')).hexdigest()!=digest:raise ValueError('numerical prerequisite authority')
        design=environment.strict(read(ROOT/NUMERICAL_REVIEW))
        if (set(design)!={'decision','findings','reviewer','scope','subject_commit'} or design['findings']
            or design['decision']!='ACCEPTED_GE_BEAM3_Q4_AFFINE_NUMERICAL_RECOVERY_DESIGN_ONLY'
            or design['reviewer'].get('independent') is not True
            or design['scope'].get('plan_sha256')!=NUMERICAL_PLAN_SHA
            or design['scope'].get('execution_authorized') is not False):raise ValueError('numerical design review')
        lemma=environment.strict(read(ROOT/'docs/reference_cases/ge_beam3_q4_affine_extension_lemma_review_v1.json'))
        if (lemma['decision']!='ACCEPTED_GE_BEAM3_Q4_AFFINE_EXTENSION_LEMMA' or lemma['findings']
            or lemma['reviewer'].get('independent') is not True
            or lemma['scope'].get('lemma_sha256')!='156d33ae5a621b953b5d04050218618f6416bac1042f94d3d3fc5c305c9f8762'):
            raise ValueError('extension lemma prerequisite review')
        observations=frozen_observations()
        if observation_capture is not None:observation_capture.update(observations)
    inventory('core',gate)
    rows=inputs();raw=read(review_path);verify_review(raw,review_sha,candidate,rows,gate)
    environment.verify(CAPSULE,CAPSULE_SHA)
    return candidate,rows,raw

def monitor(job,process,progress,clock=time.monotonic,sleep=time.sleep,start=None):
    """One terminal state; reserve 15 seconds inside the child limit to drain."""
    start=clock() if start is None else start;last=start;seen=None;peak=0
    while True:
        cpu,active,mem=job.accounting();now=clock();peak=max(peak,mem)
        current=(cpu,*progress())
        if current!=seen:seen=current;last=now
        if now-start>=585 or now-last>=120 or peak>MEMORY:
            drained=job.terminate()
            return dict(status='RESOURCE_BLOCKED',active_processes=job.accounting()[1],
                        peak_tree_bytes=peak,drained=bool(drained),elapsed_seconds=clock()-start)
        code=process.poll()
        if code is not None and active==0:
            return dict(status='PASSED' if code==0 else 'FAILED',active_processes=0,
                        peak_tree_bytes=peak,drained=True,elapsed_seconds=clock()-start)
        sleep(.1)

def validate_lease(lease,expected,review_sha,lane,out):
    candidate,rows,_=expected
    numeric=lease.get('gate')=='q4-affine-numerical'
    if (set(lease)!=({'schema','run_id','gate','lane','candidate','inputs','review_sha256','selected'}|
                    ({'observation_manifest_sha256'} if numeric else set()))
        or lease['schema']!=SCOPE or lease['gate'] not in TESTS or lease['lane']!=lane
        or lease['candidate']!=candidate or lease['inputs']!=rows
        or lease['review_sha256']!=review_sha or lease['selected']!=inventory(lane,lease['gate'])
        or str(uuid.UUID(lease['run_id']))!=lease['run_id']):raise ValueError('lease authority')
    if numeric and lease['observation_manifest_sha256']!=OBSERVATION_MANIFEST_SHA:raise ValueError('lease observation authority')

def claim_attempt(out,run_id):
    write(out/'worker-attempt.json',dict(run_id=run_id))

def q4_adjudication(records,lane):
    if lane=='smoke':return dict(terminal='NOT_ADJUDICATED_SMOKE_ONLY',recovery_qualified=False)
    fixtures=[row for row in records if row.get('test') in Q4_FIXTURES]
    if [row['test'] for row in fixtures]!=list(Q4_FIXTURES):raise ValueError('Q4 fixture inventory')
    witness=None
    for row in fixtures:
        proof=row['proof'];verification=row['verification']
        nonzero=[item for item in proof['coefficient_records'] if any(c!='0' for c in item['coefficient'])]
        first=nonzero[0] if nonzero else None
        if (len(proof['coefficient_records'])!=20150 or proof['coefficient_count']!=20150
            or proof['nonzero_count']!=len(nonzero) or proof['zero_count']!=20150-len(nonzero)
            or proof['first_nonzero']!=first or verification['first_nonzero']!=first
            or verification['nonzero_count']!=len(nonzero) or verification['independently_verified'] is not True
            or row['checker_replicas_byte_identical'] is not True):raise ValueError('Q4 adjudication authority')
        if witness is None and first is not None:witness=dict(fixture_id=row['test'],coefficient=first)
    return dict(terminal=('NO_GO_G3C_Q4_NATURAL_RETAINED_SPACE_FINITE_IDENTITY' if witness else
                         'UNCLASSIFIED_G3C_Q4_TWO_FIXTURE_COEFFICIENT_IDENTITIES'),
                first_nonzero=witness,coefficient_count=40300,recovery_qualified=False,
                universal_impossibility_claim=False)


def affine_adjudication(records,lane):
    if lane=='smoke':return dict(terminal='NOT_ADJUDICATED_SMOKE_ONLY',physical_recovery_qualified=False)
    rows=[row for row in records if row.get('test') in AFFINE_FIXTURES]
    if [row['test'] for row in rows]!=list(AFFINE_FIXTURES):raise ValueError('affine fixture inventory')
    first=None
    for row in rows:
        p,v=row['proof'],row['verification']
        nonzero=[r for r in p['coefficient_records'] if any(x!='0' for x in r['coefficient'])]
        witness=nonzero[0] if nonzero else None
        if (p['schema']!='GE_BEAM3_Q4_AFFINE_RECOVERY_EXACT_FIXTURE_V1'
            or p['fixture_id']!=row['test'] or len(p['coefficient_records'])!=7125 or p['coefficient_count']!=7125
            or p['degree_counts']!={'3':1140,'4':5985}
            or p['nonzero_count']!=len(nonzero) or p['zero_count']!=7125-len(nonzero)
            or p['first_nonzero']!=witness or v!={'fixture_id':row['test'],'coefficient_count':7125,
                'nonzero_count':len(nonzero),'first_nonzero':witness,'independently_verified':True,
                'physical_recovery_qualified':False,'full_g3c_qualified':False}
            or row['checker_replicas_byte_identical'] is not True
            or p['physical_recovery_qualified'] is not False or p['full_g3c_qualified'] is not False):
            raise ValueError('affine adjudication authority')
        if first is None and witness is not None:first=dict(fixture_id=row['test'],coefficient=witness)
    return dict(terminal='NO_GO_G3C_Q4_AFFINE_RECOVERY_VARIATIONAL_IDENTITY' if first else
                'UNCLASSIFIED_G3C_Q4_AFFINE_RECOVERY_EXACT_IDENTITIES_ONLY',first_nonzero=first,
                coefficient_count=21375,physical_recovery_qualified=False,full_g3c_qualified=False)


def g4a_adjudication(records,lane):
    if type(records)is not list or len(records)!=1:raise ValueError('G4a record inventory')
    row=records[0]
    if (type(row)is not dict or set(row)!={'test','station_families','accepted_epochs',
        'independent_replay','atomic_publication','physical_fibre_recovery','snapshot_sha256',
        'production_qualified'} or row['test']!='G4A_HETEROGENEOUS_SIX_STAGE_TRANSACTION'
        or row['station_families']!=['EXACT_ELASTIC','GENERALIZED_ELLIPSOID','PHYSICAL_FIBRE']
        or row['accepted_epochs']!=6 or row['independent_replay'] is not True
        or row['atomic_publication'] is not True or row['physical_fibre_recovery'] is not True
        or row['production_qualified'] is not False):raise ValueError('G4a scientific record')
    sha_value(row['snapshot_sha256'])
    return dict(terminal=('NOT_ADJUDICATED_SMOKE_ONLY' if lane=='smoke' else
        'PROVISIONAL_GO_GE_BEAM3_G4A_STATION_TRANSACTION_ONLY'),g4a_qualified=lane!='smoke',
        g4_qualified=False,production_qualified=False)


def g4_adjudication(records,lane):
    if type(records)is not list or len(records)!=1:raise ValueError('G4 record inventory')
    row=records[0]
    required={'test','accepted_epochs','station_families','mixed_atomic_publication',
        'generalized_connected_history','fibre_connected_history','force_control',
        'displacement_control','arc_control','load_unload_reversal','shell_joint_history',
        'accepted_origin_replay','authenticated_restart','physical_recovery',
        'resultant_only_no_invented_fibres','checkpoint_sha256','production_qualified'}
    booleans=required-{'test','accepted_epochs','station_families','checkpoint_sha256','production_qualified'}
    if (type(row)is not dict or set(row)!=required
        or row['test']!='G4_COMPLETE_MATERIAL_STATE_CONTROL_AND_RESTART'
        or row['accepted_epochs']!=6
        or row['station_families']!=['EXACT_ELASTIC','GENERALIZED_ELLIPSOID','PHYSICAL_FIBRE']
        or any(row[key] is not True for key in booleans)
        or row['production_qualified'] is not False):raise ValueError('G4 scientific record')
    sha_value(row['checkpoint_sha256'])
    return dict(terminal=('NOT_ADJUDICATED_SMOKE_ONLY' if lane=='smoke' else
        'PROVISIONAL_GO_GE_BEAM3_G4_GENERAL_STATIC_MATERIAL_STATE_ONLY'),
        closed_rows=[] if lane=='smoke' else ['S19','S20','S21','S22','S23','S24'],
        g4_qualified=lane!='smoke',g5_qualified=False,production_qualified=False)

def g5_adjudication(records,lane):
    routes=('LEGACY_B2_REFERENCE_ANALYSIS','LEGACY_B3_REFERENCE_ANALYSIS',
        'GE_REFERENCE_LOAD_INERTIA','GE_REFERENCE_MODAL_BUCKLING_TRANSIENT',
        'GE_NATIVE_STATIC_RESTART_RECOVERY','UNSUPPORTED_ROUTES_FAIL_CLOSED',
        'SELECTOR_AND_PROVENANCE_BOUNDARY')
    expected=routes[:1] if lane=='smoke' else routes
    if type(records)is not list or tuple(row.get('route') for row in records)!=expected:
        raise ValueError('G5 ordered route inventory')
    for row in records:
        if (type(row)is not dict or set(row)!={'policy','route','checks','production_qualified','record_sha256'}
            or row['policy']!='GE_BEAM3_G5_GENERAL_STATIC_INTEGRATION_V1'
            or row['production_qualified'] is not False or type(row['checks'])is not dict
            or not row['checks'] or any(value is not True for value in row['checks'].values())):
            raise ValueError('G5 scientific record')
        body={key:value for key,value in row.items() if key!='record_sha256'}
        if row['record_sha256']!=sha256(canonical(body)).hexdigest():raise ValueError('G5 scientific record hash')
    return dict(terminal=('NOT_ADJUDICATED_SMOKE_ONLY' if lane=='smoke' else
        'FUNCTIONAL_GE_BEAM3_G5_READY_FOR_INSTALLED_ARTIFACT'),
        closed_rows=[] if lane=='smoke' else ['S25'],
        functional_routes_qualified=lane!='smoke',installed_artifact_qualified=False,
        g5_qualified=False,production_qualified=False)

def job_type():
    # Importlib and sys.modules mutation are not thread-safe.  Physical waves
    # launch three children concurrently, so serialize only this inexpensive
    # class load rather than the child processes themselves.
    with _JOB_TYPE_LOCK:
        spec=importlib.util.spec_from_file_location('beam_bounded_job',ROOT/JOB)
        m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
        return m._ProcessJob

def worker(out,lease_sha):
    raw=read(out/'lease.json')
    if sha256(raw).hexdigest()!=lease_sha:raise ValueError('lease hash')
    lease=environment.strict(raw)
    if lease['gate']=='q4-affine-numerical':raise ValueError('numerical gate requires one-node assignment')
    print('BEAM CHECKPOINT initialization',flush=True)
    expected=authority(out/'review.json',lease['review_sha256'],lease['gate'])
    validate_lease(lease,expected,lease['review_sha256'],lease['lane'],out)
    claim_attempt(out,lease['run_id'])
    if any(os.environ.get(key)!='1' for key in THREADS):raise ValueError('numerical thread environment')
    print('BEAM CHECKPOINT authority complete',flush=True)
    os.environ['BEAM_QUALIFICATION_OUTPUT']=str(out)
    sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(CAPSULE.parent/'site')]
    import pytest
    class Recorder:
        def __init__(self):self.collected=[];self.passed=[];self.bad=[];self.module=None
        def pytest_collection_modifyitems(self,items):
            self.collected=[i.nodeid for i in items]
            if self.collected!=lease['selected']:raise ValueError('collected inventory differs')
            self.module=items[0].module
        def pytest_runtest_logstart(self,nodeid,location):print('BEAM CHECKPOINT test '+nodeid,flush=True)
        def pytest_runtest_logreport(self,report):
            if report.failed or report.skipped:self.bad.append(report.nodeid)
            if report.when=='call' and report.passed:self.passed.append(report.nodeid)
    recorder=Recorder()
    code=pytest.main(['-vv','-s','-p','no:cacheprovider','--basetemp',str(out/'pytest'),*lease['selected']],plugins=[recorder])
    if code!=0 or recorder.bad or recorder.passed!=lease['selected']:return 1
    records=recorder.module.SCIENTIFIC_RECORDS
    if not records:raise ValueError('missing numerical scientific records')
    if authority(out/'review.json',lease['review_sha256'],lease['gate'])!=expected:raise ValueError('final authority changed')
    scientific=dict(schema='GE_BEAM3_REGISTERED_GATE_SCIENCE_V2',gate=lease['gate'],lane=lease['lane'],
                    candidate=lease['candidate'],selected=lease['selected'],records=records,
                    full_g3c_qualified=False,production_qualified=False)
    if lease['gate']=='q4-audit':scientific['adjudication']=q4_adjudication(records,lease['lane'])
    if lease['gate']=='q4-affine-exact':scientific['adjudication']=affine_adjudication(records,lease['lane'])
    if lease['gate']=='g4a-station':scientific['adjudication']=g4a_adjudication(records,lease['lane'])
    if lease['gate']=='g4-completion':scientific['adjudication']=g4_adjudication(records,lease['lane'])
    if lease['gate']=='g5-completion':scientific['adjudication']=g5_adjudication(records,lease['lane'])
    write(out/'scientific.pending.json',scientific)
    write(out/'completion.json',dict(selected=recorder.passed,scientific=fingerprint(read(out/'scientific.pending.json'))))
    print('BEAM CHECKPOINT evidence complete',flush=True)
    return 0

def physical_assignment_nodes(assignment,correction=False):
    kind=assignment['kind']
    if kind in ('owner-static','owner'):return assignment['nodes']
    if kind in ('history','prefix'):return [PHYSICAL_HISTORY_TEST+'::'+PHYSICAL_HISTORY_NODE]
    test=PHYSICAL_CORRECTION_TEST if correction else PHYSICAL_HISTORY_TEST
    if kind=='preflight':
        if correction:
            return [test+'::'+name for name in ('test_physical_correction_preflight_guards',
                'test_runtime_compatibility_negatives','test_runtime_compatibility_positive')]
        return [test+'::'+PHYSICAL_PREFLIGHT_NODE]
    if kind=='mutation':return [test+'::'+('test_physical_correction_mutation_assignment' if correction else PHYSICAL_MUTATION_NODE)]
    if kind=='authority-mutation':return []
    raise ValueError('physical assignment kind')

def physical_adapter_ids(assignment):
    """Derive registered nonnative element IDs from frozen inert authority."""
    if (type(assignment)is not dict or set(assignment)!={'graph','variant'}
        or assignment['graph'] not in physical_support().GRAPHS
        or assignment['variant'] not in physical_support().VARIANTS):
        raise ValueError('physical adapter assignment')
    _,source=physical_support().packet.authorities()
    expanded=source.expand_inert(assignment['graph'],assignment['variant'])[1]
    ids=[e['id'] for e in expanded['graph']['elements'] if e['family']!='NATIVE']
    if not ids or any(type(value)is not int for value in ids):
        raise ValueError('physical adapter identity')
    return ids

def physical_atomicity_stages(assignment):
    return ['family:'+str(physical_adapter_ids(assignment)[-1]),
            'prepare','native_committed','before_publish']

def physical_observation_stages(assignment):
    ids=physical_adapter_ids(assignment)
    return list(dict.fromkeys(('pose','family:'+str(ids[0]),'family:'+str(ids[-1]),
                               'prepare','before_publish')))

def packet_descriptor(path):
    raw=read(path)
    return dict(path=str(path.resolve()),bytes=len(raw),sha256=sha256(raw).hexdigest())

def validate_packet_descriptor(row):
    if type(row)is not dict or set(row)!={'path','bytes','sha256'}:raise ValueError('physical packet descriptor')
    sha_value(row['sha256'])
    path=Path(row['path'])
    if type(row['bytes'])is not int or row['bytes']<=0 or not path.is_absolute():raise ValueError('physical packet bound')
    raw=read(path)
    if len(raw)!=row['bytes'] or sha256(raw).hexdigest()!=row['sha256']:raise ValueError('physical packet changed')
    return raw

def physical_closed_world(directory,expected_files,expected_directories=()):
    directory=Path(directory);expected_files=set(expected_files);expected_directories=set(expected_directories)
    actual_files=set();actual_directories=set()
    for path in directory.iterdir():
        if path.is_symlink() or (hasattr(path,'is_junction') and path.is_junction()):
            raise ValueError('physical evidence reparse entry')
        if path.is_file():
            environment.regular(path);actual_files.add(path.name)
        elif path.is_dir():actual_directories.add(path.name)
        else:raise ValueError('physical evidence special entry')
    if actual_files!=expected_files or actual_directories!=expected_directories:
        raise ValueError('physical evidence closed world')

def physical_expected_node_files(assignment):
    names={'lease.json','review.json','worker-attempt.json','stdout.log','stderr.log',
           'scientific.node.json','completion.json','process.json'}
    if assignment['kind']=='history':
        names.update('prefix-%02d.json'%index for index in range(assignment['stages']+1))
    return names

def physical_join_input_packets(lease,history_outputs):
    expected=physical_inputs_for(lease['assignment'],history_outputs)
    if not exact_json(lease['input_packets'],expected):
        raise ValueError('physical packet lineage join')

def physical_join_prerequisite_chain(current,predecessor):
    if type(current)is not list or type(predecessor)is not list or len(predecessor)>len(current):
        raise ValueError('physical prerequisite lineage schema')
    if not exact_json(current[:len(predecessor)],predecessor):
        raise ValueError('physical prerequisite lineage join')

def physical_lease_expected(lease,expected,review_sha,runtime_sha=None):
    candidate,rows,review=expected
    lane=lease.get('lane');index=lease.get('assignment_index')
    full=physical_inventory(lane)
    compatibility=lease.get('runtime_compatibility')
    extra=({'runtime_compatibility','guard_segment_id','guard_segment_manifest_sha256',
            'guard_segment_assignments_sha256'} if compatibility is not None else set())
    if (set(lease)!=({'kind','schema','run_id','gate','lane','candidate','inputs','review_sha256',
                    'implementation_review','contract_sha256','assignment_index','assignment',
                    'whole_inventory_sha256','runtime_sha256','input_packets'}|extra)
        or lease['kind']!='G3C_PHYSICAL_PRIVATE_DEVELOPMENT' or lease['schema']!=SCOPE
        or lease['gate']!='g3c-physical' or type(index)is not int or not 0<=index<len(full)
        or not exact_json(lease['candidate'],candidate) or not exact_json(lease['inputs'],rows)
        or lease['review_sha256']!=review_sha
        or canonical(lease['implementation_review'])!=review or lease['contract_sha256']!=PHYSICAL_PLAN_SHA
        or not exact_json(lease['assignment'],full[index])
        or lease['whole_inventory_sha256']!=sha256(canonical(full)).hexdigest()
        or lease['runtime_sha256']!=(physical_support().runtime_identity() if runtime_sha is None else runtime_sha)
        or type(lease['input_packets'])is not dict):raise ValueError('physical assignment lease')
    if compatibility is not None:
        physical_validate_runtime_compatibility_record(compatibility,expected,live=lease['runtime_sha256'])
        _,segment=physical_correction_guard_segment_spec(lease.get('guard_segment_id'),full)
        if (lease.get('guard_segment_manifest_sha256')!=PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA
            or lease.get('guard_segment_assignments_sha256')!=segment['assignment_sha256']
            or index not in segment['indices']):
            raise ValueError('physical correction segment lease')
    uuid.UUID(lease['run_id'])
    kind=lease['assignment']['kind']
    expected_packets=({'prefix','final'} if kind=='prefix' else {'origin'} if kind in
                      ('preflight','mutation','authority-mutation') else set())
    if set(lease['input_packets'])!=expected_packets:raise ValueError('physical packet assignment schema')
    for row in lease['input_packets'].values():validate_packet_descriptor(row)
    return full

def physical_authority_probe(lease):
    probe=lease['assignment']['probe'];review=canonical(lease['implementation_review'])
    original_review_sha=lease['review_sha256'];error=None
    if (probe['category'],probe['member'])==('R09_RUNTIME','changed_implementation_review'):
        changed=environment.strict(review);changed['subject_commit']='0'*40;raw=canonical(changed)
        try:verify_review(raw,original_review_sha,lease['candidate'],lease['inputs'],'g3c-physical')
        except ValueError as exc:error=str(exc)
        if error!='review hash':raise ValueError('changed review mutation not rejected exactly')
        before=sha256(review).hexdigest();after=sha256(raw).hexdigest()
    elif probe['category']=='R10_NORMAL_SOURCE' and probe['member'] in ('mocked_normal_authority_read','mocked_source_hash'):
        rows=json.loads(json.dumps(lease['inputs']))
        target=('docs/reference_cases/ge_beam3_g3c_fixtures_v1.json' if probe['member']=='mocked_normal_authority_read'
                else 'src/anyfesolver/_ge_beam3_g3c_definition.py')
        before=rows[target]['sha256'];rows[target]['sha256']='0'*64 if before!='0'*64 else '1'*64
        try:verify_review(review,original_review_sha,lease['candidate'],rows,'g3c-physical')
        except ValueError as exc:error=str(exc)
        if error!='physical implementation review authority':raise ValueError('source mutation not rejected exactly')
        after=rows[target]['sha256']
    else:raise ValueError('unregistered authority mutation')
    return [dict(kind='authority-mutation',assignment=probe,
                 origin_sha256=lease['input_packets']['origin']['sha256'],
                 before_sha256=before,after_sha256=after,expected_error=error,passed=True)]

def physical_worker(out,lease_sha):
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):raise ValueError('isolated physical worker required')
    raw=read(out/'lease.json')
    if sha256(raw).hexdigest()!=lease_sha:raise ValueError('physical lease hash')
    lease=environment.strict(raw)
    expected=authority(out/'review.json',lease['review_sha256'],'g3c-physical')
    physical_lease_expected(lease,expected,lease['review_sha256'])
    claim_attempt(out,lease['run_id'])
    if any(os.environ.get(key)!='1' for key in THREADS):raise ValueError('physical numerical thread environment')
    print('BEAM CHECKPOINT physical authority complete',flush=True)
    assignment=lease['assignment'];correction='runtime_compatibility' in lease
    nodes=physical_assignment_nodes(assignment,correction)
    if assignment['kind']=='authority-mutation':
        records=physical_authority_probe(lease);passed=[]
    else:
        sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(CAPSULE.parent/'site')]
        from anyfesolver import _ge_beam3_g3c_physical_authority as runtime_authority
        runtime_authority.capture(raw)
        if correction:
            import ge_beam3_g3c_correction_lease_binding as correction_binding
            correction_binding.capture(raw)
        import pytest
        class Context:
            def __init__(self):self.collected=[];self.passed=[];self._RECORDS=[]
            def pytest_collection_modifyitems(self,items):
                self.collected=[i.nodeid for i in items]
                if self.collected!=nodes:raise ValueError('physical collected inventory changed')
                module=items[0].module
                module.ASSIGNMENT=(assignment.get('assignment') if assignment['kind']=='owner' else
                    assignment if assignment['kind'] in ('history','prefix') else
                    assignment.get('probe') if assignment['kind']=='mutation' else None)
                module.OUTPUT_DIRECTORY=str(out)
                module.INPUT_PACKETS=lease['input_packets']
                module.EXPECTED_RUNTIME_SHA256=(PHYSICAL_PREDECESSOR_RUNTIME_SHA if correction else lease['runtime_sha256'])
                if correction:module.RUNTIME_COMPATIBILITY=lease['runtime_compatibility']
                self.module=module
            def pytest_runtest_logreport(self,report):
                if report.when=='call' and report.passed:self.passed.append(report.nodeid)
            def pytest_sessionfinish(self,session,exitstatus):
                self._RECORDS=list(getattr(self.module,'SCIENTIFIC_RECORDS',())) if hasattr(self,'module') else []
        context=Context()
        code=pytest.main(['-vv','-s','-p','no:cacheprovider','--basetemp',str(out/'pytest'),*nodes],plugins=[context])
        if code or context.collected!=nodes or context.passed!=nodes or not context._RECORDS:return 1
        records=context._RECORDS;passed=context.passed
    if authority(out/'review.json',lease['review_sha256'],'g3c-physical')!=expected:raise ValueError('physical final authority changed')
    science=dict(schema='GE_BEAM3_G3C_PHYSICAL_NODE_SCIENCE_V1',candidate=lease['candidate'],lane=lease['lane'],
        assignment_index=lease['assignment_index'],assignment=assignment,records=records,
        full_g3c_qualified=False,production_qualified=False)
    write(out/'scientific.node.json',science)
    write(out/'completion.json',dict(assignment_index=lease['assignment_index'],assignment=assignment,
        selected=nodes,passed=passed,scientific=fingerprint(read(out/'scientific.node.json'))))
    print('BEAM CHECKPOINT physical evidence complete',flush=True)
    return 0

def physical_verify_node(out,lease):
    raw=read(out/'scientific.node.json');science=environment.strict(raw)
    completion=environment.strict(read(out/'completion.json'))
    nodes=physical_assignment_nodes(lease['assignment'],'runtime_compatibility' in lease)
    if (set(science)!={'schema','candidate','lane','assignment_index','assignment','records','full_g3c_qualified','production_qualified'}
        or science['schema']!='GE_BEAM3_G3C_PHYSICAL_NODE_SCIENCE_V1'
        or not exact_json(science['candidate'],lease['candidate']) or science['lane']!=lease['lane']
        or type(science['assignment_index'])is not int
        or science['assignment_index']!=lease['assignment_index']
        or not exact_json(science['assignment'],lease['assignment'])
        or type(science['records'])is not list or not science['records']
        or science['full_g3c_qualified']is not False or science['production_qualified']is not False):
        raise ValueError('physical node science')
    expected_completion=dict(assignment_index=lease['assignment_index'],assignment=lease['assignment'],
        selected=nodes,passed=nodes,scientific=fingerprint(raw))
    if not exact_json(completion,expected_completion):raise ValueError('physical node completion')
    assignment=lease['assignment']
    if assignment['kind']=='history':
        record=science['records'][0]
        if (len(science['records'])!=1 or set(record)!={'kind','case_id','events','directional_errors','packets','passed'}
            or record.get('kind')!='history' or record.get('case_id')!=assignment['case']['case_id']
            or type(record.get('events'))is not int or record.get('events')!=assignment['stages']
            or record.get('passed')is not True
            or type(record.get('directional_errors'))is not list or len(record['directional_errors'])!=3
            or any(type(value)is not float or not math.isfinite(value) or value<0 or value>1e-7
                   for value in record['directional_errors'])
            or type(record.get('packets'))is not list or len(record['packets'])!=assignment['stages']+1):
            raise ValueError('physical history record')
        for index,row in enumerate(record['packets']):
            if (set(row)!={'name','bytes','sha256','prefix'} or type(row['bytes'])is not int
                or type(row['prefix'])is not int or row['name']!=f'prefix-{index:02d}.json'
                or row['prefix']!=index or Path(row['name']).name!=row['name']):raise ValueError('physical prefix manifest')
            sha_value(row['sha256'])
            actual=packet_descriptor(out/row['name'])
            if {k:actual[k] for k in ('bytes','sha256')}!={k:row[k] for k in ('bytes','sha256')}:
                raise ValueError('physical prefix bytes')
    elif assignment['kind']=='prefix':
        record=science['records'][0];packets=lease['input_packets']
        if (len(science['records'])!=1
            or set(record)!={'kind','case_id','prefix','input_sha256','final_sha256','passed'}
            or record['kind']!='prefix' or record['case_id']!=assignment['case']['case_id']
            or type(record['prefix'])is not int or record['prefix']!=assignment['prefix']
            or record['input_sha256']!=packets['prefix']['sha256']
            or record['final_sha256']!=packets['final']['sha256'] or record['passed']is not True):
            raise ValueError('physical prefix record')
    elif assignment['kind']=='preflight':
        records=science['records'];origin=lease['input_packets']['origin']['sha256']
        if 'runtime_compatibility' not in lease:
            record=records[0]
            if (len(records)!=1 or set(record)!={'kind','input_sha256','rejections','passed'}
                or record['kind']!='successor_schema_negatives' or record['input_sha256']!=origin
                or type(record['rejections'])is not int or record['rejections']!=4
                or record['passed']is not True):raise ValueError('physical preflight record')
        else:
            if type(records)is not list or len(records)!=3:raise ValueError('correction preflight record count')
            schema,negative,positive=records
            if (set(schema)!={'kind','input_sha256','rejections','passed'}
                or schema.get('kind')!='successor_schema_negatives' or schema.get('input_sha256')!=origin
                or type(schema.get('rejections'))is not int or schema.get('rejections')!=4
                or schema.get('passed')is not True
                or set(negative)!={'kind','rejections','passed'}
                or negative.get('kind')!='runtime_compatibility_negatives'
                or type(negative.get('rejections'))is not int
                or negative.get('rejections')!=PHYSICAL_CORRECTION_COMPATIBILITY_NEGATIVE_COUNT
                or negative.get('passed')is not True
                or set(positive)!={'kind','input_sha256','predecessor_runtime_sha256',
                                  'successor_runtime_sha256','replay_sha256','passed'}
                or positive.get('kind')!='runtime_compatibility_positive'
                or positive.get('input_sha256')!=origin
                or positive.get('predecessor_runtime_sha256')!=PHYSICAL_PREDECESSOR_RUNTIME_SHA
                or positive.get('successor_runtime_sha256')!=lease['runtime_sha256']
                or positive.get('passed')is not True):
                raise ValueError('correction preflight records')
            sha_value(positive['replay_sha256'])
    elif assignment['kind']=='mutation':
        record=science['records'][0];receipt=record.get('receipt');probe=assignment['probe']
        if (len(science['records'])!=1
            or set(record)!={'kind','assignment','input_sha256','receipt','passed'}
            or record['kind']!='mutation' or not exact_json(record['assignment'],probe)
            or record['input_sha256']!=lease['input_packets']['origin']['sha256']
            or record['passed']is not True or type(receipt)is not dict
            or set(receipt)!={'category','member','rejection','expected_error','before_sha256','after_sha256'}
            or receipt['category']!=probe['category'] or receipt['member']!=probe['member']
            or receipt['before_sha256']!=record['input_sha256']
            or receipt['after_sha256']==receipt['before_sha256']
            or type(receipt['rejection'])is not str or not receipt['rejection']
            or type(receipt['expected_error'])is not str or not receipt['expected_error']):
            raise ValueError('physical mutation record')
        sha_value(receipt['after_sha256'])
    elif assignment['kind']=='authority-mutation':
        record=science['records'][0];probe=assignment['probe']
        if (len(science['records'])!=1
            or set(record)!={'kind','assignment','origin_sha256','before_sha256','after_sha256','expected_error','passed'}
            or record['kind']!='authority-mutation' or not exact_json(record['assignment'],probe)
            or record['origin_sha256']!=lease['input_packets']['origin']['sha256']
            or record['before_sha256']==record['after_sha256']
            or type(record['expected_error'])is not str or not record['expected_error']
            or record['passed']is not True):
            raise ValueError('physical authority-mutation record')
        sha_value(record['before_sha256']);sha_value(record['after_sha256'])
    elif assignment['kind'] in ('owner-static','owner'):
        expected_nodes=[node.rsplit('::',1)[-1] for node in nodes]
        names=[];expected_assignment=assignment.get('assignment') if assignment['kind']=='owner' else None
        common={'test','assignment','production_qualified','full_g3c_qualified'}
        for record in science['records']:
            if (type(record)is not dict or record.get('production_qualified')is not False
                or record.get('full_g3c_qualified')is not False
                or not exact_json(record.get('assignment'),expected_assignment)
                or type(record.get('test'))is not str):
                raise ValueError('physical owner record')
            names.append('test_physical_'+record['test'])
        aliases={'test_physical_inventory':'test_physical_inventory_and_obligation_map',
                 'test_physical_inert_schema':'test_physical_inert_schema_guards',
                 'test_physical_family_work':'test_physical_family_work',
                 'test_physical_directional':'test_physical_directional',
                 'test_physical_atomicity':'test_physical_atomicity',
                 'test_physical_tokens':'test_physical_token_and_definition_guards',
                 'test_physical_independent_joint_work_transport':'test_physical_independent_joint_work_transport',
                 'test_physical_observation_cache':'test_physical_observation_and_cache_guards',
                 'test_physical_recovery_witness':'test_physical_recovery_witness_guards'}
        if [aliases.get(name,name) for name in names]!=expected_nodes:
            raise ValueError('physical owner record inventory')
        by_name={row['test']:row for row in science['records']}
        def exact(name,fields):
            if set(by_name[name])!=common|set(fields):raise ValueError('physical owner record schema: '+name)
            return by_name[name]
        if 'inventory' in by_name:
            row=exact('inventory',('histories','events','prefixes','obligations'))
            obligations=row['obligations']
            if (any(type(row[key])is not int for key in ('histories','events','prefixes'))
                or (row['histories'],row['events'],row['prefixes'])!=(375,3075,3450)
                or type(obligations)is not dict):
                raise ValueError('physical inventory record')
            if obligations!=PHYSICAL_OBLIGATIONS:
                raise ValueError('physical obligation record')
        if 'inert_schema' in by_name:
            value=exact('inert_schema',('rejections',))['rejections']
            if type(value)is not int or value!=6:raise ValueError('physical inert record')
        if 'family_work' in by_name:
            row=exact('family_work',('accepted_state_sha256','adapter_count','native_load_witnesses',
                                     'native_internal_residual_nonzero','native_load_work_nonzero'))
            sha_value(row['accepted_state_sha256']);graph=expected_assignment['graph']
            if (type(row['adapter_count'])is not int or type(row['native_load_witnesses'])is not int
                or row['adapter_count']!=(4 if graph=='J_MULTIFAMILY_LOOP' else 1)
                or row['native_load_witnesses']!=2 or row['native_internal_residual_nonzero']is not True
                or row['native_load_work_nonzero']is not True):raise ValueError('physical family-work record')
        if 'directional' in by_name:
            errors=exact('directional',('errors',))['errors']
            if (type(errors)is not list or len(errors)!=3 or any(type(v)is not float or not math.isfinite(v) or v<0 or v>1e-7 for v in errors)):
                raise ValueError('physical directional record')
        if 'atomicity' in by_name:
            if exact('atomicity',('stages',))['stages']!=physical_atomicity_stages(expected_assignment):
                raise ValueError('physical atomicity record')
        if 'tokens' in by_name:
            value=exact('tokens',('rejections',))['rejections']
            if type(value)is not int or value!=4:raise ValueError('physical token record')
        if 'independent_joint_work_transport' in by_name:
            row=exact('independent_joint_work_transport',('common_motions','errors'));errors=row['errors']
            if (type(row['common_motions'])is not int or row['common_motions']!=4
                or type(errors)is not list or len(errors)!=8
                or any(type(v)is not float or not math.isfinite(v) or v<0 or v>1e-11 for v in errors)):
                raise ValueError('physical joint-work record')
        if 'observation_cache' in by_name:
            row=exact('observation_cache',('caller_copy','rejections'));graph=expected_assignment['graph']
            stages=physical_observation_stages(expected_assignment)
            probes=[stage+':'+kind for stage in stages for kind in
                    ('foreign_nonce','cleared_nonce','replace_lock','release_lock','dispatch')]
            probes+=['coherent_definition_swap','concurrent_capture']
            if graph in ('J_Q4_PAIR','J_MULTIFAMILY_LOOP'):
                probes+=['Q4:'+kind for kind in ('tuple','entry','descriptor','registry','operator')]
            if row['caller_copy']is not True or row['rejections']!=probes:raise ValueError('physical observation record')
        if 'recovery_witness' in by_name:
            expected_mutations=4 if expected_assignment['graph'] in ('J_Q4_PAIR','J_MULTIFAMILY_LOOP') else 0
            value=exact('recovery_witness',('mutations',))['mutations']
            if type(value)is not int or value!=expected_mutations:
                raise ValueError('physical recovery-witness record')
    else:raise ValueError('physical record assignment kind')
    return science

def atomic_canonical(path,value,watchdog):
    """Validate same-directory staging before one exclusive canonical promotion."""
    if path.exists():raise ValueError('canonical evidence already exists')
    pending=path.with_name(path.stem+'.pending'+path.suffix)
    write(pending,value);watchdog.check()
    raw=read(pending);parsed=environment.strict(raw)
    if canonical(parsed)!=raw:raise ValueError('noncanonical staged evidence')
    # os.replace would silently overwrite a destination created after the
    # exists check.  Same-volume hard-link creation is exclusive on Windows;
    # removing the staging name afterward leaves one complete immutable name.
    watchdog.check();os.link(pending,path);pending.unlink()


def promote_physical_correction_bundle(stage,root,watchdog,final_authority):
    """Promote a complete validated correction bundle only after final authority."""
    names=('process.json','scientific.json','receipt.json')
    if (stage.parent!=root or any(not (stage/name).is_file() for name in names)
        or any((root/name).exists() for name in names)):
        raise ValueError('correction staged bundle authority')
    watchdog.check();final_authority();watchdog.check()
    for name in names:os.link(stage/name,root/name)
    for name in names:(stage/name).unlink()
    stage.rmdir()

def physical_verify_process(out,lease):
    physical_closed_world(out,physical_expected_node_files(lease['assignment']))
    value=environment.strict(read(out/'process.json'))
    expected_keys={'status','active_processes','peak_tree_bytes','drained','elapsed_seconds','kind',
                   'assignment_index','assignment_sha256','returncode','files'}
    if (set(value)!=expected_keys or value['status']!='PASSED'
        or type(value['active_processes'])is not int or value['active_processes']!=0
        or value['drained']is not True or type(value['peak_tree_bytes'])is not int
        or not 0<=value['peak_tree_bytes']<=MEMORY or type(value['elapsed_seconds'])is not float
        or not math.isfinite(value['elapsed_seconds']) or not 0<=value['elapsed_seconds']<600
        or value['kind']!='GE_BEAM3_G3C_PHYSICAL_CHILD_PROCESS_V1'
        or type(value['assignment_index'])is not int or value['assignment_index']!=lease['assignment_index']
        or value['assignment_sha256']!=sha256(canonical(lease['assignment'])).hexdigest()
        or type(value['returncode'])is not int or value['returncode']!=0 or type(value['files'])is not dict):
        raise ValueError('physical accepted process record')
    actual={name:fingerprint(read(out/name)) for name in sorted(physical_expected_node_files(lease['assignment'])-{'process.json'})}
    if not exact_json(value['files'],actual):raise ValueError('physical process file DAG')
    return value

def physical_make_receipt(root,base,inventory_rows,wave_process,science_raw):
    expected=(base['candidate'],base['inputs'],canonical(base['implementation_review']))
    if environment.strict(read(root/'process.json'))!=wave_process:raise ValueError('physical wave process changed')
    nodes=[]
    for index in range(len(inventory_rows)):
        out=root/f'node-{index:04d}';lease=environment.strict(read(out/'lease.json'))
        physical_lease_expected(lease,expected,base['review_sha256'])
        physical_verify_node(out,lease);physical_verify_process(out,lease)
        nodes.append(dict(assignment_index=index,
            lease=fingerprint(read(out/'lease.json')),review=fingerprint(read(out/'review.json')),
            completion=fingerprint(read(out/'completion.json')),process=fingerprint(read(out/'process.json')),
            science=fingerprint(read(out/'scientific.node.json'))))
    body=dict(schema='GE_BEAM3_G3C_PHYSICAL_EVIDENCE_RECEIPT_V1',candidate=base['candidate'],lane=base['lane'],
        inventory_sha256=sha256(canonical(inventory_rows)).hexdigest(),
        implementation_review_sha256=base['review_sha256'],inputs_sha256=sha256(canonical(base['inputs'])).hexdigest(),
        wave_process=fingerprint(read(root/'process.json')),science=fingerprint(science_raw),nodes=nodes,passed=True)
    return dict(body,self_sha256=sha256(canonical(body)).hexdigest())

def physical_child(out,lease,review,watchdog,deadline):
    out.mkdir()
    with (out/'review.json').open('xb')as stream:stream.write(review)
    write(out/'lease.json',lease);lease_hash=sha256(read(out/'lease.json')).hexdigest()
    job=job_type()(MEMORY);process=None;record=dict(status='FAILED');watchdog.attach(job)
    try:
        env=dict(os.environ,**{key:'1' for key in THREADS})
        env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTEST_ADDOPTS='',PYTHONDONTWRITEBYTECODE='1')
        with (out/'stdout.log').open('xb')as stdout,(out/'stderr.log').open('xb')as stderr:
            start=time.monotonic()
            process=job.launch([sys.executable,'-I','-S','-B','-u',str(Path(__file__).resolve()),
                '--physical-worker',str(out),lease_hash],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
            record=monitor(job,process,lambda:((out/'stdout.log').stat().st_size,(out/'stderr.log').stat().st_size),start=start)
        if time.monotonic()>=deadline:record['status']='RESOURCE_BLOCKED'
        if record['status']=='PASSED':physical_verify_node(out,lease)
    except BaseException as exc:
        record.update(status='FAILED_EVIDENCE',exception=type(exc).__name__)
    finally:
        if job.accounting()[1]:job.terminate()
        record['active_processes']=job.accounting()[1]
        if record['active_processes']:record['status']='FAILED_TO_DRAIN'
        job.close();watchdog.detach(job)
        record.update(kind='GE_BEAM3_G3C_PHYSICAL_CHILD_PROCESS_V1',assignment_index=lease['assignment_index'],
            assignment_sha256=sha256(canonical(lease['assignment'])).hexdigest(),returncode=None if process is None else process.poll(),
            files={p.name:fingerprint(p.read_bytes()) for p in out.iterdir() if p.is_file()})
        write(out/'process.json',record)
    return record

def physical_history_packets(directory,assignment):
    science=environment.strict(read(directory/'scientific.node.json'));record=science['records'][0]
    return {row['prefix']:packet_descriptor(directory/row['name']) for row in record['packets']}

def physical_inputs_for(assignment,history_outputs):
    kind=assignment['kind']
    if kind in ('owner-static','owner','history'):return {}
    origin=(assignment['origin'] if kind=='preflight' else assignment['probe']['origin'] if kind in ('mutation','authority-mutation')
            else dict(case_id=assignment['case']['case_id'],prefix=assignment['prefix']))
    if origin['case_id'] not in history_outputs or origin['prefix'] not in history_outputs[origin['case_id']]:
        raise ValueError('physical producer packet unavailable')
    row=history_outputs[origin['case_id']][origin['prefix']]
    if kind=='prefix':
        final=history_outputs[origin['case_id']][assignment['case']['accepted_stages']]
        return dict(prefix=row,final=final)
    return dict(origin=row)

def physical_run_phase(root,indexes,base,review,watchdog,deadline,history_outputs,results):
    remaining=iter(indexes);active={};failed=False
    with ThreadPoolExecutor(max_workers=3)as pool:
        def launch():
            nonlocal failed
            index=next(remaining,None)
            if index is None:return False
            try:packets=physical_inputs_for(base['inventory'][index],history_outputs)
            except BaseException as exc:
                failed=True
                results[index]=dict(status='FAILED_DEPENDENCY',exception=type(exc).__name__,active_processes=0)
                return False
            lease=dict(kind='G3C_PHYSICAL_PRIVATE_DEVELOPMENT',schema=SCOPE,run_id=str(uuid.uuid4()),
                gate='g3c-physical',lane=base['lane'],candidate=base['candidate'],inputs=base['inputs'],
                review_sha256=base['review_sha256'],implementation_review=base['implementation_review'],
                contract_sha256=PHYSICAL_PLAN_SHA,assignment_index=index,assignment=base['inventory'][index],
                whole_inventory_sha256=sha256(canonical(base['inventory'])).hexdigest(),
                runtime_sha256=base['runtime_sha256'],input_packets=packets)
            if base.get('runtime_compatibility') is not None:
                lease['runtime_compatibility']=base['runtime_compatibility']
                lease['guard_segment_id']=base['guard_segment_id']
                lease['guard_segment_manifest_sha256']=base['guard_segment_manifest_sha256']
                lease['guard_segment_assignments_sha256']=base['guard_segment_assignments_sha256']
            active[pool.submit(physical_child,root/f'node-{index:04d}',lease,review,watchdog,deadline)]=(index,lease)
            return True
        for _ in range(min(3,len(indexes))):launch()
        while active:
            done,_=wait(active,timeout=.25,return_when=FIRST_COMPLETED)
            if time.monotonic()>=deadline:failed=True
            for future in done:
                index,lease=active.pop(future)
                try:record=future.result()
                except BaseException as exc:record=dict(status='FAILED',exception=type(exc).__name__)
                results[index]=record
                if record.get('status')!='PASSED':failed=True
                else:
                    assignment=base['inventory'][index]
                    if assignment['kind']=='history':
                        history_outputs[assignment['case']['case_id']]=physical_history_packets(root/f'node-{index:04d}',assignment)
                print('BEAM PHYSICAL NODE',index,record.get('status'),flush=True)
            if not failed:
                while len(active)<3 and launch():pass
        for index in remaining:results[index]=dict(status='NOT_LAUNCHED')
    return not failed

def verify_physical_priors(paths,expected,lane,runtime_sha=None):
    # The frozen program starts with family smoke.  Complete local composition
    # follows that smoke, and rehearsal requires both accepted predecessors.
    required={'smoke':(), 'local':('smoke',), 'rehearsal':('smoke','local'),
              'formal':('smoke','local','rehearsal')}[lane]
    if len(paths)!=len(required):raise ValueError('exact physical prerequisite count')
    candidate,inputs,review_raw=expected;review_sha=sha256(review_raw).hexdigest()
    for path,expected_lane in zip(paths,required):
        path=Path(path);raw=read(path);value=environment.strict(raw)
        body={k:v for k,v in value.items()if k!='self_sha256'}
        inventory_rows=physical_inventory(expected_lane);records=value.get('records')
        if (set(value)!={'schema','candidate','lane','inventory_sha256','records','passed','terminal',
                        'full_g3c_qualified','production_qualified','self_sha256'}
            or sha256(canonical(body)).hexdigest()!=value.get('self_sha256','') or value.get('schema')!='GE_BEAM3_G3C_PHYSICAL_AGGREGATE_V1'
            or not exact_json(value.get('candidate'),candidate) or value.get('lane')!=expected_lane
            or value.get('passed')is not True
            or value.get('inventory_sha256')!=sha256(canonical(inventory_rows)).hexdigest()
            or value.get('terminal')!='COMPLETE_GE_BEAM3_G3C_PHYSICAL_'+expected_lane.upper()+'_ONLY'
            or type(records)is not list or len(records)!=len(inventory_rows)
            or value.get('full_g3c_qualified')is not False
            or value.get('production_qualified')is not False):
            raise ValueError('physical prerequisite aggregate')
        for index,row in enumerate(records):
            science=row.get('science') if type(row)is dict else None
            if (set(row)!={'assignment_index','science_sha256','science'}
                or type(row['assignment_index'])is not int or row['assignment_index']!=index
                or type(science)is not dict or type(row['science_sha256'])is not str
                or row['science_sha256']!=sha256(canonical(science)).hexdigest()
                or science.get('schema')!='GE_BEAM3_G3C_PHYSICAL_NODE_SCIENCE_V1'
                or not exact_json(science.get('candidate'),candidate) or science.get('lane')!=expected_lane
                or type(science.get('assignment_index'))is not int or science.get('assignment_index')!=index
                or not exact_json(science.get('assignment'),inventory_rows[index])
                or science.get('full_g3c_qualified')is not False
                or science.get('production_qualified')is not False):
                raise ValueError('physical prerequisite record')
        receipt_raw=read(path.parent/'receipt.json');receipt=environment.strict(receipt_raw)
        receipt_body={k:v for k,v in receipt.items()if k!='self_sha256'}
        if (set(receipt)!={'schema','candidate','lane','inventory_sha256','implementation_review_sha256',
                          'inputs_sha256','wave_process','science','nodes','passed','self_sha256'}
            or receipt['schema']!='GE_BEAM3_G3C_PHYSICAL_EVIDENCE_RECEIPT_V1'
            or not exact_json(receipt['candidate'],candidate) or receipt['lane']!=expected_lane
            or receipt['inventory_sha256']!=sha256(canonical(inventory_rows)).hexdigest()
            or receipt['implementation_review_sha256']!=review_sha
            or receipt['inputs_sha256']!=sha256(canonical(inputs)).hexdigest()
            or not exact_fingerprint(receipt['science'],raw) or receipt['passed']is not True
            or type(receipt['nodes'])is not list or len(receipt['nodes'])!=len(inventory_rows)
            or sha256(canonical(receipt_body)).hexdigest()!=receipt['self_sha256']):
            raise ValueError('physical prerequisite receipt')
        physical_closed_world(path.parent,{'process.json','receipt.json',path.name},
            {'node-%04d'%index for index in range(len(inventory_rows))})
        wave_raw=read(path.parent/'process.json')
        if not exact_fingerprint(receipt['wave_process'],wave_raw):raise ValueError('physical prerequisite wave process')
        wave=environment.strict(wave_raw)
        if (set(wave)!={'schema','lane','passed','required_nodes','terminal_nodes','elapsed_seconds',
                       'active_processes','results'}
            or wave.get('schema')!='GE_BEAM3_G3C_PHYSICAL_WAVE_PROCESS_V1' or wave.get('lane')!=expected_lane
            or wave.get('passed')is not True or type(wave.get('required_nodes'))is not int
            or type(wave.get('terminal_nodes'))is not int or type(wave.get('active_processes'))is not int
            or type(wave.get('elapsed_seconds'))is not float or not math.isfinite(wave['elapsed_seconds'])
            or wave['elapsed_seconds']<0 or type(wave.get('results'))is not dict
            or set(wave['results'])!={str(i) for i in range(len(inventory_rows))}
            or wave.get('required_nodes')!=len(inventory_rows)
            or wave.get('terminal_nodes')!=len(inventory_rows) or wave.get('active_processes')!=0):
            raise ValueError('physical prerequisite wave completion')
        for index,node in enumerate(receipt['nodes']):
            out=path.parent/f'node-{index:04d}'
            if (set(node)!={'assignment_index','lease','review','completion','process','science'}
                or type(node['assignment_index'])is not int or node['assignment_index']!=index):
                raise ValueError('physical prerequisite node receipt')
            for key,name in (('lease','lease.json'),('review','review.json'),('completion','completion.json'),
                             ('process','process.json'),('science','scientific.node.json')):
                if not exact_fingerprint(node[key],read(out/name)):raise ValueError('physical prerequisite node hash')
            lease=environment.strict(read(out/'lease.json'))
            if read(out/'review.json')!=review_raw:raise ValueError('physical prerequisite review bytes')
            physical_lease_expected(lease,expected,review_sha,runtime_sha)
            physical_verify_node(out,lease);physical_verify_process(out,lease)
            if not exact_json(wave['results'][str(index)],environment.strict(read(out/'process.json'))):
                raise ValueError('physical prerequisite wave/node process DAG')
            if records[index]['science_sha256']!=node['science']['sha256']:
                raise ValueError('physical prerequisite aggregate DAG')

def physical_evidence_descriptor(path,kind,identity):
    path=Path(path).resolve();science=read(path)
    if path.name!='scientific.json':raise ValueError('physical canonical evidence filename')
    return dict(kind=kind,identity=identity,path=str(path),science=fingerprint(science),
        receipt=fingerprint(read(path.parent/'receipt.json')),
        process=fingerprint(read(path.parent/'process.json')))

def physical_verify_evidence_descriptor(value):
    if (type(value)is not dict or set(value)!={'kind','identity','path','science','receipt','process'}
        or value['kind'] not in ('base','partition') or type(value['identity'])is not str
        or type(value['path'])is not str):
        raise ValueError('physical evidence descriptor schema')
    path=Path(value['path'])
    if not path.is_absolute() or str(path.resolve())!=value['path'] or path.name!='scientific.json':
        raise ValueError('physical evidence descriptor path')
    for key,name in (('science',path.name),('receipt','receipt.json'),('process','process.json')):
        target=path if key=='science' else path.parent/name
        if not exact_fingerprint(value[key],read(target)):
            raise ValueError('physical evidence descriptor hash')
    return path


def physical_correction_predecessor(paths):
    """Validate the exact accepted predecessor DAG under its own runtime."""
    identities=('smoke','local','R-HISTORY','R-PREFIX-A','R-PREFIX-B','R-PREFIX-C','R-PREFIX-D')
    if len(paths)!=len(identities):raise ValueError('correction exact inherited prerequisite count')
    paths=[Path(path).resolve() for path in paths]
    expected_directories={
        'smoke':'smoke-f6a6251-accepted-f_ecsqn1','local':'local-f6a6251-accepted-qn104v4f',
        'R-HISTORY':'rehearsal-r-history-f6a6251-accepted-sygrbvcn',
        'R-PREFIX-A':'rehearsal-r-prefix-a-f6a6251-accepted-gu_ml0w6',
        'R-PREFIX-B':'rehearsal-r-prefix-b-f6a6251-accepted-6to03o_7',
        'R-PREFIX-C':'rehearsal-r-prefix-c-f6a6251-accepted-vhrytg6d',
        'R-PREFIX-D':'rehearsal-r-prefix-d-f6a6251-accepted-o79yaeb1'}
    for identity,path in zip(identities,paths):
        if path.name!='scientific.json' or path.parent.name!=expected_directories[identity]:
            raise ValueError('correction inherited evidence path')
        for name,count,digest_value in PHYSICAL_INHERITED_FINGERPRINTS[identity]:
            raw=read(path.parent/name)
            if len(raw)!=count or sha256(raw).hexdigest()!=digest_value:
                raise ValueError('correction inherited evidence fingerprint')
    first=environment.strict(read(paths[0].parent/'node-0000'/'lease.json'))
    review_raw=read(paths[0].parent/'node-0000'/'review.json')
    predecessor=(first['candidate'],first['inputs'],review_raw)
    if (not exact_json(first['candidate'],PHYSICAL_PREDECESSOR)
        or first.get('runtime_sha256')!=PHYSICAL_PREDECESSOR_RUNTIME_SHA
        or sha256(review_raw).hexdigest()!=PHYSICAL_PREDECESSOR_REVIEW_SHA):
        raise ValueError('correction predecessor authority')
    verify_review(review_raw,PHYSICAL_PREDECESSOR_REVIEW_SHA,first['candidate'],first['inputs'],'g3c-physical')
    verify_physical_priors(paths[:2],predecessor,'rehearsal',PHYSICAL_PREDECESSOR_RUNTIME_SHA)
    values=[]
    for identity,path in zip(identities[2:],paths[2:]):
        values.append(physical_validate_partition(path,predecessor,identity,runtime_sha=PHYSICAL_PREDECESSOR_RUNTIME_SHA)[0])
    return predecessor,values


def physical_failed_guard_layout(results):
    """Exact launched/terminal layout of the immutable failed guard incident."""
    if type(results)is not dict or set(results)!={str(i) for i in range(90,234)}:
        raise ValueError('correction failed incident result inventory')
    launched=[]
    for index in range(90,234):
        row=results[str(index)]
        expected=('FAILED' if index==157 else 'NOT_LAUNCHED' if index>=159 else 'PASSED')
        if type(row)is not dict or row.get('status')!=expected:
            raise ValueError('correction failed incident terminal layout')
        if expected=='NOT_LAUNCHED':
            if row!={'status':'NOT_LAUNCHED'}:raise ValueError('correction failed incident unlaunched schema')
        else:launched.append(index)
    return launched


def physical_failed_guards(path,predecessor):
    path=Path(path).resolve()
    if path.name!='process.json' or path.parent.name!='rehearsal-r-guards-f6a6251-blocked-jphgio72':
        raise ValueError('correction failed incident path')
    raw=read(path)
    if not exact_fingerprint(PHYSICAL_FAILED_GUARDS_FINGERPRINT,raw):raise ValueError('correction failed incident hash')
    value=environment.strict(raw);results=value.get('results')
    if (value.get('schema')!='GE_BEAM3_G3C_PHYSICAL_PARTITION_PROCESS_V1'
        or value.get('lane')!='rehearsal' or value.get('partition_id')!='R-GUARDS'
        or value.get('partition_manifest_sha256')!=PHYSICAL_PARTITION_MANIFEST_SHA
        or value.get('passed')is not False
        or type(value.get('required_nodes'))is not int or value.get('required_nodes')!=144
        or type(value.get('terminal_nodes'))is not int or value.get('terminal_nodes')!=144
        or type(value.get('active_processes'))is not int or value.get('active_processes')!=0
        or type(results)is not dict or set(results)!={str(i) for i in range(90,234)}):
        raise ValueError('correction failed incident process')
    launched=physical_failed_guard_layout(results)
    physical_closed_world(path.parent,{'process.json'},{'node-%04d'%i for i in launched})
    for index in launched:
        node=path.parent/f'node-{index:04d}';row=results[str(index)]
        if type(row.get('files'))is not dict:raise ValueError('correction failed incident files schema')
        expected_files=set(row['files'])|{'process.json'}
        physical_closed_world(node,expected_files)
        for name,frozen in row['files'].items():
            if not exact_fingerprint(frozen,read(node/name)):
                raise ValueError('correction failed incident node file')
        if not exact_json(environment.strict(read(node/'process.json')),row):
            raise ValueError('correction failed incident node process')
    failed=[int(key) for key,row in results.items() if row.get('status')=='FAILED']
    if failed!=[157]:raise ValueError('correction failed incident node set')
    node=path.parent/'node-0157';physical_closed_world(node,
        {'lease.json','review.json','worker-attempt.json','stdout.log','stderr.log','process.json'})
    lease=environment.strict(read(node/'lease.json'));process=environment.strict(read(node/'process.json'))
    expected_assignment=physical_inventory('rehearsal')[157]
    if (not exact_json(lease.get('candidate'),PHYSICAL_PREDECESSOR)
        or lease.get('runtime_sha256')!=PHYSICAL_PREDECESSOR_RUNTIME_SHA
        or not exact_json(lease.get('assignment'),expected_assignment)
        or expected_assignment.get('probe',{}).get('category')!='R23_ADAPTER_DIAGNOSTIC'
        or expected_assignment.get('probe',{}).get('member')!='candidate_sha'
        or process.get('status')!='FAILED' or process.get('returncode')!=1
        or process.get('active_processes')!=0 or process.get('drained')is not True):
        raise ValueError('correction failed incident detail')
    return dict(path=str(path),process=fingerprint(raw))

def physical_interrupted_guards(path):
    """Validate the closed-world, noncanonical interrupted correction wave."""
    root=Path(path).resolve()
    if (root.name!='rehearsal-r-guards-4ae10ae-interrupted-gyj6_g4t'
        or not root.is_dir() or any((root/name).exists() for name in ('process.json','scientific.json','receipt.json'))):
        raise ValueError('correction interrupted incident path')
    expected_nodes={'node-%04d'%index for index in range(90,221)}
    actual_nodes=set();rows=[];total=0
    for entry in root.iterdir():
        if entry.is_symlink() or (hasattr(entry,'is_junction') and entry.is_junction()):
            raise ValueError('correction interrupted incident reparse')
        if not entry.is_dir() or entry.name not in expected_nodes:
            raise ValueError('correction interrupted incident root layout')
        actual_nodes.add(entry.name)
    if actual_nodes!=expected_nodes:raise ValueError('correction interrupted incident node inventory')
    for target in sorted(root.rglob('*'),key=lambda item:item.relative_to(root).as_posix()):
        if target.is_symlink() or (hasattr(target,'is_junction') and target.is_junction()):
            raise ValueError('correction interrupted incident reparse')
        if target.is_dir():continue
        raw=read(target);total+=len(raw)
        rows.append(dict(path=target.relative_to(root).as_posix(),bytes=len(raw),sha256=sha256(raw).hexdigest()))
    descriptor=dict(files=len(rows),bytes=total,sha256=sha256(canonical(rows)).hexdigest())
    if not exact_json(descriptor,PHYSICAL_INTERRUPTED_GUARDS_MANIFEST):
        raise ValueError('correction interrupted incident manifest')
    for index in range(90,218):
        node=root/('node-%04d'%index)
        process=environment.strict(read(node/'process.json'))
        if process.get('status')!='PASSED' or process.get('active_processes')!=0:
            raise ValueError('correction interrupted incident accepted child layout')
    for index in range(218,221):
        node=root/('node-%04d'%index)
        if set(item.name for item in node.iterdir())!={'lease.json','review.json','stderr.log','stdout.log','worker-attempt.json'}:
            raise ValueError('correction interrupted incident incomplete child layout')
    return dict(schema='GE_BEAM3_G3C_PHYSICAL_INTERRUPTED_GUARD_INCIDENT_V1',
        path=str(root),files=descriptor['files'],bytes=descriptor['bytes'],manifest_sha256=descriptor['sha256'])


def physical_validate_runtime_compatibility_record(value,expected,predecessor=None,live=None):
    candidate,inputs,review_raw=expected;live=physical_support().runtime_identity() if live is None else live
    keys={'schema','mode','predecessor','successor','addendum_sha256','design_review_sha256',
          'revision_sha256','revision_review_sha256','unchanged_inputs_sha256',
          'partition_recovery_sha256','partition_recovery_review_sha256',
          'guard_segment_manifest_sha256','allowed_changed_paths','self_sha256'}
    if type(value)is not dict or set(value)!=keys:raise ValueError('correction compatibility schema')
    body={key:item for key,item in value.items()if key!='self_sha256'}
    expected_predecessor=dict(commit=PHYSICAL_PREDECESSOR['commit'],tree=PHYSICAL_PREDECESSOR['tree'],
        runtime_sha256=PHYSICAL_PREDECESSOR_RUNTIME_SHA,review_sha256=PHYSICAL_PREDECESSOR_REVIEW_SHA)
    expected_successor=dict(commit=candidate['commit'],tree=candidate['tree'],runtime_sha256=live,
        review_sha256=sha256(review_raw).hexdigest())
    if (value.get('schema')!=physical_support().CORRECTION_COMPATIBILITY_SCHEMA
        or value.get('mode')!='R-GUARDS'
        or not exact_json(value.get('predecessor'),expected_predecessor)
        or not exact_json(value.get('successor'),expected_successor)
        or value.get('addendum_sha256')!=PHYSICAL_CORRECTION_ADDENDUM_SHA
        or value.get('design_review_sha256')!=PHYSICAL_CORRECTION_REVIEW_SHA
        or value.get('revision_sha256')!=PHYSICAL_CORRECTION_REVISION_SHA
        or value.get('revision_review_sha256')!=PHYSICAL_CORRECTION_REVISION_REVIEW_SHA
        or value.get('partition_recovery_sha256')!=PHYSICAL_CORRECTION_RECOVERY_SHA
        or value.get('partition_recovery_review_sha256')!=PHYSICAL_CORRECTION_RECOVERY_REVIEW_SHA
        or value.get('guard_segment_manifest_sha256')!=PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA
        or not exact_json(value.get('allowed_changed_paths'),sorted(PHYSICAL_CORRECTION_CHANGED_PATHS))
        or value.get('unchanged_inputs_sha256')!=PHYSICAL_CORRECTION_UNCHANGED_INPUTS_SHA
        or value.get('self_sha256')!=sha256(canonical(body)).hexdigest()):
        raise ValueError('correction compatibility authority')
    sha_value(value['unchanged_inputs_sha256']);sha_value(value['self_sha256'])
    if predecessor is not None:
        old_inputs=predecessor[1]
        changed={name for name in set(inputs)|set(old_inputs)
                 if name not in inputs or name not in old_inputs or not exact_json(inputs[name],old_inputs[name])}
        if changed!=PHYSICAL_CORRECTION_CHANGED_PATHS:raise ValueError('correction exact changed input set')
        unchanged={name:inputs[name] for name in sorted(set(inputs)&set(old_inputs)-changed)}
        if value['unchanged_inputs_sha256']!=sha256(canonical(unchanged)).hexdigest():
            raise ValueError('correction unchanged input authority')
    return value


def physical_runtime_compatibility(expected,predecessor):
    current_inputs=expected[1];old_inputs=predecessor[1]
    changed={name for name in set(current_inputs)|set(old_inputs)
             if name not in current_inputs or name not in old_inputs or not exact_json(current_inputs[name],old_inputs[name])}
    if changed!=PHYSICAL_CORRECTION_CHANGED_PATHS:raise ValueError('correction exact changed input set')
    unchanged={name:current_inputs[name] for name in sorted(set(current_inputs)&set(old_inputs)-changed)}
    live=physical_support().runtime_identity()
    if live==PHYSICAL_PREDECESSOR_RUNTIME_SHA:raise ValueError('correction successor runtime not distinct')
    body=dict(schema=physical_support().CORRECTION_COMPATIBILITY_SCHEMA,mode='R-GUARDS',
        predecessor=dict(commit=PHYSICAL_PREDECESSOR['commit'],tree=PHYSICAL_PREDECESSOR['tree'],
            runtime_sha256=PHYSICAL_PREDECESSOR_RUNTIME_SHA,review_sha256=PHYSICAL_PREDECESSOR_REVIEW_SHA),
        successor=dict(commit=expected[0]['commit'],tree=expected[0]['tree'],runtime_sha256=live,
            review_sha256=sha256(expected[2]).hexdigest()),
        addendum_sha256=PHYSICAL_CORRECTION_ADDENDUM_SHA,design_review_sha256=PHYSICAL_CORRECTION_REVIEW_SHA,
        revision_sha256=PHYSICAL_CORRECTION_REVISION_SHA,
        revision_review_sha256=PHYSICAL_CORRECTION_REVISION_REVIEW_SHA,
        partition_recovery_sha256=PHYSICAL_CORRECTION_RECOVERY_SHA,
        partition_recovery_review_sha256=PHYSICAL_CORRECTION_RECOVERY_REVIEW_SHA,
        guard_segment_manifest_sha256=PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA,
        unchanged_inputs_sha256=sha256(canonical(unchanged)).hexdigest(),
        allowed_changed_paths=sorted(PHYSICAL_CORRECTION_CHANGED_PATHS))
    value=dict(body,self_sha256=sha256(canonical(body)).hexdigest())
    physical_validate_runtime_compatibility_record(value,expected,predecessor,live)
    return value

def physical_partition_receipt(root,base,indices,partition_id,process,science_raw,prior_paths):
    expected=(base['candidate'],base['inputs'],canonical(base['implementation_review']))
    nodes=[]
    for index in indices:
        out=root/f'node-{index:04d}';lease=environment.strict(read(out/'lease.json'))
        physical_lease_expected(lease,expected,base['review_sha256'])
        physical_verify_node(out,lease);physical_verify_process(out,lease)
        nodes.append(dict(assignment_index=index,
            lease=fingerprint(read(out/'lease.json')),review=fingerprint(read(out/'review.json')),
            completion=fingerprint(read(out/'completion.json')),process=fingerprint(read(out/'process.json')),
            science=fingerprint(read(out/'scientific.node.json'))))
    ordinal,_=physical_partition_spec(partition_id,base['inventory'])
    identities=['smoke','local']+list(PHYSICAL_PARTITION_IDS[:ordinal])
    if len(prior_paths)!=len(identities):raise ValueError('physical partition prerequisite count')
    prerequisites=[physical_evidence_descriptor(path,'base' if i<2 else 'partition',identity)
                   for i,(path,identity) in enumerate(zip(prior_paths,identities))]
    body=dict(schema='GE_BEAM3_G3C_PHYSICAL_PARTITION_RECEIPT_V1',candidate=base['candidate'],
        lane='rehearsal',partition_id=partition_id,partition_manifest_sha256=PHYSICAL_PARTITION_MANIFEST_SHA,
        whole_inventory_sha256=PHYSICAL_PARTITION_INVENTORY_SHA,
        implementation_review_sha256=base['review_sha256'],
        inputs_sha256=sha256(canonical(base['inputs'])).hexdigest(),prerequisites=prerequisites,
        wave_process=fingerprint(read(root/'process.json')),science=fingerprint(science_raw),nodes=nodes,passed=True)
    return dict(body,self_sha256=sha256(canonical(body)).hexdigest())

def physical_validate_partition(path,expected,partition_id,stack=(),runtime_sha=None):
    """Recursively validate one accepted partition and its complete input DAG."""
    if partition_id in stack:raise ValueError('physical partition prerequisite cycle')
    path=Path(path).resolve();raw=read(path);value=environment.strict(raw)
    candidate,inputs,review_raw=expected;review_sha=sha256(review_raw).hexdigest()
    inventory_rows=physical_inventory('rehearsal');ordinal,spec=physical_partition_spec(partition_id,inventory_rows)
    indices=spec['indices'];records=value.get('records')
    body={k:v for k,v in value.items()if k!='self_sha256'}
    if (set(value)!={'schema','candidate','lane','partition_id','partition_manifest_sha256',
                    'whole_inventory_sha256','indices','records','passed','terminal',
                    'full_g3c_qualified','production_qualified','self_sha256'}
        or value.get('schema')!='GE_BEAM3_G3C_PHYSICAL_PARTITION_AGGREGATE_V1'
        or sha256(canonical(body)).hexdigest()!=value.get('self_sha256','')
        or not exact_json(value.get('candidate'),candidate) or value.get('lane')!='rehearsal'
        or value.get('partition_id')!=partition_id
        or value.get('partition_manifest_sha256')!=PHYSICAL_PARTITION_MANIFEST_SHA
        or value.get('whole_inventory_sha256')!=PHYSICAL_PARTITION_INVENTORY_SHA
        or not exact_json(value.get('indices'),indices) or value.get('passed')is not True
        or value.get('terminal')!='COMPLETE_GE_BEAM3_G3C_PHYSICAL_REHEARSAL_PARTITION_ONLY'
        or value.get('full_g3c_qualified')is not False or value.get('production_qualified')is not False
        or type(records)is not list or len(records)!=len(indices)):
        raise ValueError('physical partition aggregate')
    for offset,(index,row) in enumerate(zip(indices,records)):
        science=row.get('science') if type(row)is dict else None
        if (set(row)!={'assignment_index','science_sha256','science'}
            or type(row['assignment_index'])is not int or row['assignment_index']!=index
            or type(science)is not dict or type(row['science_sha256'])is not str
            or row['science_sha256']!=sha256(canonical(science)).hexdigest()
            or science.get('schema')!='GE_BEAM3_G3C_PHYSICAL_NODE_SCIENCE_V1'
            or not exact_json(science.get('candidate'),candidate) or science.get('lane')!='rehearsal'
            or type(science.get('assignment_index'))is not int or science.get('assignment_index')!=index
            or not exact_json(science.get('assignment'),inventory_rows[index])
            or science.get('full_g3c_qualified')is not False
            or science.get('production_qualified')is not False):
            raise ValueError('physical partition record')
    receipt_raw=read(path.parent/'receipt.json');receipt=environment.strict(receipt_raw)
    receipt_body={k:v for k,v in receipt.items()if k!='self_sha256'}
    expected_priors=['smoke','local']+list(PHYSICAL_PARTITION_IDS[:ordinal])
    if (set(receipt)!={'schema','candidate','lane','partition_id','partition_manifest_sha256',
                      'whole_inventory_sha256','implementation_review_sha256','inputs_sha256',
                      'prerequisites','wave_process','science','nodes','passed','self_sha256'}
        or receipt.get('schema')!='GE_BEAM3_G3C_PHYSICAL_PARTITION_RECEIPT_V1'
        or not exact_json(receipt.get('candidate'),candidate) or receipt.get('lane')!='rehearsal'
        or receipt.get('partition_id')!=partition_id
        or receipt.get('partition_manifest_sha256')!=PHYSICAL_PARTITION_MANIFEST_SHA
        or receipt.get('whole_inventory_sha256')!=PHYSICAL_PARTITION_INVENTORY_SHA
        or receipt.get('implementation_review_sha256')!=review_sha
        or receipt.get('inputs_sha256')!=sha256(canonical(inputs)).hexdigest()
        or not exact_fingerprint(receipt.get('science'),raw) or receipt.get('passed')is not True
        or type(receipt.get('prerequisites'))is not list
        or [row.get('identity') if type(row)is dict else None for row in receipt['prerequisites']]!=expected_priors
        or type(receipt.get('nodes'))is not list or len(receipt['nodes'])!=len(indices)
        or sha256(canonical(receipt_body)).hexdigest()!=receipt.get('self_sha256','')):
        raise ValueError('physical partition receipt')
    prerequisite_paths=[]
    for position,descriptor in enumerate(receipt['prerequisites']):
        expected_kind='base' if position<2 else 'partition'
        if descriptor.get('kind')!=expected_kind:raise ValueError('physical partition prerequisite kind')
        prerequisite_paths.append(physical_verify_evidence_descriptor(descriptor))
    verify_physical_priors(prerequisite_paths[:2],expected,'rehearsal',runtime_sha)
    for prior_id,prior_path in zip(PHYSICAL_PARTITION_IDS[:ordinal],prerequisite_paths[2:]):
        _,prior_receipt=physical_validate_partition(prior_path,expected,prior_id,stack+(partition_id,),runtime_sha)
        physical_join_prerequisite_chain(receipt['prerequisites'],prior_receipt['prerequisites'])
    history_outputs=(physical_history_outputs_from_partition(path) if partition_id=='R-HISTORY'
                     else physical_history_outputs_from_partition(prerequisite_paths[2]))
    physical_closed_world(path.parent,{'process.json','receipt.json','scientific.json'},
        {'node-%04d'%index for index in indices})
    wave_raw=read(path.parent/'process.json')
    if not exact_fingerprint(receipt['wave_process'],wave_raw):raise ValueError('physical partition process hash')
    wave=environment.strict(wave_raw)
    if (set(wave)!={'schema','lane','partition_id','partition_manifest_sha256','passed','required_nodes',
                   'terminal_nodes','elapsed_seconds','active_processes','results'}
        or wave.get('schema')!='GE_BEAM3_G3C_PHYSICAL_PARTITION_PROCESS_V1'
        or wave.get('lane')!='rehearsal' or wave.get('partition_id')!=partition_id
        or wave.get('partition_manifest_sha256')!=PHYSICAL_PARTITION_MANIFEST_SHA
        or wave.get('passed')is not True or type(wave.get('required_nodes'))is not int
        or wave.get('required_nodes')!=len(indices) or type(wave.get('terminal_nodes'))is not int
        or wave.get('terminal_nodes')!=len(indices) or type(wave.get('active_processes'))is not int
        or wave.get('active_processes')!=0 or type(wave.get('elapsed_seconds'))is not float
        or not math.isfinite(wave['elapsed_seconds']) or wave['elapsed_seconds']<0
        or type(wave.get('results'))is not dict or set(wave['results'])!={str(i) for i in indices}):
        raise ValueError('physical partition process')
    for index,node in zip(indices,receipt['nodes']):
        out=path.parent/f'node-{index:04d}'
        if (set(node)!={'assignment_index','lease','review','completion','process','science'}
            or type(node['assignment_index'])is not int or node['assignment_index']!=index):
            raise ValueError('physical partition node receipt')
        for key,name in (('lease','lease.json'),('review','review.json'),('completion','completion.json'),
                         ('process','process.json'),('science','scientific.node.json')):
            if not exact_fingerprint(node[key],read(out/name)):raise ValueError('physical partition node hash')
        lease=environment.strict(read(out/'lease.json'))
        if read(out/'review.json')!=review_raw:raise ValueError('physical partition review bytes')
        physical_lease_expected(lease,expected,review_sha,runtime_sha);physical_verify_node(out,lease);physical_verify_process(out,lease)
        physical_join_input_packets(lease,history_outputs)
        if not exact_json(wave['results'][str(index)],environment.strict(read(out/'process.json'))):
            raise ValueError('physical partition process DAG')
        record=records[indices.index(index)]
        if record['science_sha256']!=node['science']['sha256']:
            raise ValueError('physical partition aggregate DAG')
    return value,receipt

def physical_verify_partition_priors(paths,expected,partition_id):
    ordinal,_=physical_partition_spec(partition_id)
    if len(paths)!=2+ordinal:raise ValueError('exact physical partition prerequisite count')
    verify_physical_priors(paths[:2],expected,'rehearsal')
    for prior_id,path in zip(PHYSICAL_PARTITION_IDS[:ordinal],paths[2:]):
        physical_validate_partition(path,expected,prior_id)

def physical_history_outputs_from_partition(path):
    outputs={};inventory_rows=physical_inventory('rehearsal')
    for index in physical_partition_spec('R-HISTORY',inventory_rows)[1]['indices']:
        assignment=inventory_rows[index]
        outputs[assignment['case']['case_id']]=physical_history_packets(Path(path).parent/f'node-{index:04d}',assignment)
    return outputs

def execute_physical_partition(args,watchdog,expected):
    started=time.monotonic();prior_paths=args.prior or []
    physical_verify_partition_priors(prior_paths,expected,args.partition_id)
    inventory_rows=physical_inventory('rehearsal');_,spec=physical_partition_spec(args.partition_id,inventory_rows)
    indices=spec['indices'];runtime=physical_support().runtime_identity()
    root=Path(tempfile.mkdtemp(prefix='anyfesolver-g3c-physical-partition-'));print('DIAGNOSTICS '+str(root),flush=True)
    base=dict(lane='rehearsal',candidate=expected[0],inputs=expected[1],review_sha256=args.review_sha256,
        implementation_review=environment.strict(expected[2]),runtime_sha256=runtime,inventory=inventory_rows)
    outputs={} if args.partition_id=='R-HISTORY' else physical_history_outputs_from_partition(prior_paths[2])
    results={};deadline=started+1800
    passed=physical_run_phase(root,indices,base,expected[2],watchdog,deadline,outputs,results)
    if time.monotonic()>=deadline:passed=False
    if authority(args.review,args.review_sha256,'g3c-physical')!=expected:passed=False
    process=dict(schema='GE_BEAM3_G3C_PHYSICAL_PARTITION_PROCESS_V1',lane='rehearsal',
        partition_id=args.partition_id,partition_manifest_sha256=PHYSICAL_PARTITION_MANIFEST_SHA,passed=passed,
        required_nodes=len(indices),terminal_nodes=len(results),elapsed_seconds=time.monotonic()-started,
        active_processes=sum(1 for row in results.values()if row.get('active_processes',0)),
        results={str(key):results[key] for key in sorted(results)})
    atomic_canonical(root/'process.json',process,watchdog)
    if not passed:
        print(canonical(process).decode(),flush=True);return 1
    records=[]
    for index in indices:
        raw=read(root/f'node-{index:04d}'/'scientific.node.json')
        records.append(dict(assignment_index=index,science_sha256=sha256(raw).hexdigest(),science=environment.strict(raw)))
    value=dict(schema='GE_BEAM3_G3C_PHYSICAL_PARTITION_AGGREGATE_V1',candidate=expected[0],lane='rehearsal',
        partition_id=args.partition_id,partition_manifest_sha256=PHYSICAL_PARTITION_MANIFEST_SHA,
        whole_inventory_sha256=PHYSICAL_PARTITION_INVENTORY_SHA,indices=indices,records=records,passed=True,
        terminal='COMPLETE_GE_BEAM3_G3C_PHYSICAL_REHEARSAL_PARTITION_ONLY',
        full_g3c_qualified=False,production_qualified=False)
    value['self_sha256']=sha256(canonical(value)).hexdigest()
    atomic_canonical(root/'scientific.json',value,watchdog);science_raw=read(root/'scientific.json')
    receipt=physical_partition_receipt(root,base,indices,args.partition_id,process,science_raw,prior_paths)
    atomic_canonical(root/'receipt.json',receipt,watchdog)
    physical_validate_partition(root/'scientific.json',expected,args.partition_id)
    print(canonical(process).decode(),flush=True);return 0

def physical_union_receipt(root,expected,science_raw,prior_paths):
    identities=['smoke','local']+list(PHYSICAL_PARTITION_IDS)
    prerequisites=[physical_evidence_descriptor(path,'base' if i<2 else 'partition',identity)
                   for i,(path,identity) in enumerate(zip(prior_paths,identities))]
    body=dict(schema='GE_BEAM3_G3C_PHYSICAL_UNION_RECEIPT_V1',candidate=expected[0],lane='rehearsal',
        partition_manifest_sha256=PHYSICAL_PARTITION_MANIFEST_SHA,
        whole_inventory_sha256=PHYSICAL_PARTITION_INVENTORY_SHA,
        implementation_review_sha256=sha256(expected[2]).hexdigest(),
        inputs_sha256=sha256(canonical(expected[1])).hexdigest(),prerequisites=prerequisites,
        union_process=fingerprint(read(root/'process.json')),science=fingerprint(science_raw),passed=True)
    return dict(body,self_sha256=sha256(canonical(body)).hexdigest())

def physical_union_records(partition_values):
    if type(partition_values)is not list or len(partition_values)!=len(PHYSICAL_PARTITION_IDS):
        raise ValueError('physical union exact partition count')
    records=[]
    for expected_id,value in zip(PHYSICAL_PARTITION_IDS,partition_values):
        if type(value)is not dict or value.get('partition_id')!=expected_id or type(value.get('records'))is not list:
            raise ValueError('physical union partition order')
        records.extend(value['records'])
    actual=[row.get('assignment_index') if type(row)is dict else None for row in records]
    expected=list(range(len(physical_inventory('rehearsal'))))
    if not exact_json(actual,expected):
        raise ValueError('physical union exact coverage')
    return records

def physical_validate_union(path,expected):
    path=Path(path).resolve();raw=read(path);value=environment.strict(raw);inventory_rows=physical_inventory('rehearsal')
    body={k:v for k,v in value.items()if k!='self_sha256'};records=value.get('records')
    if (set(value)!={'schema','candidate','lane','inventory_sha256','records','passed','terminal',
                    'full_g3c_qualified','production_qualified','self_sha256'}
        or value.get('schema')!='GE_BEAM3_G3C_PHYSICAL_AGGREGATE_V1'
        or sha256(canonical(body)).hexdigest()!=value.get('self_sha256','')
        or not exact_json(value.get('candidate'),expected[0]) or value.get('lane')!='rehearsal'
        or value.get('inventory_sha256')!=PHYSICAL_PARTITION_INVENTORY_SHA
        or value.get('passed')is not True
        or value.get('terminal')!='COMPLETE_GE_BEAM3_G3C_PHYSICAL_REHEARSAL_ONLY'
        or value.get('full_g3c_qualified')is not False or value.get('production_qualified')is not False
        or type(records)is not list or len(records)!=len(inventory_rows)):
        raise ValueError('physical union aggregate')
    if not exact_json([row.get('assignment_index') if type(row)is dict else None for row in records],
                      list(range(len(inventory_rows)))):
        raise ValueError('physical union ordering')
    receipt=environment.strict(read(path.parent/'receipt.json'));receipt_body={k:v for k,v in receipt.items()if k!='self_sha256'}
    expected_ids=['smoke','local']+list(PHYSICAL_PARTITION_IDS)
    if (set(receipt)!={'schema','candidate','lane','partition_manifest_sha256','whole_inventory_sha256',
                      'implementation_review_sha256','inputs_sha256','prerequisites','union_process',
                      'science','passed','self_sha256'}
        or receipt.get('schema')!='GE_BEAM3_G3C_PHYSICAL_UNION_RECEIPT_V1'
        or not exact_json(receipt.get('candidate'),expected[0]) or receipt.get('lane')!='rehearsal'
        or receipt.get('partition_manifest_sha256')!=PHYSICAL_PARTITION_MANIFEST_SHA
        or receipt.get('whole_inventory_sha256')!=PHYSICAL_PARTITION_INVENTORY_SHA
        or receipt.get('implementation_review_sha256')!=sha256(expected[2]).hexdigest()
        or receipt.get('inputs_sha256')!=sha256(canonical(expected[1])).hexdigest()
        or not exact_fingerprint(receipt.get('science'),raw) or receipt.get('passed')is not True
        or type(receipt.get('prerequisites'))is not list
        or [row.get('identity') if type(row)is dict else None for row in receipt['prerequisites']]!=expected_ids
        or sha256(canonical(receipt_body)).hexdigest()!=receipt.get('self_sha256','')):
        raise ValueError('physical union receipt')
    paths=[physical_verify_evidence_descriptor(row) for row in receipt['prerequisites']]
    verify_physical_priors(paths[:2],expected,'rehearsal')
    partition_records={}
    for ordinal,(partition_id,prior) in enumerate(zip(PHYSICAL_PARTITION_IDS,paths[2:])):
        partition_value,partition_receipt=physical_validate_partition(prior,expected,partition_id)
        physical_join_prerequisite_chain(receipt['prerequisites'],partition_receipt['prerequisites'])
        if not exact_json(receipt['prerequisites'][2+ordinal],
                          physical_evidence_descriptor(prior,'partition',partition_id)):
            raise ValueError('physical union partition lineage join')
        partition_records[partition_id]=partition_value['records']
    expected_records=physical_union_records([dict(partition_id=partition_id,records=partition_records[partition_id])
                                             for partition_id in PHYSICAL_PARTITION_IDS])
    if not exact_json(records,expected_records):raise ValueError('physical union partition DAG')
    process_raw=read(path.parent/'process.json')
    if not exact_fingerprint(receipt['union_process'],process_raw):raise ValueError('physical union process hash')
    process=environment.strict(process_raw)
    if (set(process)!={'schema','lane','partition_manifest_sha256','passed','required_partitions',
                      'terminal_partitions','elapsed_seconds','active_processes'}
        or process.get('schema')!='GE_BEAM3_G3C_PHYSICAL_UNION_PROCESS_V1'
        or process.get('lane')!='rehearsal' or process.get('partition_manifest_sha256')!=PHYSICAL_PARTITION_MANIFEST_SHA
        or process.get('passed')is not True or type(process.get('required_partitions'))is not int
        or process.get('required_partitions')!=6 or type(process.get('terminal_partitions'))is not int
        or process.get('terminal_partitions')!=6 or type(process.get('active_processes'))is not int
        or process.get('active_processes')!=0 or type(process.get('elapsed_seconds'))is not float
        or not math.isfinite(process['elapsed_seconds']) or process['elapsed_seconds']<0):
        raise ValueError('physical union process')
    physical_closed_world(path.parent,{'process.json','receipt.json','scientific.json'})
    return value,receipt

def execute_physical_union(args,watchdog,expected):
    started=time.monotonic();paths=args.prior or []
    if len(paths)!=8:raise ValueError('exact physical union prerequisite count')
    verify_physical_priors(paths[:2],expected,'rehearsal')
    partition_values=[]
    for partition_id,path in zip(PHYSICAL_PARTITION_IDS,paths[2:]):
        partition_values.append(physical_validate_partition(path,expected,partition_id)[0])
    if authority(args.review,args.review_sha256,'g3c-physical')!=expected:raise ValueError('physical union final authority')
    root=Path(tempfile.mkdtemp(prefix='anyfesolver-g3c-physical-union-'));print('DIAGNOSTICS '+str(root),flush=True)
    process=dict(schema='GE_BEAM3_G3C_PHYSICAL_UNION_PROCESS_V1',lane='rehearsal',
        partition_manifest_sha256=PHYSICAL_PARTITION_MANIFEST_SHA,passed=True,required_partitions=6,
        terminal_partitions=6,elapsed_seconds=time.monotonic()-started,active_processes=0)
    atomic_canonical(root/'process.json',process,watchdog)
    records=physical_union_records(partition_values)
    value=dict(schema='GE_BEAM3_G3C_PHYSICAL_AGGREGATE_V1',candidate=expected[0],lane='rehearsal',
        inventory_sha256=PHYSICAL_PARTITION_INVENTORY_SHA,records=records,passed=True,
        terminal='COMPLETE_GE_BEAM3_G3C_PHYSICAL_REHEARSAL_ONLY',
        full_g3c_qualified=False,production_qualified=False)
    value['self_sha256']=sha256(canonical(value)).hexdigest();atomic_canonical(root/'scientific.json',value,watchdog)
    science_raw=read(root/'scientific.json')
    atomic_canonical(root/'receipt.json',physical_union_receipt(root,expected,science_raw,paths),watchdog)
    physical_validate_union(root/'scientific.json',expected)
    print(canonical(process).decode(),flush=True);return 0


def physical_correction_guard_receipt(root,base,process,science_raw,prior_paths,incident):
    nodes=[];expected=(base['candidate'],base['inputs'],canonical(base['implementation_review']))
    for index in range(90,234):
        out=root/f'node-{index:04d}';lease=environment.strict(read(out/'lease.json'))
        physical_lease_expected(lease,expected,base['review_sha256'])
        physical_verify_node(out,lease);physical_verify_process(out,lease)
        nodes.append(dict(assignment_index=index,lease=fingerprint(read(out/'lease.json')),
            review=fingerprint(read(out/'review.json')),completion=fingerprint(read(out/'completion.json')),
            process=fingerprint(read(out/'process.json')),science=fingerprint(read(out/'scientific.node.json'))))
    identities=('smoke','local','R-HISTORY','R-PREFIX-A','R-PREFIX-B','R-PREFIX-C','R-PREFIX-D')
    prerequisites=[physical_evidence_descriptor(path,'base' if i<2 else 'partition',identity)
                   for i,(path,identity) in enumerate(zip(prior_paths,identities))]
    body=dict(schema='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARDS_RECEIPT_V1',candidate=base['candidate'],
        inherited_candidate=PHYSICAL_PREDECESSOR,lane='rehearsal',partition_id='R-GUARDS',
        partition_manifest_sha256=PHYSICAL_PARTITION_MANIFEST_SHA,
        whole_inventory_sha256=PHYSICAL_PARTITION_INVENTORY_SHA,
        implementation_review_sha256=base['review_sha256'],inputs_sha256=sha256(canonical(base['inputs'])).hexdigest(),
        runtime_compatibility=base['runtime_compatibility'],failed_incident=incident,
        prerequisites=prerequisites,wave_process=fingerprint(read(root/'process.json')),
        science=fingerprint(science_raw),nodes=nodes,passed=True)
    return dict(body,self_sha256=sha256(canonical(body)).hexdigest())


def physical_validate_correction_guards(path,expected,prior_paths,failed_path):
    predecessor,_=physical_correction_predecessor(prior_paths)
    incident=physical_failed_guards(failed_path,predecessor)
    compatibility=physical_runtime_compatibility(expected,predecessor)
    path=Path(path).resolve();raw=read(path);value=environment.strict(raw)
    body={k:v for k,v in value.items()if k!='self_sha256'};records=value.get('records');indices=list(range(90,234))
    if (set(value)!={'schema','candidate','inherited_candidate','lane','partition_id','partition_manifest_sha256',
                    'whole_inventory_sha256','indices','records','passed','terminal','full_g3c_qualified',
                    'production_qualified','self_sha256'}
        or value.get('schema')!='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARDS_AGGREGATE_V1'
        or sha256(canonical(body)).hexdigest()!=value.get('self_sha256')
        or not exact_json(value.get('candidate'),expected[0])
        or not exact_json(value.get('inherited_candidate'),PHYSICAL_PREDECESSOR)
        or value.get('lane')!='rehearsal' or value.get('partition_id')!='R-GUARDS'
        or value.get('partition_manifest_sha256')!=PHYSICAL_PARTITION_MANIFEST_SHA
        or value.get('whole_inventory_sha256')!=PHYSICAL_PARTITION_INVENTORY_SHA
        or not exact_json(value.get('indices'),indices) or type(records)is not list or len(records)!=144
        or value.get('passed')is not True
        or value.get('terminal')!='COMPLETE_GE_BEAM3_G3C_PHYSICAL_CORRECTED_GUARDS_PARTITION_ONLY'
        or value.get('full_g3c_qualified')is not False or value.get('production_qualified')is not False):
        raise ValueError('correction guards aggregate')
    inventory_rows=physical_inventory('rehearsal')
    for index,row in zip(indices,records):
        science=row.get('science') if type(row)is dict else None
        if (type(row)is not dict or set(row)!={'assignment_index','science_sha256','science'}
            or type(row.get('assignment_index'))is not int or row.get('assignment_index')!=index
            or type(science)is not dict or row.get('science_sha256')!=sha256(canonical(science)).hexdigest()
            or science.get('schema')!='GE_BEAM3_G3C_PHYSICAL_NODE_SCIENCE_V1'
            or not exact_json(science.get('candidate'),expected[0]) or science.get('lane')!='rehearsal'
            or type(science.get('assignment_index'))is not int or science.get('assignment_index')!=index
            or not exact_json(science.get('assignment'),inventory_rows[index])):
            raise ValueError('correction guards record')
    receipt=environment.strict(read(path.parent/'receipt.json'));receipt_body={k:v for k,v in receipt.items()if k!='self_sha256'}
    expected_ids=('smoke','local','R-HISTORY','R-PREFIX-A','R-PREFIX-B','R-PREFIX-C','R-PREFIX-D')
    if (set(receipt)!={'schema','candidate','inherited_candidate','lane','partition_id','partition_manifest_sha256',
                      'whole_inventory_sha256','implementation_review_sha256','inputs_sha256',
                      'runtime_compatibility','failed_incident','prerequisites','wave_process','science','nodes',
                      'passed','self_sha256'}
        or receipt.get('schema')!='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARDS_RECEIPT_V1'
        or not exact_json(receipt.get('candidate'),expected[0])
        or not exact_json(receipt.get('inherited_candidate'),PHYSICAL_PREDECESSOR)
        or receipt.get('lane')!='rehearsal' or receipt.get('partition_id')!='R-GUARDS'
        or receipt.get('partition_manifest_sha256')!=PHYSICAL_PARTITION_MANIFEST_SHA
        or receipt.get('whole_inventory_sha256')!=PHYSICAL_PARTITION_INVENTORY_SHA
        or receipt.get('implementation_review_sha256')!=sha256(expected[2]).hexdigest()
        or receipt.get('inputs_sha256')!=sha256(canonical(expected[1])).hexdigest()
        or not exact_json(receipt.get('runtime_compatibility'),compatibility)
        or not exact_json(receipt.get('failed_incident'),incident)
        or type(receipt.get('prerequisites'))is not list or len(receipt['prerequisites'])!=7
        or [row.get('identity') if type(row)is dict else None for row in receipt['prerequisites']]!=list(expected_ids)
        or not exact_fingerprint(receipt.get('science'),raw) or receipt.get('passed')is not True
        or type(receipt.get('nodes'))is not list or len(receipt['nodes'])!=144
        or sha256(canonical(receipt_body)).hexdigest()!=receipt.get('self_sha256')):
        raise ValueError('correction guards receipt')
    for descriptor,actual in zip(receipt['prerequisites'],prior_paths):
        if physical_verify_evidence_descriptor(descriptor)!=Path(actual).resolve():
            raise ValueError('correction guards prerequisite path')
    physical_closed_world(path.parent,{'process.json','receipt.json','scientific.json'},
        {'node-%04d'%i for i in indices})
    process_raw=read(path.parent/'process.json')
    if not exact_fingerprint(receipt['wave_process'],process_raw):raise ValueError('correction guards process hash')
    process=environment.strict(process_raw)
    if (set(process)!={'schema','lane','partition_id','partition_manifest_sha256','passed','required_nodes',
                      'terminal_nodes','elapsed_seconds','active_processes','results'}
        or process.get('schema')!='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARDS_PROCESS_V1'
        or process.get('lane')!='rehearsal' or process.get('passed')is not True
        or process.get('partition_id')!='R-GUARDS'
        or process.get('partition_manifest_sha256')!=PHYSICAL_PARTITION_MANIFEST_SHA
        or type(process.get('required_nodes'))is not int or process.get('required_nodes')!=144
        or type(process.get('terminal_nodes'))is not int or process.get('terminal_nodes')!=144
        or type(process.get('active_processes'))is not int or process.get('active_processes')!=0
        or type(process.get('results'))is not dict
        or type(process.get('elapsed_seconds'))is not float or not math.isfinite(process['elapsed_seconds'])
        or not 0<=process['elapsed_seconds']<1800
        or set(process['results'])!={str(i) for i in indices}):
        raise ValueError('correction guards process')
    history_outputs=physical_history_outputs_from_partition(prior_paths[2])
    for index,node in zip(indices,receipt['nodes']):
        out=path.parent/f'node-{index:04d}';lease=environment.strict(read(out/'lease.json'))
        if (type(node)is not dict or set(node)!={'assignment_index','lease','review','completion','process','science'}
            or type(node.get('assignment_index'))is not int or node.get('assignment_index')!=index):
            raise ValueError('correction guards node order')
        for key,name in (('lease','lease.json'),('review','review.json'),('completion','completion.json'),
                         ('process','process.json'),('science','scientific.node.json')):
            if not exact_fingerprint(node.get(key),read(out/name)):raise ValueError('correction guards node hash')
        physical_lease_expected(lease,expected,sha256(expected[2]).hexdigest())
        if not exact_json(lease.get('runtime_compatibility'),compatibility):raise ValueError('correction node compatibility')
        physical_join_input_packets(lease,history_outputs);physical_verify_node(out,lease);physical_verify_process(out,lease)
        if not exact_json(process['results'][str(index)],environment.strict(read(out/'process.json'))):
            raise ValueError('correction guards process DAG')
        if records[index-90]['science_sha256']!=node['science']['sha256']:
            raise ValueError('correction guards aggregate DAG')
    return value,receipt


def execute_physical_correction_guards(args,watchdog,expected):
    started=time.monotonic();prior_paths=args.prior or []
    predecessor,_=physical_correction_predecessor(prior_paths)
    incident=physical_failed_guards(args.failed_guards_process,predecessor)
    compatibility=physical_runtime_compatibility(expected,predecessor)
    inventory_rows=physical_inventory('rehearsal');indices=list(range(90,234));runtime=physical_support().runtime_identity()
    root=Path(tempfile.mkdtemp(prefix='anyfesolver-g3c-physical-correction-guards-'));print('DIAGNOSTICS '+str(root),flush=True)
    base=dict(lane='rehearsal',candidate=expected[0],inputs=expected[1],review_sha256=args.review_sha256,
        implementation_review=environment.strict(expected[2]),runtime_sha256=runtime,inventory=inventory_rows,
        runtime_compatibility=compatibility)
    outputs=physical_history_outputs_from_partition(prior_paths[2]);results={};deadline=started+1800
    passed=physical_run_phase(root,indices,base,expected[2],watchdog,deadline,outputs,results)
    if time.monotonic()>=deadline or authority(args.review,args.review_sha256,'g3c-physical')!=expected:passed=False
    process=dict(schema='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARDS_PROCESS_V1',lane='rehearsal',
        partition_id='R-GUARDS',partition_manifest_sha256=PHYSICAL_PARTITION_MANIFEST_SHA,passed=passed,
        required_nodes=144,terminal_nodes=len(results),elapsed_seconds=time.monotonic()-started,
        active_processes=sum(1 for row in results.values()if row.get('active_processes',0)),
        results={str(key):results[key] for key in sorted(results)})
    atomic_canonical(root/'process.json',process,watchdog)
    if not passed:print(canonical(process).decode(),flush=True);return 1
    records=[]
    for index in indices:
        raw=read(root/f'node-{index:04d}'/'scientific.node.json')
        records.append(dict(assignment_index=index,science_sha256=sha256(raw).hexdigest(),science=environment.strict(raw)))
    value=dict(schema='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARDS_AGGREGATE_V1',candidate=expected[0],
        inherited_candidate=PHYSICAL_PREDECESSOR,lane='rehearsal',partition_id='R-GUARDS',
        partition_manifest_sha256=PHYSICAL_PARTITION_MANIFEST_SHA,whole_inventory_sha256=PHYSICAL_PARTITION_INVENTORY_SHA,
        indices=indices,records=records,passed=True,
        terminal='COMPLETE_GE_BEAM3_G3C_PHYSICAL_CORRECTED_GUARDS_PARTITION_ONLY',
        full_g3c_qualified=False,production_qualified=False)
    value['self_sha256']=sha256(canonical(value)).hexdigest();atomic_canonical(root/'scientific.json',value,watchdog)
    science_raw=read(root/'scientific.json')
    receipt=physical_correction_guard_receipt(root,base,process,science_raw,prior_paths,incident)
    atomic_canonical(root/'receipt.json',receipt,watchdog)
    physical_validate_correction_guards(root/'scientific.json',expected,prior_paths,args.failed_guards_process)
    print(canonical(process).decode(),flush=True);return 0


def physical_correction_segment_receipt(root,base,segment_id,indices,science_raw,prior_paths,incident,interrupted):
    expected=(base['candidate'],base['inputs'],canonical(base['implementation_review']))
    ordinal,spec=physical_correction_guard_segment_spec(segment_id,base['inventory'])
    nodes=[]
    for index in indices:
        out=root/f'node-{index:04d}';lease=environment.strict(read(out/'lease.json'))
        physical_lease_expected(lease,expected,base['review_sha256'])
        physical_verify_node(out,lease);physical_verify_process(out,lease)
        nodes.append(dict(assignment_index=index,lease=fingerprint(read(out/'lease.json')),
            review=fingerprint(read(out/'review.json')),completion=fingerprint(read(out/'completion.json')),
            process=fingerprint(read(out/'process.json')),science=fingerprint(read(out/'scientific.node.json'))))
    inherited_ids=('smoke','local','R-HISTORY','R-PREFIX-A','R-PREFIX-B','R-PREFIX-C','R-PREFIX-D')
    prerequisites=[physical_evidence_descriptor(path,'base' if i<2 else 'partition',identity)
                   for i,(path,identity) in enumerate(zip(prior_paths[:7],inherited_ids))]
    prior_segments=[physical_evidence_descriptor(path,'partition',identity)
                    for path,identity in zip(prior_paths[7:],PHYSICAL_CORRECTION_GUARD_SEGMENT_IDS[:ordinal])]
    body=dict(schema='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARD_SEGMENT_RECEIPT_V1',
        candidate=base['candidate'],inherited_candidate=PHYSICAL_PREDECESSOR,lane='rehearsal',
        partition_id='R-GUARDS',segment_id=segment_id,
        guard_segment_manifest_sha256=PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA,
        assignments_sha256=spec['assignment_sha256'],whole_inventory_sha256=PHYSICAL_PARTITION_INVENTORY_SHA,
        implementation_review_sha256=base['review_sha256'],inputs_sha256=sha256(canonical(base['inputs'])).hexdigest(),
        runtime_compatibility=base['runtime_compatibility'],failed_incident=incident,
        interrupted_incident=interrupted,prerequisites=prerequisites,prior_segments=prior_segments,
        wave_process=fingerprint(read(root/'process.json')),science=fingerprint(science_raw),nodes=nodes,passed=True)
    return dict(body,self_sha256=sha256(canonical(body)).hexdigest())

def physical_validate_correction_guard_segment(path,expected,prior_paths,segment_id,failed_path,interrupted_root,stack=(),memo=None):
    memo={} if memo is None else memo
    if segment_id in stack:raise ValueError('correction guard segment prerequisite cycle')
    ordinal,spec=physical_correction_guard_segment_spec(segment_id);indices=spec['indices']
    if len(prior_paths)!=7+ordinal:raise ValueError('correction guard segment exact prerequisite count')
    path=Path(path).resolve()
    if segment_id in memo:
        accepted_path,value,receipt=memo[segment_id]
        if path!=accepted_path:raise ValueError('correction guard segment memoized path')
        return value,receipt
    if '_context' in memo:
        predecessor,incident,interrupted,compatibility=memo['_context']
    else:
        predecessor,_=physical_correction_predecessor(prior_paths[:7])
        incident=physical_failed_guards(failed_path,predecessor);interrupted=physical_interrupted_guards(interrupted_root)
        compatibility=physical_runtime_compatibility(expected,predecessor)
        memo['_context']=(predecessor,incident,interrupted,compatibility)
    raw=read(path);value=environment.strict(raw);records=value.get('records')
    body={key:item for key,item in value.items()if key!='self_sha256'}
    if (set(value)!={'schema','candidate','inherited_candidate','lane','partition_id','segment_id',
                    'guard_segment_manifest_sha256','assignments_sha256','whole_inventory_sha256',
                    'indices','records','passed','terminal','full_g3c_qualified','production_qualified','self_sha256'}
        or value.get('schema')!='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARD_SEGMENT_AGGREGATE_V1'
        or value.get('self_sha256')!=sha256(canonical(body)).hexdigest()
        or not exact_json(value.get('candidate'),expected[0])
        or not exact_json(value.get('inherited_candidate'),PHYSICAL_PREDECESSOR)
        or value.get('lane')!='rehearsal' or value.get('partition_id')!='R-GUARDS'
        or value.get('segment_id')!=segment_id
        or value.get('guard_segment_manifest_sha256')!=PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA
        or value.get('assignments_sha256')!=spec['assignment_sha256']
        or value.get('whole_inventory_sha256')!=PHYSICAL_PARTITION_INVENTORY_SHA
        or not exact_json(value.get('indices'),indices) or type(records)is not list or len(records)!=len(indices)
        or value.get('passed')is not True
        or value.get('terminal')!='COMPLETE_GE_BEAM3_G3C_PHYSICAL_CORRECTED_GUARD_SEGMENT_ONLY'
        or value.get('full_g3c_qualified')is not False or value.get('production_qualified')is not False):
        raise ValueError('correction guard segment aggregate')
    inventory_rows=physical_inventory('rehearsal')
    for index,row in zip(indices,records):
        science=row.get('science') if type(row)is dict else None
        if (type(row)is not dict or set(row)!={'assignment_index','science_sha256','science'}
            or type(row.get('assignment_index'))is not int or row.get('assignment_index')!=index
            or type(science)is not dict or row.get('science_sha256')!=sha256(canonical(science)).hexdigest()
            or science.get('schema')!='GE_BEAM3_G3C_PHYSICAL_NODE_SCIENCE_V1'
            or not exact_json(science.get('candidate'),expected[0]) or science.get('lane')!='rehearsal'
            or science.get('assignment_index')!=index or not exact_json(science.get('assignment'),inventory_rows[index])
            or science.get('full_g3c_qualified')is not False or science.get('production_qualified')is not False):
            raise ValueError('correction guard segment record')
    receipt=environment.strict(read(path.parent/'receipt.json'));receipt_body={k:v for k,v in receipt.items()if k!='self_sha256'}
    inherited_ids=('smoke','local','R-HISTORY','R-PREFIX-A','R-PREFIX-B','R-PREFIX-C','R-PREFIX-D')
    expected_keys={'schema','candidate','inherited_candidate','lane','partition_id','segment_id',
        'guard_segment_manifest_sha256','assignments_sha256','whole_inventory_sha256',
        'implementation_review_sha256','inputs_sha256','runtime_compatibility','failed_incident',
        'interrupted_incident','prerequisites','prior_segments','wave_process','science','nodes','passed','self_sha256'}
    if (set(receipt)!=expected_keys
        or receipt.get('schema')!='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARD_SEGMENT_RECEIPT_V1'
        or not exact_json(receipt.get('candidate'),expected[0])
        or not exact_json(receipt.get('inherited_candidate'),PHYSICAL_PREDECESSOR)
        or receipt.get('lane')!='rehearsal' or receipt.get('partition_id')!='R-GUARDS'
        or receipt.get('segment_id')!=segment_id
        or receipt.get('guard_segment_manifest_sha256')!=PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA
        or receipt.get('assignments_sha256')!=spec['assignment_sha256']
        or receipt.get('whole_inventory_sha256')!=PHYSICAL_PARTITION_INVENTORY_SHA
        or receipt.get('implementation_review_sha256')!=sha256(expected[2]).hexdigest()
        or receipt.get('inputs_sha256')!=sha256(canonical(expected[1])).hexdigest()
        or not exact_json(receipt.get('runtime_compatibility'),compatibility)
        or not exact_json(receipt.get('failed_incident'),incident)
        or not exact_json(receipt.get('interrupted_incident'),interrupted)
        or [row.get('identity') if type(row)is dict else None for row in receipt.get('prerequisites',[])]!=list(inherited_ids)
        or [row.get('identity') if type(row)is dict else None for row in receipt.get('prior_segments',[])]!=list(PHYSICAL_CORRECTION_GUARD_SEGMENT_IDS[:ordinal])
        or not exact_fingerprint(receipt.get('science'),raw) or receipt.get('passed')is not True
        or type(receipt.get('nodes'))is not list or len(receipt['nodes'])!=len(indices)
        or receipt.get('self_sha256')!=sha256(canonical(receipt_body)).hexdigest()):
        raise ValueError('correction guard segment receipt')
    for descriptor,actual in zip(receipt['prerequisites'],prior_paths[:7]):
        if physical_verify_evidence_descriptor(descriptor)!=Path(actual).resolve():
            raise ValueError('correction guard segment inherited path')
    for prior_id,descriptor,actual in zip(PHYSICAL_CORRECTION_GUARD_SEGMENT_IDS[:ordinal],receipt['prior_segments'],prior_paths[7:]):
        prior_path=physical_verify_evidence_descriptor(descriptor)
        if prior_path!=Path(actual).resolve():raise ValueError('correction guard segment prior path')
        physical_validate_correction_guard_segment(prior_path,expected,prior_paths[:7]+prior_paths[7:7+PHYSICAL_CORRECTION_GUARD_SEGMENT_IDS.index(prior_id)],
            prior_id,failed_path,interrupted_root,stack+(segment_id,),memo)
    physical_closed_world(path.parent,{'process.json','receipt.json','scientific.json'},
        {'node-%04d'%index for index in indices})
    process_raw=read(path.parent/'process.json')
    if not exact_fingerprint(receipt.get('wave_process'),process_raw):raise ValueError('correction guard segment process hash')
    process=environment.strict(process_raw)
    if (set(process)!={'schema','lane','partition_id','segment_id','guard_segment_manifest_sha256',
                      'assignments_sha256','passed','required_nodes','terminal_nodes','elapsed_seconds',
                      'active_processes','results'}
        or process.get('schema')!='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARD_SEGMENT_PROCESS_V1'
        or process.get('lane')!='rehearsal' or process.get('partition_id')!='R-GUARDS'
        or process.get('segment_id')!=segment_id
        or process.get('guard_segment_manifest_sha256')!=PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA
        or process.get('assignments_sha256')!=spec['assignment_sha256'] or process.get('passed')is not True
        or type(process.get('required_nodes'))is not int or process.get('required_nodes')!=len(indices)
        or type(process.get('terminal_nodes'))is not int or process.get('terminal_nodes')!=len(indices)
        or type(process.get('active_processes'))is not int or process.get('active_processes')!=0
        or type(process.get('elapsed_seconds'))is not float or not math.isfinite(process['elapsed_seconds'])
        or not 0<=process['elapsed_seconds']<1800 or type(process.get('results'))is not dict
        or set(process['results'])!={str(index) for index in indices}):
        raise ValueError('correction guard segment process')
    history_outputs=physical_history_outputs_from_partition(prior_paths[2])
    for index,node in zip(indices,receipt['nodes']):
        out=path.parent/f'node-{index:04d}';lease=environment.strict(read(out/'lease.json'))
        if (type(node)is not dict or set(node)!={'assignment_index','lease','review','completion','process','science'}):
            raise ValueError('correction guard segment node order')
        exact_assignment_index(node.get('assignment_index'),index)
        for key,name in (('lease','lease.json'),('review','review.json'),('completion','completion.json'),
                         ('process','process.json'),('science','scientific.node.json')):
            if not exact_fingerprint(node.get(key),read(out/name)):raise ValueError('correction guard segment node hash')
        physical_lease_expected(lease,expected,sha256(expected[2]).hexdigest())
        if not exact_json(lease.get('runtime_compatibility'),compatibility):raise ValueError('correction segment node compatibility')
        physical_join_input_packets(lease,history_outputs);physical_verify_node(out,lease);physical_verify_process(out,lease)
        if not exact_json(process['results'][str(index)],environment.strict(read(out/'process.json'))):
            raise ValueError('correction guard segment process DAG')
        if records[indices.index(index)]['science_sha256']!=node['science']['sha256']:
            raise ValueError('correction guard segment aggregate DAG')
    memo[segment_id]=(path,value,receipt)
    return value,receipt

def execute_physical_correction_guard_segment(args,watchdog,expected):
    started=time.monotonic();prior_paths=args.prior or [];segment_id=args.guard_segment_id
    ordinal,spec=physical_correction_guard_segment_spec(segment_id);indices=spec['indices']
    if len(prior_paths)!=7+ordinal:raise ValueError('correction guard segment exact prerequisite count')
    predecessor,_=physical_correction_predecessor(prior_paths[:7])
    incident=physical_failed_guards(args.failed_guards_process,predecessor)
    interrupted=physical_interrupted_guards(args.interrupted_guards_root)
    compatibility=physical_runtime_compatibility(expected,predecessor)
    memo={'_context':(predecessor,incident,interrupted,compatibility)}
    for prior_id,prior_path in zip(PHYSICAL_CORRECTION_GUARD_SEGMENT_IDS[:ordinal],prior_paths[7:]):
        prior_ordinal=PHYSICAL_CORRECTION_GUARD_SEGMENT_IDS.index(prior_id)
        physical_validate_correction_guard_segment(prior_path,expected,prior_paths[:7+prior_ordinal],prior_id,
            args.failed_guards_process,args.interrupted_guards_root,memo=memo)
    inventory_rows=physical_inventory('rehearsal')
    root=Path(tempfile.mkdtemp(prefix='anyfesolver-g3c-physical-correction-'+segment_id.lower()+'-'));print('DIAGNOSTICS '+str(root),flush=True)
    base=dict(lane='rehearsal',candidate=expected[0],inputs=expected[1],review_sha256=args.review_sha256,
        implementation_review=environment.strict(expected[2]),runtime_sha256=physical_support().runtime_identity(),
        inventory=inventory_rows,runtime_compatibility=compatibility,guard_segment_id=segment_id,
        guard_segment_manifest_sha256=PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA,
        guard_segment_assignments_sha256=spec['assignment_sha256'])
    outputs=physical_history_outputs_from_partition(prior_paths[2]);results={};deadline=started+1800
    passed=physical_run_phase(root,indices,base,expected[2],watchdog,deadline,outputs,results)
    if time.monotonic()>=deadline or authority(args.review,args.review_sha256,'g3c-physical')!=expected:passed=False
    process=dict(schema='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARD_SEGMENT_PROCESS_V1',lane='rehearsal',
        partition_id='R-GUARDS',segment_id=segment_id,
        guard_segment_manifest_sha256=PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA,
        assignments_sha256=spec['assignment_sha256'],passed=passed,required_nodes=len(indices),
        terminal_nodes=len(results),elapsed_seconds=time.monotonic()-started,
        active_processes=sum(1 for row in results.values()if row.get('active_processes',0)),
        results={str(key):results[key] for key in sorted(results)})
    atomic_canonical(root/'process.json',process,watchdog)
    if not passed:print(canonical(process).decode(),flush=True);return 1
    records=[]
    for index in indices:
        raw=read(root/f'node-{index:04d}'/'scientific.node.json')
        records.append(dict(assignment_index=index,science_sha256=sha256(raw).hexdigest(),science=environment.strict(raw)))
    value=dict(schema='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARD_SEGMENT_AGGREGATE_V1',
        candidate=expected[0],inherited_candidate=PHYSICAL_PREDECESSOR,lane='rehearsal',partition_id='R-GUARDS',
        segment_id=segment_id,guard_segment_manifest_sha256=PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA,
        assignments_sha256=spec['assignment_sha256'],whole_inventory_sha256=PHYSICAL_PARTITION_INVENTORY_SHA,
        indices=indices,records=records,passed=True,
        terminal='COMPLETE_GE_BEAM3_G3C_PHYSICAL_CORRECTED_GUARD_SEGMENT_ONLY',
        full_g3c_qualified=False,production_qualified=False)
    value['self_sha256']=sha256(canonical(value)).hexdigest();atomic_canonical(root/'scientific.json',value,watchdog)
    science_raw=read(root/'scientific.json')
    atomic_canonical(root/'receipt.json',physical_correction_segment_receipt(root,base,segment_id,indices,science_raw,
        prior_paths,incident,interrupted),watchdog)
    physical_validate_correction_guard_segment(root/'scientific.json',expected,prior_paths,segment_id,
        args.failed_guards_process,args.interrupted_guards_root)
    print(canonical(process).decode(),flush=True);return 0

def physical_validate_correction_guards(path,expected,prior_paths,failed_path,interrupted_root):
    predecessor,_=physical_correction_predecessor(prior_paths);incident=physical_failed_guards(failed_path,predecessor)
    interrupted=physical_interrupted_guards(interrupted_root);compatibility=physical_runtime_compatibility(expected,predecessor)
    path=Path(path).resolve();raw=read(path);value=environment.strict(raw);records=value.get('records')
    body={key:item for key,item in value.items()if key!='self_sha256'}
    if (set(value)!={'schema','candidate','inherited_candidate','lane','partition_id','guard_segment_manifest_sha256',
                    'whole_inventory_sha256','indices','records','passed','terminal','full_g3c_qualified',
                    'production_qualified','self_sha256'}
        or value.get('schema')!='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARDS_AGGREGATE_V2'
        or value.get('self_sha256')!=sha256(canonical(body)).hexdigest()
        or not exact_json(value.get('candidate'),expected[0]) or not exact_json(value.get('inherited_candidate'),PHYSICAL_PREDECESSOR)
        or value.get('lane')!='rehearsal' or value.get('partition_id')!='R-GUARDS'
        or value.get('guard_segment_manifest_sha256')!=PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA
        or value.get('whole_inventory_sha256')!=PHYSICAL_PARTITION_INVENTORY_SHA
        or not exact_json(value.get('indices'),list(range(90,234))) or type(records)is not list or len(records)!=144
        or [row.get('assignment_index') if type(row)is dict else None for row in records]!=list(range(90,234))
        or value.get('passed')is not True
        or value.get('terminal')!='COMPLETE_GE_BEAM3_G3C_PHYSICAL_CORRECTED_GUARDS_PARTITION_ONLY'
        or value.get('full_g3c_qualified')is not False or value.get('production_qualified')is not False):
        raise ValueError('correction guard union aggregate')
    receipt=environment.strict(read(path.parent/'receipt.json'));receipt_body={k:v for k,v in receipt.items()if k!='self_sha256'}
    inherited_ids=('smoke','local','R-HISTORY','R-PREFIX-A','R-PREFIX-B','R-PREFIX-C','R-PREFIX-D')
    expected_keys={'schema','candidate','inherited_candidate','lane','partition_id','guard_segment_manifest_sha256',
        'whole_inventory_sha256','implementation_review_sha256','inputs_sha256','runtime_compatibility',
        'failed_incident','interrupted_incident','prerequisites','segments','union_process','science','passed','self_sha256'}
    if (set(receipt)!=expected_keys or receipt.get('schema')!='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARDS_RECEIPT_V2'
        or not exact_json(receipt.get('candidate'),expected[0]) or not exact_json(receipt.get('inherited_candidate'),PHYSICAL_PREDECESSOR)
        or receipt.get('lane')!='rehearsal' or receipt.get('partition_id')!='R-GUARDS'
        or receipt.get('guard_segment_manifest_sha256')!=PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA
        or receipt.get('whole_inventory_sha256')!=PHYSICAL_PARTITION_INVENTORY_SHA
        or receipt.get('implementation_review_sha256')!=sha256(expected[2]).hexdigest()
        or receipt.get('inputs_sha256')!=sha256(canonical(expected[1])).hexdigest()
        or not exact_json(receipt.get('runtime_compatibility'),compatibility)
        or not exact_json(receipt.get('failed_incident'),incident)
        or not exact_json(receipt.get('interrupted_incident'),interrupted)
        or [row.get('identity') if type(row)is dict else None for row in receipt.get('prerequisites',[])]!=list(inherited_ids)
        or [row.get('identity') if type(row)is dict else None for row in receipt.get('segments',[])]!=list(PHYSICAL_CORRECTION_GUARD_SEGMENT_IDS)
        or not exact_fingerprint(receipt.get('science'),raw) or receipt.get('passed')is not True
        or receipt.get('self_sha256')!=sha256(canonical(receipt_body)).hexdigest()):
        raise ValueError('correction guard union receipt')
    for descriptor,actual in zip(receipt['prerequisites'],prior_paths):
        if physical_verify_evidence_descriptor(descriptor)!=Path(actual).resolve():raise ValueError('correction guard union inherited path')
    segment_values=[];segment_paths=[];memo={'_context':(predecessor,incident,interrupted,compatibility)}
    for ordinal,(segment_id,descriptor) in enumerate(zip(PHYSICAL_CORRECTION_GUARD_SEGMENT_IDS,receipt['segments'])):
        segment_path=physical_verify_evidence_descriptor(descriptor);segment_paths.append(segment_path)
        segment_values.append(physical_validate_correction_guard_segment(segment_path,expected,
            list(prior_paths)+segment_paths[:-1],segment_id,failed_path,interrupted_root,memo=memo)[0])
    expected_records=[row for segment in segment_values for row in segment['records']]
    if not exact_json(records,expected_records):raise ValueError('correction guard union segment DAG')
    process_raw=read(path.parent/'process.json');process=environment.strict(process_raw)
    if (not exact_fingerprint(receipt.get('union_process'),process_raw)
        or set(process)!={'schema','lane','partition_id','guard_segment_manifest_sha256','passed',
                         'required_segments','terminal_segments','elapsed_seconds','active_processes'}
        or process.get('schema')!='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARDS_PROCESS_V2'
        or process.get('lane')!='rehearsal' or process.get('partition_id')!='R-GUARDS'
        or process.get('guard_segment_manifest_sha256')!=PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA
        or process.get('passed')is not True or process.get('required_segments')!=6 or process.get('terminal_segments')!=6
        or type(process.get('required_segments'))is not int or type(process.get('terminal_segments'))is not int
        or type(process.get('active_processes'))is not int or process.get('active_processes')!=0
        or type(process.get('elapsed_seconds'))is not float or not math.isfinite(process['elapsed_seconds'])
        or not 0<=process['elapsed_seconds']<1800):raise ValueError('correction guard union process')
    physical_closed_world(path.parent,{'process.json','receipt.json','scientific.json'})
    return value,receipt

def execute_physical_correction_guard_union(args,watchdog,expected):
    started=time.monotonic();paths=args.prior or []
    if len(paths)!=13:raise ValueError('correction guard union exact prerequisite count')
    predecessor,_=physical_correction_predecessor(paths[:7]);incident=physical_failed_guards(args.failed_guards_process,predecessor)
    interrupted=physical_interrupted_guards(args.interrupted_guards_root);segments=[]
    compatibility=physical_runtime_compatibility(expected,predecessor)
    memo={'_context':(predecessor,incident,interrupted,compatibility)}
    for ordinal,(segment_id,path) in enumerate(zip(PHYSICAL_CORRECTION_GUARD_SEGMENT_IDS,paths[7:])):
        segments.append(physical_validate_correction_guard_segment(path,expected,paths[:7+ordinal],segment_id,
            args.failed_guards_process,args.interrupted_guards_root,memo=memo)[0])
    records=[row for segment in segments for row in segment['records']]
    if [row.get('assignment_index') for row in records]!=list(range(90,234)):
        raise ValueError('correction guard union exact ordering')
    root=Path(tempfile.mkdtemp(prefix='anyfesolver-g3c-physical-correction-guards-union-'));print('DIAGNOSTICS '+str(root),flush=True)
    stage=root/'staged';stage.mkdir()
    process=dict(schema='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARDS_PROCESS_V2',lane='rehearsal',partition_id='R-GUARDS',
        guard_segment_manifest_sha256=PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA,passed=True,
        required_segments=6,terminal_segments=6,elapsed_seconds=time.monotonic()-started,active_processes=0)
    atomic_canonical(stage/'process.json',process,watchdog)
    value=dict(schema='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARDS_AGGREGATE_V2',candidate=expected[0],
        inherited_candidate=PHYSICAL_PREDECESSOR,lane='rehearsal',partition_id='R-GUARDS',
        guard_segment_manifest_sha256=PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA,
        whole_inventory_sha256=PHYSICAL_PARTITION_INVENTORY_SHA,indices=list(range(90,234)),records=records,passed=True,
        terminal='COMPLETE_GE_BEAM3_G3C_PHYSICAL_CORRECTED_GUARDS_PARTITION_ONLY',
        full_g3c_qualified=False,production_qualified=False)
    value['self_sha256']=sha256(canonical(value)).hexdigest();atomic_canonical(stage/'scientific.json',value,watchdog)
    inherited_ids=('smoke','local','R-HISTORY','R-PREFIX-A','R-PREFIX-B','R-PREFIX-C','R-PREFIX-D')
    prerequisites=[physical_evidence_descriptor(path,'base' if i<2 else 'partition',identity)
                   for i,(path,identity) in enumerate(zip(paths[:7],inherited_ids))]
    segment_descriptors=[physical_evidence_descriptor(path,'partition',identity)
                         for path,identity in zip(paths[7:],PHYSICAL_CORRECTION_GUARD_SEGMENT_IDS)]
    body=dict(schema='GE_BEAM3_G3C_PHYSICAL_CORRECTION_GUARDS_RECEIPT_V2',candidate=expected[0],
        inherited_candidate=PHYSICAL_PREDECESSOR,lane='rehearsal',partition_id='R-GUARDS',
        guard_segment_manifest_sha256=PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA,
        whole_inventory_sha256=PHYSICAL_PARTITION_INVENTORY_SHA,
        implementation_review_sha256=sha256(expected[2]).hexdigest(),inputs_sha256=sha256(canonical(expected[1])).hexdigest(),
        runtime_compatibility=physical_runtime_compatibility(expected,predecessor),failed_incident=incident,
        interrupted_incident=interrupted,prerequisites=prerequisites,segments=segment_descriptors,
        union_process=fingerprint(read(stage/'process.json')),science=fingerprint(read(stage/'scientific.json')),passed=True)
    receipt=dict(body,self_sha256=sha256(canonical(body)).hexdigest());atomic_canonical(stage/'receipt.json',receipt,watchdog)
    physical_validate_correction_guards(stage/'scientific.json',expected,paths[:7],args.failed_guards_process,args.interrupted_guards_root)
    def final_authority():
        if authority(args.review,args.review_sha256,'g3c-physical')!=expected:raise ValueError('correction guard union final authority')
    promote_physical_correction_bundle(stage,root,watchdog,final_authority)
    print(canonical(process).decode(),flush=True);return 0

def physical_validate_correction_union(path,expected,paths,failed_path,interrupted_root):
    predecessor,old_values=physical_correction_predecessor(paths[:7])
    incident=physical_failed_guards(failed_path,predecessor)
    interrupted=physical_interrupted_guards(interrupted_root)
    guards,_=physical_validate_correction_guards(paths[7],expected,paths[:7],failed_path,interrupted_root)
    expected_records=[]
    for value in old_values:expected_records.extend(value['records'])
    expected_records.extend(guards['records'])
    path=Path(path).resolve();raw=read(path);value=environment.strict(raw)
    body={k:v for k,v in value.items()if k!='self_sha256'}
    if (set(value)!={'schema','candidate','inherited_candidate','lane','inventory_sha256','records','passed',
                    'terminal','full_g3c_qualified','production_qualified','self_sha256'}
        or value.get('schema')!='GE_BEAM3_G3C_PHYSICAL_CORRECTION_UNION_V1'
        or sha256(canonical(body)).hexdigest()!=value.get('self_sha256')
        or not exact_json(value.get('candidate'),expected[0])
        or not exact_json(value.get('inherited_candidate'),PHYSICAL_PREDECESSOR)
        or value.get('lane')!='rehearsal' or value.get('inventory_sha256')!=PHYSICAL_PARTITION_INVENTORY_SHA
        or not exact_json(value.get('records'),expected_records)
        or type(value.get('records'))is not list
        or any(type(row)is not dict or type(row.get('assignment_index'))is not int for row in value['records'])
        or [row.get('assignment_index') for row in value['records']]!=list(range(234))
        or value.get('passed')is not True
        or value.get('terminal')!='COMPLETE_GE_BEAM3_G3C_PHYSICAL_CORRECTED_REHEARSAL_ONLY'
        or value.get('full_g3c_qualified')is not False or value.get('production_qualified')is not False):
        raise ValueError('correction union aggregate')
    receipt=environment.strict(read(path.parent/'receipt.json'));receipt_body={k:v for k,v in receipt.items()if k!='self_sha256'}
    identities=('smoke','local','R-HISTORY','R-PREFIX-A','R-PREFIX-B','R-PREFIX-C','R-PREFIX-D','R-GUARDS')
    if (set(receipt)!={'schema','candidate','inherited_candidate','lane','inventory_sha256',
                      'implementation_review_sha256','inputs_sha256','runtime_compatibility','failed_incident',
                      'interrupted_incident',
                      'prerequisites','union_process','science','passed','self_sha256'}
        or receipt.get('schema')!='GE_BEAM3_G3C_PHYSICAL_CORRECTION_UNION_RECEIPT_V1'
        or not exact_json(receipt.get('candidate'),expected[0])
        or not exact_json(receipt.get('inherited_candidate'),PHYSICAL_PREDECESSOR)
        or receipt.get('lane')!='rehearsal' or receipt.get('inventory_sha256')!=PHYSICAL_PARTITION_INVENTORY_SHA
        or receipt.get('implementation_review_sha256')!=sha256(expected[2]).hexdigest()
        or receipt.get('inputs_sha256')!=sha256(canonical(expected[1])).hexdigest()
        or not exact_json(receipt.get('runtime_compatibility'),physical_runtime_compatibility(expected,predecessor))
        or not exact_json(receipt.get('failed_incident'),incident)
        or not exact_json(receipt.get('interrupted_incident'),interrupted)
        or type(receipt.get('prerequisites'))is not list or len(receipt['prerequisites'])!=8
        or [row.get('identity') if type(row)is dict else None for row in receipt['prerequisites']]!=list(identities)
        or not exact_fingerprint(receipt.get('science'),raw) or receipt.get('passed')is not True
        or sha256(canonical(receipt_body)).hexdigest()!=receipt.get('self_sha256')):
        raise ValueError('correction union receipt')
    for descriptor,actual in zip(receipt['prerequisites'],paths):
        if physical_verify_evidence_descriptor(descriptor)!=Path(actual).resolve():
            raise ValueError('correction union prerequisite path')
    process_raw=read(path.parent/'process.json');process=environment.strict(process_raw)
    if (not exact_fingerprint(receipt.get('union_process'),process_raw)
        or set(process)!={'schema','lane','passed','required_partitions','terminal_partitions',
                         'elapsed_seconds','active_processes'}
        or process.get('schema')!='GE_BEAM3_G3C_PHYSICAL_CORRECTION_UNION_PROCESS_V1'
        or process.get('lane')!='rehearsal' or process.get('passed')is not True
        or type(process.get('required_partitions'))is not int or process.get('required_partitions')!=6
        or type(process.get('terminal_partitions'))is not int or process.get('terminal_partitions')!=6
        or type(process.get('active_processes'))is not int or process.get('active_processes')!=0
        or type(process.get('elapsed_seconds'))is not float
        or not math.isfinite(process['elapsed_seconds']) or not 0<=process['elapsed_seconds']<1800):
        raise ValueError('correction union process')
    physical_closed_world(path.parent,{'process.json','receipt.json','scientific.json'})
    return value,receipt


def execute_physical_correction_union(args,watchdog,expected):
    started=time.monotonic();paths=args.prior or []
    if len(paths)!=8:raise ValueError('correction union exact prerequisite count')
    predecessor,old_values=physical_correction_predecessor(paths[:7])
    incident=physical_failed_guards(args.failed_guards_process,predecessor)
    interrupted=physical_interrupted_guards(args.interrupted_guards_root)
    guards,_=physical_validate_correction_guards(paths[7],expected,paths[:7],args.failed_guards_process,args.interrupted_guards_root)
    records=[]
    for value in old_values:records.extend(value['records'])
    records.extend(guards['records'])
    if (any(type(row)is not dict or type(row.get('assignment_index'))is not int for row in records)
        or [row.get('assignment_index') for row in records]!=list(range(234))):
        raise ValueError('correction union exact ordering')
    root=Path(tempfile.mkdtemp(prefix='anyfesolver-g3c-physical-correction-union-'));print('DIAGNOSTICS '+str(root),flush=True)
    stage=root/'staged';stage.mkdir()
    process=dict(schema='GE_BEAM3_G3C_PHYSICAL_CORRECTION_UNION_PROCESS_V1',lane='rehearsal',passed=True,
        required_partitions=6,terminal_partitions=6,elapsed_seconds=time.monotonic()-started,active_processes=0)
    atomic_canonical(stage/'process.json',process,watchdog)
    value=dict(schema='GE_BEAM3_G3C_PHYSICAL_CORRECTION_UNION_V1',candidate=expected[0],
        inherited_candidate=PHYSICAL_PREDECESSOR,lane='rehearsal',inventory_sha256=PHYSICAL_PARTITION_INVENTORY_SHA,
        records=records,passed=True,terminal='COMPLETE_GE_BEAM3_G3C_PHYSICAL_CORRECTED_REHEARSAL_ONLY',
        full_g3c_qualified=False,production_qualified=False)
    value['self_sha256']=sha256(canonical(value)).hexdigest();atomic_canonical(stage/'scientific.json',value,watchdog)
    identities=('smoke','local','R-HISTORY','R-PREFIX-A','R-PREFIX-B','R-PREFIX-C','R-PREFIX-D','R-GUARDS')
    prerequisites=[physical_evidence_descriptor(path,'base' if i<2 else 'partition',identity)
                   for i,(path,identity) in enumerate(zip(paths,identities))]
    body=dict(schema='GE_BEAM3_G3C_PHYSICAL_CORRECTION_UNION_RECEIPT_V1',candidate=expected[0],
        inherited_candidate=PHYSICAL_PREDECESSOR,lane='rehearsal',inventory_sha256=PHYSICAL_PARTITION_INVENTORY_SHA,
        implementation_review_sha256=sha256(expected[2]).hexdigest(),inputs_sha256=sha256(canonical(expected[1])).hexdigest(),
        runtime_compatibility=physical_runtime_compatibility(expected,predecessor),failed_incident=incident,
        interrupted_incident=interrupted,
        prerequisites=prerequisites,union_process=fingerprint(read(stage/'process.json')),
        science=fingerprint(read(stage/'scientific.json')),passed=True)
    receipt=dict(body,self_sha256=sha256(canonical(body)).hexdigest());atomic_canonical(stage/'receipt.json',receipt,watchdog)
    physical_validate_correction_union(stage/'scientific.json',expected,paths,args.failed_guards_process,args.interrupted_guards_root)
    def final_authority():
        if authority(args.review,args.review_sha256,'g3c-physical')!=expected:
            raise ValueError('correction union final authority')
    promote_physical_correction_bundle(stage,root,watchdog,final_authority)
    print(canonical(process).decode(),flush=True);return 0

def execute_physical(args,watchdog):
    started=time.monotonic();expected=authority(args.review,args.review_sha256,'g3c-physical')
    correction=getattr(args,'inherit_correction',False);failed=getattr(args,'failed_guards_process',None)
    interrupted=getattr(args,'interrupted_guards_root',None);segment=getattr(args,'guard_segment_id',None)
    finalize_segments=getattr(args,'finalize_guard_segments',False)
    formal_measurement=getattr(args,'physical_formal_measurement',False)
    proof_cycle=getattr(args,'physical_proof_compressed_cycle',None)
    if proof_cycle is not None:
        if (args.lane!='formal' or formal_measurement or correction or failed is not None
            or interrupted is not None or segment is not None or finalize_segments
            or args.partition_id is not None or args.finalize_partitions or args.prior):
            raise ValueError('proof-compressed mode is exclusive')
        args.proof_cycle=proof_cycle
        return execute_physical_proof_compressed(args,watchdog,expected)
    if formal_measurement:
        if (args.lane!='formal' or correction or failed is not None or interrupted is not None
            or segment is not None or finalize_segments or args.partition_id is not None
            or args.finalize_partitions or args.prior):
            raise ValueError('formal measurement mode is exclusive')
        return execute_physical_formal_measurement(args,watchdog,expected)
    if correction:
        if args.lane!='rehearsal' or failed is None or interrupted is None:
            raise ValueError('correction inheritance rehearsal authority')
        if args.partition_id=='R-GUARDS' and segment is not None and not finalize_segments and not args.finalize_partitions:
            return execute_physical_correction_guard_segment(args,watchdog,expected)
        if args.partition_id=='R-GUARDS' and segment is None and finalize_segments and not args.finalize_partitions:
            return execute_physical_correction_guard_union(args,watchdog,expected)
        if args.partition_id is None and segment is None and not finalize_segments and args.finalize_partitions:
            return execute_physical_correction_union(args,watchdog,expected)
        raise ValueError('correction inheritance mode')
    if failed is not None or interrupted is not None or segment is not None or finalize_segments:
        raise ValueError('correction arguments are correction-only')
    if args.lane=='formal':raise ValueError('formal physical partition/authorization addendum not frozen')
    if args.lane=='rehearsal':
        if (args.partition_id is None)==(not args.finalize_partitions):
            raise ValueError('select exactly one rehearsal partition mode')
        return (execute_physical_union(args,watchdog,expected) if args.finalize_partitions
                else execute_physical_partition(args,watchdog,expected))
    if args.partition_id is not None or args.finalize_partitions:
        raise ValueError('physical partition mode is rehearsal-only')
    verify_physical_priors(args.prior or [],expected,args.lane)
    inventory_rows=physical_inventory(args.lane);runtime=physical_support().runtime_identity()
    root=Path(tempfile.mkdtemp(prefix='anyfesolver-g3c-physical-'));print('DIAGNOSTICS '+str(root),flush=True)
    base=dict(lane=args.lane,candidate=expected[0],inputs=expected[1],review_sha256=args.review_sha256,
        implementation_review=environment.strict(expected[2]),runtime_sha256=runtime,inventory=inventory_rows)
    history=[i for i,a in enumerate(inventory_rows)if a['kind'] in ('owner-static','owner','history')]
    dependent=[i for i,a in enumerate(inventory_rows)if a['kind'] not in ('owner-static','owner','history')]
    deadline=started+1800;outputs={};results={};passed=physical_run_phase(root,history,base,expected[2],watchdog,deadline,outputs,results)
    if passed and dependent:passed=physical_run_phase(root,dependent,base,expected[2],watchdog,deadline,outputs,results)
    if time.monotonic()>=deadline:passed=False
    if authority(args.review,args.review_sha256,'g3c-physical')!=expected:passed=False
    process=dict(schema='GE_BEAM3_G3C_PHYSICAL_WAVE_PROCESS_V1',lane=args.lane,passed=passed,
        required_nodes=len(inventory_rows),terminal_nodes=len(results),elapsed_seconds=time.monotonic()-started,
        active_processes=sum(1 for r in results.values()if r.get('active_processes',0)),results={str(k):results[k] for k in sorted(results)})
    atomic_canonical(root/'process.json',process,watchdog)
    if not passed:
        print(canonical(process).decode(),flush=True);return 1
    records=[]
    for index in range(len(inventory_rows)):
        raw=read(root/f'node-{index:04d}'/'scientific.node.json')
        records.append(dict(assignment_index=index,science_sha256=sha256(raw).hexdigest(),science=environment.strict(raw)))
    value=dict(schema='GE_BEAM3_G3C_PHYSICAL_AGGREGATE_V1',candidate=expected[0],lane=args.lane,
        inventory_sha256=sha256(canonical(inventory_rows)).hexdigest(),records=records,passed=True,
        terminal=('COMPLETE_GE_BEAM3_G3C_PHYSICAL_'+args.lane.upper()+'_ONLY'),
        full_g3c_qualified=False,production_qualified=False)
    value['self_sha256']=sha256(canonical(value)).hexdigest()
    atomic_canonical(root/'scientific.json',value,watchdog)
    science_raw=read(root/'scientific.json')
    atomic_canonical(root/'receipt.json',physical_make_receipt(root,base,inventory_rows,process,science_raw),watchdog)
    print(canonical(process).decode(),flush=True)
    return 0


def q4_checker(out,fixture,replica,proof_sha,gate='q4-audit'):
    """Independent fresh checker process inside the parent Windows job tree."""
    if gate not in ('q4-audit','q4-affine-exact'):raise ValueError('checker gate')
    fixtures=AFFINE_FIXTURES if gate=='q4-affine-exact' else Q4_FIXTURES
    if fixture not in fixtures or replica not in ('1','2'):raise ValueError('checker identity')
    lease=environment.strict(read(out/'lease.json'))
    if lease['gate']!=gate or lease['lane']!='core':raise ValueError('checker gate authority')
    expected=authority(out/'review.json',lease['review_sha256'],gate)
    validate_lease(lease,expected,lease['review_sha256'],'core',out)
    if any(os.environ.get(key)!='1' for key in THREADS):raise ValueError('checker thread authority')
    directory=out/fixture
    write(directory/('checker'+replica+'.attempt.json'),dict(run_id=lease['run_id'],fixture=fixture,replica=replica))
    proof_raw=read(directory/'proof.json')
    if sha256(proof_raw).hexdigest()!=proof_sha:raise ValueError('checker proof hash')
    proof=environment.strict(proof_raw)
    if canonical(proof)!=proof_raw or proof.get('fixture_id')!=fixture:raise ValueError('canonical proof authority')
    sys.path.insert(0,str(ROOT/'docs/reference_cases'))
    if gate=='q4-affine-exact':
        from ge_beam3_q4_affine_recovery_checker import verify
    else:
        from ge_beam3_q4_recovery_coefficient_checker import verify
    result=verify(proof,lambda message:print('BEAM CHECKPOINT '+message,flush=True))
    if read(directory/'proof.json')!=proof_raw:raise ValueError('checker proof changed')
    if authority(out/'review.json',lease['review_sha256'],gate)!=expected:raise ValueError('checker final authority')
    schema='GE_BEAM3_Q4_AFFINE_CHECKER_RESULT_V1' if gate=='q4-affine-exact' else 'GE_BEAM3_Q4_CHECKER_RESULT_V1'
    write(directory/('checker'+replica+'.json'),dict(schema=schema,
        candidate=expected[0],inputs_sha256=sha256(canonical(expected[1])).hexdigest(),
        proof_sha256=proof_sha,verification=result))
    print('BEAM CHECKPOINT checker complete',flush=True)
    return 0

class WaveWatchdog:
    """Active coordinator deadline, including authority IO and finalization.

    Start drain with twenty seconds left; an independent hard timer exits even
    if draining or coordinator IO stalls. Windows job handles are kill-on-close.
    """
    def __init__(self,timer=threading.Timer,exit_process=os._exit):
        self.expired=threading.Event();self.job=None;self.jobs=[];self.exit_process=exit_process
        self._timer=timer;self._arm()

    def _arm(self):
        self.soft=self._timer(1780,self.expire);self.hard=self._timer(1800,self.hard_exit)
        for t in (self.hard,self.soft):t.daemon=True;t.start()

    def renew_wave(self):
        """Start a new bounded wave only after every prior tree is drained."""
        if self.jobs or self.expired.is_set():raise ValueError('cannot renew active or expired wave')
        self.soft.cancel();self.hard.cancel();self._arm()

    def check(self):
        if self.expired.is_set():raise TimeoutError('whole invocation deadline')

    def attach(self,job):
        self.job=job
        self.jobs.append(job)
        self.check()

    def detach(self,job):
        self.jobs.remove(job)
        if self.job is job:self.job=None

    def expire(self):
        self.expired.set()
        try:
            for job in tuple(self.jobs):job.terminate()
        finally:self.exit_process(124)

    def hard_exit(self):
        self.expired.set()
        self.exit_process(124)

    def close(self):
        self.soft.cancel();self.hard.cancel()


def execute(args):
    watchdog=WaveWatchdog()
    try:
        if args.gate=='q4-affine-numerical':return execute_numerical(args,watchdog)
        if args.gate=='g3c-physical':return execute_physical(args,watchdog)
        return execute_guarded(args,watchdog)
    finally:watchdog.close()


def numerical_assignment(lease,index):
    """No free-form module dispatch: each assignment is one registered node."""
    if lease['gate']!='q4-affine-numerical' or type(index)is not int or not 0<=index<len(lease['selected']):
        raise ValueError('numerical assignment index')
    sha_value(lease['observation_manifest_sha256'])
    return dict(schema='GE_BEAM3_REGISTERED_NODE_ASSIGNMENT_V1',run_id=lease['run_id'],
        index=index,node=lease['selected'][index],gate=lease['gate'],lane=lease['lane'],
        candidate=lease['candidate'],inputs_sha256=sha256(canonical(lease['inputs'])).hexdigest(),
        whole_inventory_sha256=sha256(canonical(lease['selected'])).hexdigest(),
        parent_lease_sha256=sha256(canonical(lease)).hexdigest(),observation_manifest_sha256=lease['observation_manifest_sha256'])


def validate_assignment(value,lease,index):
    if canonical(value)!=canonical(numerical_assignment(lease,index)):raise ValueError('node assignment authority')


def numerical_worker(out,index,assignment_sha):
    lease_raw=read(out/'lease.json');lease=environment.strict(lease_raw)
    if canonical(lease)!=lease_raw:raise ValueError('noncanonical parent lease')
    observations={}
    expected=authority(out/'review.json',lease['review_sha256'],'q4-affine-numerical',observation_capture=observations)
    validate_lease(lease,expected,lease['review_sha256'],lease['lane'],out)
    directory=out/('node-%02d'%index)
    raw=read(directory/'assignment.json')
    if sha256(raw).hexdigest()!=assignment_sha:raise ValueError('assignment hash')
    assignment=environment.strict(raw);validate_assignment(assignment,lease,index)
    claim_attempt(directory,lease['run_id'])
    if any(os.environ.get(k)!='1' for k in THREADS):raise ValueError('numerical threads')
    print('BEAM CHECKPOINT node initialization '+assignment['node'],flush=True)
    os.environ['BEAM_QUALIFICATION_OUTPUT']=str(directory)
    os.environ['BEAM_QUALIFICATION_NODE']=assignment['node']
    sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(CAPSULE.parent/'site'),str(ROOT/'docs/reference_cases')]
    import pytest
    class Recorder:
        def __init__(self):self.passed=[];self.bad=[];self.module=None
        def pytest_collection_modifyitems(self,items):
            if [i.nodeid for i in items]!=[assignment['node']]:raise ValueError('single node collection')
            self.module=items[0].module
        def pytest_runtest_logreport(self,report):
            if report.failed or report.skipped:self.bad.append(report.nodeid)
            if report.when=='call' and report.passed:self.passed.append(report.nodeid)
    recorder=Recorder()
    code=pytest.main(['-vv','-s','-p','no:cacheprovider','--basetemp',str(directory/'pytest'),assignment['node']],plugins=[recorder])
    # Assertions, skips and unexpected exceptions are process/evidence failures,
    # not a typed scientific contradiction and never produce canonical evidence.
    if code!=0 or recorder.bad or recorder.passed!=[assignment['node']]:return 1
    records=recorder.module.SCIENTIFIC_RECORDS
    contradictions=recorder.module.CONTRADICTIONS
    if type(records)is not list or not records or type(contradictions)is not list:raise ValueError('node evidence missing')
    validate_numerical_tables(assignment['node'],records)
    validate_registered_observations(records)
    from ge_beam3_q4_affine_numerical_checker import verify_contradiction
    verified=[]
    for payload in contradictions:
        if (payload.get('candidate_identity')!=lease['candidate']['commit']
            or payload.get('source_identity')!=sha256(canonical(lease['inputs'])).hexdigest()):
            raise ValueError('contradiction source/candidate identity')
        verification=verify_contradiction(payload)
        if type(verification)is not dict or verification.get('accepted') is not True:raise ValueError('contradiction not independently accepted')
        verified.append(dict(payload=payload,verification=verification))
    validate_contradictions(assignment['node'],records,verified,lease,observations)
    if authority(out/'review.json',lease['review_sha256'],lease['gate'])!=expected:raise ValueError('node final authority')
    if read(directory/'assignment.json')!=raw or read(out/'lease.json')!=lease_raw:raise ValueError('node authority changed')
    result=dict(schema='GE_BEAM3_REGISTERED_NUMERICAL_NODE_V1',node=assignment['node'],index=index,
        candidate=lease['candidate'],inputs_sha256=assignment['inputs_sha256'],
        whole_inventory_sha256=assignment['whole_inventory_sha256'],lane=lease['lane'],
        observation_manifest_sha256=lease['observation_manifest_sha256'],
        records=records,contradictions=verified,status='CONTRADICTION' if verified else 'PASSED',
        physical_recovery_scope='REGISTERED_AFFINE_LOCAL_ONLY',full_g3c_qualified=False,production_qualified=False)
    write(directory/'scientific.pending.json',result)
    write(directory/'completion.json',dict(assignment_sha256=assignment_sha,node=assignment['node'],
        scientific=fingerprint(read(directory/'scientific.pending.json'))))
    print('BEAM CHECKPOINT node evidence complete',flush=True)
    return 0


def validate_numerical_result(raw,completion,lease,index,assignment_sha,observations):
    value=environment.strict(raw);assignment=numerical_assignment(lease,index)
    if canonical(value)!=raw:raise ValueError('noncanonical node science')
    if (set(value)!={'schema','node','index','candidate','inputs_sha256','whole_inventory_sha256','lane',
                    'records','contradictions','status','physical_recovery_scope','full_g3c_qualified','production_qualified','observation_manifest_sha256'}
        or value['schema']!='GE_BEAM3_REGISTERED_NUMERICAL_NODE_V1'
        or value['node']!=assignment['node'] or type(value['index'])is not int or value['index']!=index
        or value['candidate']!=lease['candidate'] or value['lane']!=lease['lane']
        or value['inputs_sha256']!=assignment['inputs_sha256']
        or value['whole_inventory_sha256']!=assignment['whole_inventory_sha256']
        or value['observation_manifest_sha256']!=lease['observation_manifest_sha256']
        or type(value['records'])is not list or not value['records']
        or type(value['contradictions'])is not list
        or value['status']!=('CONTRADICTION' if value['contradictions'] else 'PASSED')
        or value['physical_recovery_scope']!='REGISTERED_AFFINE_LOCAL_ONLY'
        or value['full_g3c_qualified'] is not False or value['production_qualified'] is not False
        or completion!={'assignment_sha256':assignment_sha,'node':assignment['node'],'scientific':fingerprint(raw)}):
        raise ValueError('numerical node evidence authority')
    validate_numerical_tables(value['node'],value['records'])
    validate_contradictions(value['node'],value['records'],value['contradictions'],lease,observations)
    return value


def registered_observation_state(table,row_id):
    """Fixture data only. Never construct a facade or evaluate any operator."""
    from anyfesolver import _ge_beam3_g3c_affine_q4_registry as registry
    c=registry.construction(row_construction(table,row_id))
    pose=row_id.rsplit('::',1)[1] if table in ('work','graph','smoke') else 'MIXED'
    amplitude=float(row_id.split('::amplitude=')[1]) if table=='tiny' else 1.
    q,accepted=registry.pose(c.construction_id,pose,amplitude)
    if table=='common_motion':q,accepted,_=registry.common_motion(q,accepted,c.coordinates,int(row_id.split('::motion=')[1]))
    if table=='rebase':q,accepted=registry.rebase(q,accepted)
    return dict(construction_id=c.construction_id,coordinates=c.coordinates.tolist(),q=q.tolist(),
        accepted=accepted.tolist(),normal=c.normal.tolist(),material_direction=c.material_direction.tolist(),
        director_polarity=c.director_polarity)


def observation_specs():
    result=[];path=TESTS['q4-affine-numerical'][0]
    for name,tables in zip(NUMERICAL_TESTS,NUMERICAL_TABLE_IDS):
        for table,ids in tables.items():
            if table in PHYSICAL_TABLES:
                result.extend(dict(node=path+'::'+name,table=table,row_id=identity) for identity in ids)
    result.extend(dict(node=path+'::'+NUMERICAL_SMOKE,table='smoke',row_id=identity)
                  for identity in ('SQUARE::1::ZERO','SQUARE::1::MIXED'))
    return result


def fixture_source_bindings():
    """Whole importable ANYfeSolver Python graph plus frozen fixture/environment."""
    paths={p.relative_to(ROOT).as_posix() for p in (ROOT/'src/anyfesolver').rglob('*.py')}
    paths.update(('docs/reference_cases/ge_beam3_g3c_fixtures_v1.json',
                  'scripts/ge_beam3_g3b_environment.py',JOB))
    return {path:fingerprint(read(ROOT/path).replace(b'\r\n',b'\n')) for path in sorted(paths)}


def fixture_generator_bindings():
    # Individual fixed functions avoid circularity with the later inserted
    # manifest hash. The accepted implementation review binds the whole runner.
    source=read(Path(__file__)).decode().replace('\r\n','\n');tree=ast.parse(source)
    names={'registered_observation_state','observation_specs','fixture_source_bindings',
           'fixture_generator_bindings','observation_authority','prepare_observation_manifest',
           'row_construction','canonical','fingerprint','read'}
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    return {name:sha256(ast.get_source_segment(source,functions[name]).encode()).hexdigest() for name in sorted(names)}


def observation_authority():
    return dict(source_bindings=fixture_source_bindings(),generator_bindings=fixture_generator_bindings(),
        capsule_sha256=CAPSULE_SHA,inventory_sha256=sha256(canonical(observation_specs())).hexdigest())


def prepare_observation_manifest(output,expected_authority_sha256):
    """Reviewed prefreeze fixture-only command, enclosed in existing ProcessJob.

    This is not a scientific node, not a qualification result and not a retry.
    Imported fixture helpers use NumPy/rotation construction, but no facade or
    operator is called. The external caller must enforce the registered limits.
    """
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):raise ValueError('isolated fixture preparation required')
    if any(os.environ.get(k)!='1' for k in THREADS):raise ValueError('fixture thread envelope')
    output=Path(output)
    if not output.is_absolute() or output.exists():raise ValueError('exclusive external fixture output')
    expected=observation_authority()
    if sha256(canonical(expected)).hexdigest()!=expected_authority_sha256:raise ValueError('fixture preparation source authority')
    environment.verify(CAPSULE,CAPSULE_SHA)
    sys.path[:0]=[str(ROOT/'src'),str(CAPSULE.parent/'site')]
    rows=[]
    for spec in observation_specs():
        print('BEAM CHECKPOINT fixture '+spec['row_id'],flush=True)
        state=registered_observation_state(spec['table'],spec['row_id'])
        rows.append(dict(spec,state_sha256=sha256(canonical(state)).hexdigest()))
    environment.verify(CAPSULE,CAPSULE_SHA)
    if observation_authority()!=expected:raise ValueError('fixture preparation final authority changed')
    value=dict(schema='GE_BEAM3_REGISTERED_OBSERVATION_HASHES_V1',**expected,observations=rows,
        fixture_data_only=True,numerical_qualification=False)
    write(output,value)
    print('BEAM CHECKPOINT fixture manifest complete '+sha256(read(output)).hexdigest(),flush=True)


def frozen_observations():
    raw=read(ROOT/OBSERVATION_MANIFEST)
    if sha256(raw).hexdigest()!=OBSERVATION_MANIFEST_SHA:raise ValueError('frozen observation manifest hash')
    value=environment.strict(raw)
    if canonical(value)!=raw:raise ValueError('canonical observation manifest required')
    exact_keys(value,('schema','source_bindings','generator_bindings','capsule_sha256','inventory_sha256',
                      'observations','fixture_data_only','numerical_qualification'))
    expected=observation_authority()
    if (value['schema']!='GE_BEAM3_REGISTERED_OBSERVATION_HASHES_V1'
        or any(value[k]!=v for k,v in expected.items())
        or value['fixture_data_only'] is not True or value['numerical_qualification'] is not False):
        raise ValueError('frozen fixture graph authority')
    specs=observation_specs();rows=value['observations']
    if type(rows)is not list or len(rows)!=len(specs):raise ValueError('observation manifest inventory')
    result={}
    for row,spec in zip(rows,specs):
        exact_keys(row,('node','table','row_id','state_sha256'))
        if any(row[k]!=v for k,v in spec.items()):raise ValueError('observation manifest row identity')
        sha_value(row['state_sha256']);result[(row['node'],row['table'],row['row_id'])]=row['state_sha256']
    return result


def validate_registered_observations(records):
    """Authorized child only: shared fixture recipes, never checker mechanics."""
    for table,rows in records[0]['tables'].items():
        if table not in PHYSICAL_TABLES:continue
        for row in rows:
            state=observed_state(row,table);expected=registered_observation_state(table,row['id'])
            if canonical(state)!=canonical(expected):raise ValueError('observed state is not assigned registered fixture')


def validate_contradictions(node,records,contradictions,lease,observations):
    """Strict stdlib-only binding: one complete receipt per failed row predicate."""
    lookup={(table,row['id']):row for table,rows in records[0]['tables'].items() for row in rows}
    for table,rows in records[0]['tables'].items():
        if table not in PHYSICAL_TABLES:continue
        for row in rows:
            if observations.get((node,table,row['id']))!=row['state_sha256']:raise ValueError('state differs from frozen observation manifest')
    failed={(table,row['id'],predicate) for table,rows in records[0]['tables'].items() if table in PHYSICAL_TABLES
            for row in rows for predicate,check in row['physical_checks'].items() if not check['passed']}
    seen=set()
    for item in contradictions:
        exact_keys(item,('payload','verification'));payload=item['payload'];receipt=item['verification']
        exact_keys(payload,('schema','predicate','coordinates','q','accepted','normal','material_direction',
            'director_polarity','actual','tolerance','scale_mode','source_identity','candidate_identity',
            'fixture_identity','node','table','row_id','state_sha256'))
        exact_keys(receipt,('accepted','predicate','relative_error','node','table','row_id','fixture_identity',
            'payload_sha256','state_sha256','actual','expected'))
        table,row_id,predicate=payload['table'],payload['row_id'],payload['predicate']
        if any(type(v)is not str for v in (table,row_id,predicate)):raise ValueError('contradiction locator types')
        key=(table,row_id,predicate)
        if key not in failed or key in seen:raise ValueError('orphan or reused contradiction')
        row=lookup[(table,row_id)];state=observed_state(row,table);check=row['physical_checks'][predicate]
        if (payload['schema']!='Q4_AFFINE_NUMERICAL_CONTRADICTION_V1' or payload['node']!=node
            or type(payload['tolerance']) not in (int,float) or payload['tolerance']!=1e-11
            or payload['scale_mode']!='REFERENCE_EDGE_NONDIMENSIONAL_V1'
            or payload['source_identity']!=sha256(canonical(lease['inputs'])).hexdigest()
            or payload['candidate_identity']!=lease['candidate']['commit']
            or payload['fixture_identity']!=state['construction_id']):raise ValueError('contradiction assigned authority')
        supplied_state=dict(construction_id=payload['fixture_identity'],**{name:payload[name] for name in
            ('coordinates','q','accepted','normal','material_direction','director_polarity')})
        if canonical(supplied_state)!=canonical(state) or payload['state_sha256']!=row['state_sha256']:raise ValueError('contradiction observed state mismatch')
        actual=numeric_array(payload['actual'],PHYSICAL_SHAPES[predicate])
        if actual!=check['actual']:raise ValueError('contradiction differs from actual row observation')
        array_fingerprint(receipt['actual'],PHYSICAL_SHAPES[predicate]);array_fingerprint(receipt['expected'],PHYSICAL_SHAPES[predicate])
        for keyname in ('payload_sha256','state_sha256'):sha_value(receipt[keyname])
        if (receipt['accepted'] is not True or receipt['predicate']!=predicate or receipt['node']!=node
            or receipt['table']!=table or receipt['row_id']!=row_id or receipt['fixture_identity']!=state['construction_id']
            or receipt['payload_sha256']!=sha256(canonical(payload)).hexdigest()
            or receipt['state_sha256']!=row['state_sha256'] or receipt['actual']!=actual
            or finite_number(receipt['relative_error'])<=1e-11
            or receipt['relative_error']!=check['relative_error']):raise ValueError('independent contradiction receipt mismatch')
        seen.add(key)
    if seen!=failed:raise ValueError('failed physical checks missing verified contradiction')


def exact_keys(value,keys):
    if type(value)is not dict or set(value)!=set(keys):raise ValueError('closed numerical object schema')


def finite_number(value):
    if type(value) not in (float,int) or not math.isfinite(value):raise ValueError('finite numerical scalar required')
    return value


def sha_value(value):
    if type(value)is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('SHA256 encoding')


def residual(value,tolerance=1e-11):
    if not 0<=finite_number(value)<=tolerance:raise ValueError('residual outside registered threshold')


def numeric_array(value,shape):
    """Shape/finite validation and LE-binary64 digest without importing NumPy."""
    numbers=[]
    def visit(v,dimensions):
        if not dimensions:numbers.append(finite_number(v));return
        if type(v)is not list or len(v)!=dimensions[0]:raise ValueError('numerical array shape')
        for item in v:visit(item,dimensions[1:])
    visit(value,shape)
    return dict(shape=list(shape),sha256=sha256(b''.join(struct.pack('<d',v) for v in numbers)).hexdigest())


def array_fingerprint(value,shape):
    exact_keys(value,('shape','sha256'))
    if type(value['shape'])is not list or any(type(v)is not int for v in value['shape']) or value['shape']!=shape:raise ValueError('array digest shape')
    sha_value(value['sha256'])


def physical_check(value,predicate):
    exact_keys(value,('relative_error','passed','actual'))
    error=finite_number(value['relative_error'])
    if error<0 or type(value['passed'])is not bool or value['passed']!=(error<=1e-11):raise ValueError('physical predicate truth mismatch')
    array_fingerprint(value['actual'],PHYSICAL_SHAPES[predicate])


def row_construction(table,identity):
    if table in ('work','graph','smoke'):return identity.rsplit('::',1)[0]
    if table=='common_motion':return identity.split('::motion=')[0]
    if table=='tiny':return identity.split('::amplitude=')[0]
    return identity


def observed_state(row,table):
    state=row['state'];exact_keys(state,('construction_id','coordinates','q','accepted','normal','material_direction','director_polarity'))
    if state['construction_id']!=row_construction(table,row['id']):raise ValueError('row construction mismatch')
    for name,shape in (('coordinates',[4,3]),('q',[24]),('accepted',[4,3,3]),('normal',[3]),('material_direction',[3])):
        numeric_array(state[name],shape)
    if type(state['director_polarity'])is not int or state['director_polarity'] not in (-1,1):raise ValueError('director polarity type')
    if state['director_polarity']!=(-1 if state['construction_id'].endswith('::DIRECTOR:-1') else 1):raise ValueError('registered physical director polarity')
    sha_value(row['state_sha256'])
    if row['state_sha256']!=sha256(canonical(state)).hexdigest():raise ValueError('observed state hash mismatch')
    return state


def validate_numerical_row(table,row):
    if type(row)is not dict or type(row.get('id'))is not str:raise ValueError('numerical row identity schema')
    identity=row['id'];physical=table in PHYSICAL_TABLES
    base={'id','state','state_sha256','physical_checks'} if physical else {'id'}
    if table=='definitions':
        exact_keys(row,base|{'recipe_sha256','descriptor_sha256'})
        sha_value(row['recipe_sha256']);sha_value(row['descriptor_sha256'])
    elif table=='source_graph':
        exact_keys(row,base|{'hashes'})
        if row['hashes']!=NUMERICAL_SOURCE_HASHES:raise ValueError('source graph hashes')
    elif table=='extension_lemma':
        exact_keys(row,base|{'lemma_sha256','review_sha256'})
        if row['lemma_sha256']!='156d33ae5a621b953b5d04050218618f6416bac1042f94d3d3fc5c305c9f8762' or row['review_sha256']!='e28f184023ce1bba825a89079bcd2f9af99cf7861d701d7d918e2d30492b073e':raise ValueError('lemma evidence binding')
    elif table=='station_join':
        exact_keys(row,base|{'station_association_id','verified','rejections'})
        if row['station_association_id']!=STATION_ASSOCIATION_ID or row['verified'] is not True or type(row['rejections'])is not int or row['rejections']!=4:raise ValueError('station coordinate association evidence')
    elif table=='chart_authority':
        exact_keys(row,base|{'chart_numerics_id','addendum_sha256','review_sha256','increment_addendum_sha256','increment_review_sha256','sources','verified'})
        if (row['chart_numerics_id']!=CHART_ID or row['addendum_sha256']!=CHART_ADDENDUM_SHA
            or row['increment_addendum_sha256']!=INCREMENT_ADDENDUM_SHA or row['increment_review_sha256']!=INCREMENT_REVIEW_SHA
            or row['review_sha256']!=CHART_REVIEW_SHA or row['sources']!=CHART_SOURCES
            or row['verified'] is not True):raise ValueError('stable chart caller authority')
    elif table=='fingerprint':
        exact_keys(row,base|{'distinct_fingerprints','nonfinite_rejections','evidence_sha256','verified'})
        sha_value(row['evidence_sha256'])
        if row['verified'] is not True:raise ValueError('typed fingerprint evidence required')
        if type(row['distinct_fingerprints'])is not int or row['distinct_fingerprints']!=20:raise ValueError('typed fingerprint inventory')
        if type(row['nonfinite_rejections'])is not int or row['nonfinite_rejections']!=5:raise ValueError('nonfinite fingerprint inventory')
    elif table=='station':
        exact_keys(row,base|{'checks','energy','stations'});finite_number(row['energy'])
        if type(row['stations'])is not int or row['stations']!=4:raise ValueError('station count')
        exact_keys(row['checks'],('d','D','D2','R','Q','x'))
        for value in row['checks'].values():residual(value)
    elif table=='independent':
        exact_keys(row,base|{'stations'})
        if type(row['stations'])is not list or len(row['stations'])!=4:raise ValueError('independent station inventory')
        for item in row['stations']:
            exact_keys(item,('M','strain','resultant','frame','constitutive'))
            for value in item.values():residual(value)
    elif table=='schur':
        exact_keys(row,base|{'schur','full_internal_dimension'});array_fingerprint(row['schur'],[24,24])
        if type(row['full_internal_dimension'])is not int or row['full_internal_dimension']!=64:raise ValueError('full station dimension')
    elif table=='work':
        exact_keys(row,base|{'checks','chart_image_sentinels'})
        if type(row['chart_image_sentinels'])is not bool:raise ValueError('chart sentinel type')
    elif table=='directional':
        exact_keys(row,base|{'work','tangent'});residual(row['work'],1e-7);residual(row['tangent'],1e-7)
    elif table=='rigid':
        exact_keys(row,base|{'rigid_columns','total_positive_modes','eigenvalues'});array_fingerprint(row['eigenvalues'],[24])
        if type(row['rigid_columns'])is not int or row['rigid_columns']!=6 or type(row['total_positive_modes'])is not int or row['total_positive_modes']!=18:raise ValueError('rigid mode inventory')
    elif table in ('common_motion','passive','rebase','tiny'):
        exact_keys(row,base|{'energy','checks'}|({'chart'} if table=='tiny' else set()));energy=finite_number(row['energy'])
        if table=='tiny' and energy<=0:raise ValueError('positive tiny physical energy required')
        if table=='tiny':
            exact_keys(row['chart'],('d','D','D2','R','Q','x'))
            for value in row['chart'].values():residual(value)
    elif table=='d4':
        exact_keys(row,base|{'station_map','checks'})
        k=int(identity.rsplit(':',1)[1]);corners=(0,1,2,3) if k<4 else (0,3,2,1)
        expected=[(i+k%4)%4 for i in corners]
        if type(row['station_map'])is not list or any(type(i)is not int for i in row['station_map']) or row['station_map']!=expected:raise ValueError('D4 station transport')
    elif table=='director':
        exact_keys(row,base|{'physical_polarity','checks'})
        if type(row['physical_polarity'])is not int or row['physical_polarity']!=int(identity.rsplit(':',1)[1]):raise ValueError('physical polarity authority')
    elif table=='graph':
        exact_keys(row,base|{'node_ids','element_id','recipe_sha256','checks'});sha_value(row['recipe_sha256'])
        ids=[101,102,103,104] if identity.startswith('J_Q4_PAIR::') else [301,302,303,304]
        element=11 if identity.startswith('J_Q4_PAIR::') else 13
        if '::RENUMBERED::' in identity:ids=[10000+7*i for i in ids];element=20000+5*element
        if '::CONNECTIVITY_REVERSED::' in identity:ids=[ids[i] for i in (0,3,2,1)]
        if type(row['node_ids'])is not list or any(type(i)is not int for i in row['node_ids']) or row['node_ids']!=ids or type(row['element_id'])is not int or row['element_id']!=element:raise ValueError('graph source identities')
    elif table=='channels':
        exact_keys(row,base|{'physical','numerical'});finite_number(row['physical'])
        if type(row['numerical'])is not list or len(row['numerical'])!=2:raise ValueError('numerical energy channel inventory')
        for item,name in zip(row['numerical'],('NUMERICAL_PL','NUMERICAL_HOURGLASS')):
            exact_keys(item,('name','energy'))
            if item['name']!=name:raise ValueError('numerical channel identity')
            finite_number(item['energy'])
    elif table=='races':
        exact_keys(row,base|{'rejected_before_family'})
        if row['rejected_before_family'] is not True:raise ValueError('state safety rejection required')
    elif table=='immutability':
        extra={'array_count'} if identity=='all_detached_arrays' else {'rejections'} if identity=='reentry' else {'fingerprint_sha256'} if identity=='nested_candidate_bytes' else set()
        exact_keys(row,base|{'verified','prior_arrays_sha256'}|extra);sha_value(row['prior_arrays_sha256'])
        if row['verified'] is not True:raise ValueError('immutable output evidence required')
        if identity=='all_detached_arrays' and (type(row['array_count'])is not int or row['array_count']<=30):raise ValueError('detached array inventory')
        if identity=='reentry' and (type(row['rejections'])is not int or row['rejections']!=2):raise ValueError('reentry checks')
        if identity=='nested_candidate_bytes':sha_value(row['fingerprint_sha256'])
    elif table=='rejections':
        if identity in ('before_work','before_publication','invalid_callback'):
            exact_keys(row,base|{'callbacks','family_entries','published'})
            callbacks,entries=(2,1) if identity=='before_publication' else (1,0)
            if type(row['callbacks'])is not int or row['callbacks']!=callbacks or type(row['family_entries'])is not int or row['family_entries']!=entries or row['published'] is not False:raise ValueError('cancellation receipt')
        else:
            exact_keys(row,base|{'rejected_before_family'})
            if row['rejected_before_family'] is not True:raise ValueError('admission rejection receipt')
    elif table=='eigen_derivatives':
        exact_keys(row,base|{'derivative_coordinates','derivative_pairs','derivative_residuals'})
        if type(row['derivative_coordinates'])is not int or row['derivative_coordinates']!=24 or type(row['derivative_pairs'])is not int or row['derivative_pairs']!=576:raise ValueError('eigen derivative inventory')
        exact_keys(row['derivative_residuals'],('first_normalization','second_normalization','first_stationarity','second_stationarity','second_symmetry'))
        for value in row['derivative_residuals'].values():residual(value)
    elif table=='chart_mutations':
        compared=identity in ('missing_delta_covariance','naive_reference_subtraction','omitted_delta_derivatives')
        exact_keys(row,base|{'rejection'}|({'relative_error'} if compared else set()))
        expected=('INDEPENDENT_CHART_COMPARISON' if compared else 'BRANCH_REJECTION' if identity in ('wrong_davenport_branch','wrong_polar_branch')
                  else 'DERIVATIVE_IDENTITY_REJECTION' if identity in ('eigen_first_derivative','eigen_second_derivative') else 'NONCONVERGENCE_REJECTION')
        if row['rejection']!=expected:raise ValueError('increment chart mutation mechanism')
        if compared and finite_number(row['relative_error'])<=1e-11:raise ValueError('increment mutation did not violate frozen tolerance')
    elif table=='mutations':
        exact_keys(row,base|{'rejection'});mutation=identity.split('::',1)[1]
        enum=('INDEPENDENT_HESSIAN' if mutation in ('force_weighted_Hessian','chart_second') else
              'STATION_EQUILIBRIUM' if mutation=='coupling_sign' else 'STATION_INVERSE' if mutation=='inverse' else
              'MATERIAL_ENERGY' if mutation=='numerical_energy_leak' else 'INDEPENDENT_STATION_COMPARISON')
        if row['rejection']!=enum:raise ValueError('mutation rejection mechanism')
    elif table=='smoke':exact_keys(row,base|{'checks'})
    else:raise ValueError('unregistered numerical table')
    if physical:
        observed_state(row,table)
        mapping={'energy':'physical_energy','force':'physical_force','hessian':'physical_hessian'}
        if table=='work' and not identity.endswith('::ZERO'):
            mapping.update(source_energy='source_physical_energy',source_force='source_physical_force',source_hessian='source_physical_hessian')
        exact_keys(row['physical_checks'],mapping.values())
        exact_keys(row['checks'],set(mapping)|{'symmetry','spatial_force','spatial_tangent'})
        for key,predicate in mapping.items():
            physical_check(row['physical_checks'][predicate],predicate)
            if row['checks'][key]!=row['physical_checks'][predicate]:raise ValueError('physical check row mismatch')
        for key in ('symmetry','spatial_force','spatial_tangent'):residual(row['checks'][key])
        if table=='work':
            source_failed=any(not value['passed'] for key,value in row['physical_checks'].items() if key.startswith('source_'))
            if row['chart_image_sentinels'] is source_failed:raise ValueError('chart image sentinel disposition')


def validate_numerical_tables(node,records):
    name=node.split('::')[-1]
    tables=({'smoke':2} if name==NUMERICAL_SMOKE else NUMERICAL_TABLES[NUMERICAL_TESTS.index(name)])
    identities=({'smoke':['SQUARE::1::ZERO','SQUARE::1::MIXED']} if name==NUMERICAL_SMOKE else NUMERICAL_TABLE_IDS[NUMERICAL_TESTS.index(name)])
    if len(records)!=1:raise ValueError('one complete named-node record required')
    record=records[0]
    if (type(record)is not dict or set(record)!={'test','tables','full_g3c_qualified','production_qualified'}
        or record['test']!=name.removeprefix('test_affine_recovery_')
        or record['full_g3c_qualified'] is not False or record['production_qualified'] is not False
        or type(record['tables'])is not dict or set(record['tables'])!=set(tables)):
        raise ValueError('numerical named table schema')
    for key,count in tables.items():
        rows=record['tables'][key]
        if (type(rows)is not list or len(rows)!=count or any(type(row)is not dict or type(row.get('id'))is not str or not row['id'] for row in rows)
            or [row['id'] for row in rows]!=identities[key]):raise ValueError('numerical table coverage')
        for row in rows:validate_numerical_row(key,row)


def numerical_union(lease,nodes,observations):
    if len(nodes)!=len(lease['selected']) or [n['node'] for n in nodes]!=lease['selected']:
        raise ValueError('incomplete or reordered numerical union')
    if [n['index'] for n in nodes]!=list(range(len(nodes))):raise ValueError('duplicated numerical nodes')
    for node in nodes:
        validate_numerical_tables(node['node'],node['records'])
        validate_contradictions(node['node'],node['records'],node['contradictions'],lease,observations)
        if node['status']!=('CONTRADICTION' if node['contradictions'] else 'PASSED'):raise ValueError('node contradiction disposition')
    contradictions=[dict(node=n['node'],evidence=c) for n in nodes for c in n['contradictions']]
    terminal=('NOT_ADJUDICATED_SMOKE_ONLY' if lease['lane']=='smoke' else
        'NO_GO_G3C_Q4_AFFINE_RECOVERY_VARIATIONAL_OR_STATE' if contradictions else
        'PROVISIONAL_GO_G3C_Q4_AFFINE_LOCAL_PHYSICAL_RECOVERY_ONLY')
    return dict(schema='GE_BEAM3_REGISTERED_NUMERICAL_UNION_V1',gate=lease['gate'],lane=lease['lane'],
        candidate=lease['candidate'],inputs_sha256=sha256(canonical(lease['inputs'])).hexdigest(),
        observation_manifest_sha256=lease['observation_manifest_sha256'],
        selected=lease['selected'],nodes=nodes,terminal=terminal,contradictions=contradictions,
        physical_recovery_qualified=lease['lane']=='core' and not contradictions,
        full_g3c_qualified=False,production_qualified=False)


def monitor_batch(entries,watchdog,clock=time.monotonic,sleep=time.sleep):
    """Independent tree accounting prevents a busy sibling masking inactivity."""
    if not 1<=len(entries)<=3:raise ValueError('numerical batch worker count')
    for e in entries:e.update(last=e['start'],seen=None,peak=0,record=None)
    while any(e['record'] is None for e in entries):
        watchdog.check()
        for e in entries:
            if e['record'] is not None:continue
            cpu,active,mem=e['job'].accounting();now=clock();e['peak']=max(e['peak'],mem)
            progress=(cpu,*e['progress']())
            if progress!=e['seen']:e['seen']=progress;e['last']=now
            reason=('wall' if now-e['start']>=585 else 'inactivity' if now-e['last']>=120 else
                    'memory' if mem>MEMORY else None)
            code=e['process'].poll()
            if reason:
                e['record']=dict(status='RESOURCE_BLOCKED',reason=reason,drained=False)
            elif code is not None and active==0:
                e['record']=dict(status='PASSED' if code==0 else 'FAILED',drained=True)
            if e['record'] is not None:
                e['record'].update(returncode=e['process'].poll(),active_processes=e['job'].accounting()[1],
                    peak_tree_bytes=e['peak'],elapsed_seconds=clock()-e['start'])
                if e['record']['active_processes'] and not reason:e['record']['status']='FAILED_TO_DRAIN'
        if any(e['record'] and e['record']['status']!='PASSED' for e in entries):
            # Drain the three independent jobs concurrently, not three successive
            # fifteen-second drains that could exceed a sibling's 600s ceiling.
            drains=[]
            for e in entries:
                if e['job'].accounting()[1]:
                    def drain(entry=e):
                        try:entry['drained']=bool(entry['job'].terminate())
                        except BaseException:entry['drained']=False
                    thread=threading.Thread(target=drain,daemon=True);drains.append((e,thread));thread.start()
            drain_deadline=time.monotonic()+15
            for e,thread in drains:thread.join(max(0,drain_deadline-time.monotonic()))
            for e in entries:
                if e['record'] is None:e['record']=dict(status='ABORTED_PEER_FAILURE')
                e['record'].update(drained=e.get('drained',e['job'].accounting()[1]==0),
                    active_processes=e['job'].accounting()[1],returncode=e['process'].poll(),
                    peak_tree_bytes=e['peak'],elapsed_seconds=clock()-e['start'])
                if e['record']['active_processes']:e['record']['status']='FAILED_TO_DRAIN'
            return
        if any(e['record'] is None for e in entries):sleep(.1)


def execute_numerical(args,watchdog):
    observations={}
    expected=authority(args.review,args.review_sha256,args.gate,observation_capture=observations);watchdog.check()
    out=Path(tempfile.mkdtemp(prefix='anyfesolver-beam-qualification-'))
    print('DIAGNOSTICS '+str(out),flush=True)
    lease=dict(schema=SCOPE,run_id=str(uuid.uuid4()),gate=args.gate,lane=args.lane,candidate=expected[0],
        inputs=expected[1],review_sha256=args.review_sha256,selected=inventory(args.lane,args.gate),
        observation_manifest_sha256=OBSERVATION_MANIFEST_SHA)
    write(out/'lease.json',lease)
    with (out/'review.json').open('xb') as stream:stream.write(expected[2])
    env=dict(os.environ,**{key:'1' for key in THREADS})
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTEST_ADDOPTS='',PYTHONDONTWRITEBYTECODE='1')
    nodes=[];process_records=[];failed=False;start=time.monotonic()
    try:
        for offset in range(0,len(lease['selected']),3):
            entries=[]
            try:
                for index in range(offset,min(offset+3,len(lease['selected']))):
                    watchdog.check();directory=out/('node-%02d'%index);directory.mkdir(exist_ok=False)
                    assignment=numerical_assignment(lease,index);write(directory/'assignment.json',assignment)
                    digest=sha256(read(directory/'assignment.json')).hexdigest()
                    job=job_type()(MEMORY);watchdog.attach(job)
                    entry=dict(index=index,job=job,record=None,directory=directory,assignment_sha=digest,streams=[])
                    entries.append(entry)
                    stdout=(directory/'stdout.log').open('xb');entry['streams'].append(stdout)
                    stderr=(directory/'stderr.log').open('xb');entry['streams'].append(stderr)
                    entry['start']=time.monotonic()
                    entry['process']=job.launch([sys.executable,'-I','-S','-B','-u',str(Path(__file__).resolve()),
                        '--numerical-node',str(out),str(index),digest],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
                    entry['progress']=lambda d=directory:((d/'stdout.log').stat().st_size,(d/'stderr.log').stat().st_size)
                monitor_batch(entries,watchdog)
            finally:
                for e in entries:
                    try:
                        if e['job'].accounting()[1] and e.get('record') is None:e['job'].terminate()
                        active=e['job'].accounting()[1]
                        record=e.get('record') or dict(status='FAILED',drained=active==0)
                        record.update(index=e['index'],node=lease['selected'][e['index']],active_processes=active)
                        if active:record['status']='FAILED_TO_DRAIN'
                        for stream in e['streams']:stream.close()
                        write(e['directory']/'process.json',record);process_records.append(record)
                    finally:
                        e['job'].close();watchdog.detach(e['job'])
            if any(r['status']!='PASSED' or r['active_processes'] for r in process_records):failed=True;break
            for e in entries:
                pending=read(e['directory']/'scientific.pending.json')
                completion=environment.strict(read(e['directory']/'completion.json'))
                nodes.append(validate_numerical_result(pending,completion,lease,e['index'],e['assignment_sha'],observations))
        if not failed:
            if authority(args.review,args.review_sha256,args.gate)!=expected:raise ValueError('numerical union final authority')
            watchdog.check();science=numerical_union(lease,nodes,observations)
            write(out/'scientific.pending.json',science)
    except BaseException:
        failed=True
        raise
    finally:
        write(out/'process.json',dict(status='BLOCKED' if failed else 'PASSED',run_id=lease['run_id'],
            completed_nodes=process_records,launched_nodes=len(process_records),required_nodes=len(lease['selected']),
            elapsed_seconds=time.monotonic()-start,
            terminal='BLOCKED_G3C_Q4_AFFINE_RECOVERY_PROCESS_OR_EVIDENCE' if failed else 'COMPLETE_PROCESS_INVENTORY'))
    if not failed:
        watchdog.check()
        if (out/'scientific.json').exists():raise ValueError('exclusive numerical union')
        os.rename(out/'scientific.pending.json',out/'scientific.json')
    return int(failed)


def execute_guarded(args,watchdog):
    wave_start=time.monotonic()
    expected=authority(args.review,args.review_sha256,args.gate)
    watchdog.check()
    out=Path(tempfile.mkdtemp(prefix='anyfesolver-beam-qualification-'))
    print('DIAGNOSTICS '+str(out),flush=True)
    lease=dict(schema=SCOPE,run_id=str(uuid.uuid4()),gate=args.gate,lane=args.lane,
               candidate=expected[0],inputs=expected[1],review_sha256=args.review_sha256,selected=inventory(args.lane,args.gate))
    write(out/'lease.json',lease)
    with (out/'review.json').open('xb') as stream:stream.write(expected[2])
    job=job_type()(MEMORY);record=dict(status='FAILED');process=None
    try:
        watchdog.attach(job)
        env=dict(os.environ,**{key:'1' for key in THREADS})
        env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTEST_ADDOPTS='',PYTHONDONTWRITEBYTECODE='1')
        with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
            child_start=time.monotonic()
            watchdog.check()
            process=job.launch([sys.executable,'-I','-S','-B','-u',str(Path(__file__).resolve()),'--worker',str(out),
                sha256(read(out/'lease.json')).hexdigest()],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
            record=monitor(job,process,lambda:((out/'stdout.log').stat().st_size,(out/'stderr.log').stat().st_size),start=child_start)
        if record['status']=='PASSED':
            completion=environment.strict(read(out/'completion.json'))
            pending=read(out/'scientific.pending.json');science=environment.strict(pending)
            if (completion!={'selected':lease['selected'],'scientific':fingerprint(pending)}
                or set(science)!=({'schema','gate','lane','candidate','selected','records','full_g3c_qualified','production_qualified'}
                                  | ({'adjudication'} if lease['gate'] in ('q4-audit','q4-affine-exact','g4a-station','g4-completion','g5-completion') else set()))
                or science['schema']!='GE_BEAM3_REGISTERED_GATE_SCIENCE_V2' or science['gate']!=lease['gate']
                or science['lane']!=lease['lane'] or type(science['records']) is not list or not science['records']
                or science['candidate']!=lease['candidate'] or science['selected']!=lease['selected']
                or science['full_g3c_qualified'] or science['production_qualified']):raise ValueError('completion mismatch')
            if lease['gate']=='q4-audit' and canonical(science['adjudication'])!=canonical(q4_adjudication(science['records'],lease['lane'])):
                raise ValueError('Q4 terminal mismatch')
            if lease['gate']=='q4-affine-exact' and canonical(science['adjudication'])!=canonical(affine_adjudication(science['records'],lease['lane'])):
                raise ValueError('affine terminal mismatch')
            if lease['gate']=='g4a-station' and canonical(science['adjudication'])!=canonical(g4a_adjudication(science['records'],lease['lane'])):
                raise ValueError('G4a terminal mismatch')
            if lease['gate']=='g4-completion' and canonical(science['adjudication'])!=canonical(g4_adjudication(science['records'],lease['lane'])):
                raise ValueError('G4 completion terminal mismatch')
            if lease['gate']=='g5-completion' and canonical(science['adjudication'])!=canonical(g5_adjudication(science['records'],lease['lane'])):
                raise ValueError('G5 completion terminal mismatch')
            if authority(args.review,args.review_sha256,args.gate)!=expected:raise ValueError('coordinator final authority')
            if time.monotonic()-wave_start>=1800:raise ValueError('wave deadline')
    except BaseException as exc:
        record.update(status='FAILED',exception=type(exc).__name__)
        raise
    finally:
        try:
            if job.accounting()[1]:
                # monitor already consumed its drain reserve on resource failure.
                if record.get('status')!='RESOURCE_BLOCKED':job.terminate()
            record['active_processes']=job.accounting()[1]
            if record['active_processes']:record['status']='FAILED_TO_DRAIN'
        finally:job.close()
        record.update(scope=SCOPE,run_id=lease['run_id'],returncode=process.poll() if process else None,
                      wave_elapsed_seconds=time.monotonic()-wave_start,
                      files={p.name:fingerprint(p.read_bytes()) for p in out.iterdir() if p.is_file()})
        write(out/'process.json',record)
        print(canonical(record).decode(),flush=True)
    if record['status']=='PASSED':
        watchdog.check()
        if (out/'scientific.json').exists():raise ValueError('exclusive publication')
        os.rename(out/'scientific.pending.json',out/'scientific.json')
    return int(record['status']!='PASSED')

def main():
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):raise ValueError('use -I -S -B')
    if len(sys.argv)==4 and sys.argv[1]=='--worker':return worker(Path(sys.argv[2]),sys.argv[3])
    if len(sys.argv)==4 and sys.argv[1]=='--physical-worker':return physical_worker(Path(sys.argv[2]),sys.argv[3])
    if len(sys.argv)==4 and sys.argv[1]=='--physical-formal-worker':return physical_formal_worker(Path(sys.argv[2]),sys.argv[3])
    if len(sys.argv)==4 and sys.argv[1]=='--physical-proof-checker':return physical_proof_checker_worker(Path(sys.argv[2]),sys.argv[3])
    if len(sys.argv)==5 and sys.argv[1]=='--numerical-node':return numerical_worker(Path(sys.argv[2]),int(sys.argv[3]),sys.argv[4])
    if len(sys.argv)==6 and sys.argv[1]=='--q4-checker':
        return q4_checker(Path(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5])
    if len(sys.argv)==6 and sys.argv[1]=='--q4-affine-checker':
        return q4_checker(Path(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5],'q4-affine-exact')
    if os.name!='nt':raise ValueError('Windows process-tree execution required')
    parser=argparse.ArgumentParser()
    parser.add_argument('--gate',choices=list(TESTS),required=True)
    parser.add_argument('--lane',choices=['smoke','core','local','rehearsal','formal'],required=True)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--review-sha256',required=True)
    parser.add_argument('--prior',type=Path,action='append')
    parser.add_argument('--partition-id',choices=PHYSICAL_PARTITION_IDS)
    parser.add_argument('--finalize-partitions',action='store_true')
    parser.add_argument('--inherit-correction',action='store_true')
    parser.add_argument('--failed-guards-process',type=Path)
    parser.add_argument('--interrupted-guards-root',type=Path)
    parser.add_argument('--guard-segment-id',choices=PHYSICAL_CORRECTION_GUARD_SEGMENT_IDS)
    parser.add_argument('--finalize-guard-segments',action='store_true')
    parser.add_argument('--physical-formal-measurement',action='store_true')
    parser.add_argument('--physical-proof-compressed-cycle',type=int,choices=(1,2))
    return execute(parser.parse_args())

if __name__=='__main__':raise SystemExit(main())
