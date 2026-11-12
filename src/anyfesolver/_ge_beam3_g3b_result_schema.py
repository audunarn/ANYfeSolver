"""Closed inert JSON result schemas for the five private G3b families.

No solve, recovery, owner construction, or runtime schema inference is allowed.
Numeric fields are finite JSON numbers, never booleans. Semantic replay remains
mandatory after the complete journal passes this structural preflight.
"""
import math


def _number(value, shape=()):
    if shape:
        if type(value) is not list or len(value)!=shape[0]:
            raise ValueError("mixed result array shape")
        for item in value: _number(item,shape[1:])
    elif type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError("mixed result finite numeric value required")


def _fields(value, numeric, constants):
    if type(value) is not dict or set(value)!=set(numeric)|set(constants):
        raise ValueError("mixed result exact field schema")
    for key,shape in numeric.items(): _number(value[key],shape)
    for key,expected in constants.items():
        if not _same(value[key],expected):
            raise ValueError("mixed result provenance/enum mismatch: "+key)


def _same(value, expected):
    if type(value) is not type(expected): return False
    if type(expected) is list:
        return len(value)==len(expected) and all(_same(a,b) for a,b in zip(value,expected))
    return value==expected


def _shell_schema(s3):
    n=3 if s3 else 4
    numeric={key:(n,3) for key in ("bending_resultants","curvature","membrane_resultants",
        "membrane_strain","global_transverse_shear_resultants")}
    numeric.update({key:(n,2) for key in ("transverse_shear_resultants","transverse_shear_strain")})
    numeric.update({key:(n,3,3) for key in ("global_bending_resultant_tensors","global_membrane_resultant_tensors")})
    numeric.update({f"{frame}_{component}_{side}":(n,) for frame in ("local","global")
                    for component in ("xx","xy","xz","yy","yz","zz") for side in ("bot","top")})
    constants=dict(numerical_fields_excluded=True,physical_stress_available=True)
    if s3:
        numeric.update({key:(3,3) for key in ("_recovery_coordinates","_recovery_stress_frame",
            "_reference_surface_recovery_coordinates","external_barycentric_coordinates",
            "frame","physical_station_coordinates")})
        numeric.update(hammer_points=(3,2),physical_weights=(3,),min3_relaxation_phi_squared=())
        constants.update(formulation_id="CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1",
            implementation_id="E4_PL_S3_V2D_RECOVERY_CURRENT_EIGEN_GATE_V1",
            internal_order=[0,1,2],physical_layer_recovery_available=True,qualified_recovery=False,
            recovery_scope="PHYSICAL_GLOBAL_SURFACE_TENSORS",
            relaxation_authority_sha256="0AE9DAA05B63A43D456423BCDC676E7421AB3583F152EE5DB3D0E36FE60A17A0",
            resultant_policy_id="SHELL_VARIATIONAL_RESULTANTS_V1")
    else:
        numeric.update({key:(4,3) for key in ("compatible_curvature","compatible_membrane_strain")})
        numeric["compatible_transverse_shear_strain"]=(4,2)
        numeric.update({key:(4,) for key in ("bending_xx","bending_xy","bending_yy",
            "membrane_xx","membrane_xy","membrane_yy","equivalent_stress","hill_utilization",
            "shear_xz","shear_yz","von_mises")})
        numeric["physical_director"]=(3,)
        constants.update(director_polarity_policy_id="Q4_ELEMENT_OWNED_PHYSICAL_DIRECTOR_INDEPENDENT_OF_D4_NUMBERING_V1",
            director_reversal_transform_id="Q4_EPS_S_KAPPA_SIGN_S_SHEAR_SIGN_P_ABD_CONGRUENCE_V1",
            equivalent_stress_measure="von_mises",
            implementation_id="E4_PL_Q4_HYBRID_STATIONARY_RECOVERY_DIRECTOR_RUIZ_V7",
            membrane_resultant_order=["11","22","12"],numbered_frame_director_sign=1,
            physical_director_authoritative=True,
            recovery_policy_id="Q4_HYBRID_PLANAR_STATIONARY_WARPED_VARYING_FRAME_PHYSICAL_DIRECTOR_RECOVERY_V3",
            recovery_scope="qualified_q4_local_and_global_physical",
            transverse_shear_resultant_order=["13","23"])
    return numeric,constants


def validate_result(problem, result):
    family=type(problem).__name__
    tag={"B2TranslationReferenceProblem":"B2","B3TranslationReferenceProblem":"B3",
         "Q4TranslationReferenceProblem":"Q4","S3TranslationReferenceProblem":"S3",
         "WeightedQ4TranslationReferenceProblem":"Q4_WEIGHTED"}.get(family)
    if tag is None: raise ValueError("mixed result unsupported family")
    beam=tag in ("B2","B3")
    numeric={key:(problem.size,) for key in ("u","support_reactions","tie_forces","residual")}
    numeric.update(multipliers=(problem.J.shape[0],),internal=(24,),energy=())
    constants=dict(policy=f"G3B_M_{tag}_REFERENCE_LINEAR_TRANSLATIONS_ONLY_DEVELOPMENT_V1",
        qualification=False,node_ids=list(problem.ids),
        native_station_policy="REFERENCE_LINEAR_STATIONARY_RECOVERY_NOT_FINITE_FRAMES")
    nested={"native_stations","legacy_recovery" if beam else "shell_recovery"}
    if not beam:
        numeric["reference_normal"]=(3,)
        constants["shell_formulation_id"]=("CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1" if tag=="S3"
                                           else "E4_PL_QUALIFIED_Q4_HYBRID_V2")
    if tag=="Q4_WEIGHTED":
        numeric["trace_position"]=(3,)
        constants["trace_weights"]=["1/4"]*4
    if type(result) is not dict or set(result)!=set(numeric)|set(constants)|nested:
        raise ValueError("mixed result exact field schema")
    _fields({key:value for key,value in result.items() if key not in nested},numeric,constants)
    stations=result["native_stations"]
    if type(stations) is not list or len(stations)!=len(problem.native.operator.cell.stations):
        raise ValueError("mixed result station inventory")
    for station in stations:
        _fields(station,dict(strain=(6,),resultants=(6,)),dict(history=[]))
    if beam:
        _fields(result["legacy_recovery"],{key:() for key in ("axial_stress","bending_stress_y",
            "bending_stress_z","equivalent_stress","shear_stress_y","shear_stress_z",
            "torsional_stress","von_mises")},dict(equivalent_stress_measure="von_mises"))
    else:
        _fields(result["shell_recovery"],*_shell_schema(tag=="S3"))
