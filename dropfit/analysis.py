"""Quality-gated reanalysis. All sensitivities are deterministic, not confidence intervals."""
from __future__ import annotations
from dataclasses import asdict
import numpy as np
from .geometry import Baseline,detect_green_baseline,fit_circle,fit_local_parametric
from .imaging import extract_contour,ImageQualityError

DEFAULTS={"level":110.,"exclude_band":6.,"color_delta":60.,"local_fraction":.4,
          "local_degree":2,"max_circle_rmse_px":2.,"max_circle_sensitivity_span_deg":5.,
          "max_local_circle_difference_deg":6.,"max_local_asymmetry_deg":8.,
          "max_masked_fraction":.55}


def nominal_fit(rgb,baseline,cfg):
    contour=extract_contour(rgb,baseline,level=cfg["level"],exclude_band=cfg["exclude_band"],
                            color_delta=cfg["color_delta"],roi=cfg.get("roi"))
    points=contour["points_uv"][contour["used"]]
    circle=fit_circle(points)
    local=None;local_error=None
    try:
        local=fit_local_parametric(points,circle,cfg["local_fraction"],cfg["local_degree"])
    except (ValueError,np.linalg.LinAlgError) as exc:
        local_error=str(exc)
    h=contour["diagnostics"]["height_observed_px"]
    # Height/width check shares extrapolated local contact points, not an independent measurement.
    half=float(np.degrees(2*np.arctan2(2*h,local["base_width_px"]))) if local else None
    result={"circle":circle,"local":local,"theta_half_deg":half,
            "local_error":local_error,**contour["diagnostics"]}
    return result,contour


def _scenario_values(fit):
    return {"circle_deg":fit["circle"]["theta_deg"],
            "local_mean_deg":fit["local"]["mean_deg"] if fit["local"] else None,
            "theta_half_deg":fit["theta_half_deg"]}


def sensitivity_analysis(rgb,baseline,cfg,nominal):
    scenarios=[{"name":"nominal","ok":True,**_scenario_values(nominal)}]
    changes=[("baseline_up_1px",{"baseline_shift":1.}),
             ("baseline_down_1px",{"baseline_shift":-1.}),
             ("threshold_minus_25",{"level":max(20.,cfg["level"]-25.)}),
             ("threshold_plus_25",{"level":min(235.,cfg["level"]+25.)}),
             ("exclude_band_minus_2px",{"exclude_band":max(3.,cfg["exclude_band"]-2.)}),
             ("exclude_band_plus_2px",{"exclude_band":cfg["exclude_band"]+2.}),
             ("local_window_minus_0.1",{"local_fraction":max(.2,cfg["local_fraction"]-.1)}),
             ("local_window_plus_0.1",{"local_fraction":min(.7,cfg["local_fraction"]+.1)}),
             ("local_polynomial_degree_alternative",{"local_degree":3 if cfg["local_degree"]==2 else 2})]
    for name,change in changes:
        b=baseline.shifted(change.get("baseline_shift",0.)); settings={**cfg,**change}
        try:
            f,_=nominal_fit(rgb,b,settings)
            scenarios.append({"name":name,"ok":True,"change":change,**_scenario_values(f)})
        except (ValueError,np.linalg.LinAlgError) as exc:
            scenarios.append({"name":name,"ok":False,"change":change,"reason":str(exc)})
    ranges={}
    for key in ("circle_deg","local_mean_deg","theta_half_deg"):
        vals=[s[key] for s in scenarios if s.get(key) is not None and np.isfinite(s[key])]
        ranges[key]={"min":min(vals),"max":max(vals),"span":max(vals)-min(vals),"n":len(vals)} if vals else None
    return {"kind":"one_at_a_time_parameter_sensitivity_not_confidence_interval",
            "scenarios":scenarios,"ranges":ranges}


def analyze_record(record,override=None,sensitivity=True):
    override=override or {};cfg={**DEFAULTS,**override}
    meta=record.meta.copy();rgb=record.rgb
    result={**meta,"status":"NA","reportable_theta_deg":None,"flags":[],"info_flags":[],
            "config":cfg,"fit":None,"sensitivity":None,"baseline":None}
    diagnostics=None
    if override.get("image_sha256") and override["image_sha256"]!=meta["image_sha256"]:
        result["flags"].append("STALE_OVERRIDE_IMAGE_HASH");result["error"]="Re-annotate this changed image."
        return result,diagnostics
    try:
        baseline=(Baseline.from_points(override["baseline"],override.get("baseline_validated",False))
                  if "baseline" in override else detect_green_baseline(rgb,override.get("roi")))
        result["baseline"]=asdict(baseline)
        if baseline.source=="report_green_line": result["info_flags"].append("SHARED_REPORT_BASELINE")
        if not baseline.validated: result["info_flags"].append("BASELINE_NOT_INDEPENDENTLY_VALIDATED")
        fit,contour=nominal_fit(rgb,baseline,cfg)
        result["fit"]=fit
        diagnostics={"baseline":baseline,"contour":contour}
        if sensitivity:
            result["sensitivity"]=sensitivity_analysis(rgb,baseline,cfg,fit)
        warnings=result["flags"]
        if fit["circle"]["rmse_px"]>cfg["max_circle_rmse_px"]: warnings.append("CIRCLE_MODEL_RESIDUAL_HIGH")
        if fit["circle"]["jacobian_condition"]>1e6: warnings.append("CIRCLE_POORLY_IDENTIFIED")
        if fit["masked_fraction"]>cfg["max_masked_fraction"]: warnings.append("SEVERE_ANNOTATION_OCCLUSION")
        if fit["local"] is None:
            warnings.append("LOCAL_FIT_UNRESOLVED")
        else:
            if abs(fit["local"]["mean_deg"]-fit["circle"]["theta_deg"])>cfg["max_local_circle_difference_deg"]:
                warnings.append("LOCAL_GLOBAL_DISAGREEMENT")
            if fit["local"]["asymmetry_deg"]>cfg["max_local_asymmetry_deg"]:
                warnings.append("LOCAL_ASYMMETRY_OR_EDGE_ARTIFACT")
        if result["sensitivity"]:
            sens=result["sensitivity"]
            cr=sens["ranges"]["circle_deg"]
            if cr and cr["span"]>cfg["max_circle_sensitivity_span_deg"]: warnings.append("CIRCLE_PARAMETER_SENSITIVE")
            lr=sens["ranges"]["local_mean_deg"]
            if lr and lr["span"]>10: result["info_flags"].append("LOCAL_DERIVATIVE_PARAMETER_SENSITIVE")
            if any(not s["ok"] for s in sens["scenarios"]): warnings.append("SENSITIVITY_RUN_FAILED")
        else:
            result["info_flags"].append("SENSITIVITY_NOT_RUN")
        warnings.extend(meta.get("extraction_warnings",[]))
        result["status"]="REVIEW" if warnings else "CONDITIONAL"
        if not warnings: result["reportable_theta_deg"]=fit["circle"]["theta_deg"]
        original=meta.get("reported_angles",{}).get("mean_deg")
        result["difference_from_report_deg"]=(fit["circle"]["theta_deg"]-original if original is not None else None)
        # Difference from the instrument reading NEVER changes status or fits.
    except ImageQualityError as exc:
        result["flags"].append(exc.code);result["error"]=str(exc);result["image_diagnostics"]=exc.diagnostics
    except (ValueError,np.linalg.LinAlgError) as exc:
        result["flags"].append("BASELINE_OR_FIT_UNRESOLVED");result["error"]=str(exc)
    return result,diagnostics
