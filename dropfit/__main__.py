"""Command-line interface. Run: python -m dropfit --help"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

from . import __version__
from .io import read_records,input_files
from .analysis import analyze_record
from .outputs import save_diagnostics,write_outputs,write_json


def load_config(path):
    if not path: return {"defaults":{},"images":{}}
    value=json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value,dict): raise ValueError("Configuration must be a JSON object.")
    return value


def batch(input_path,output,config_path=None,sensitivity=True,progress=print):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    files=input_files(Path(input_path),output)
    if not files: raise ValueError("No PDF or supported image files found.")
    cfg=load_config(config_path);results=[];errors=[];seen={};started=time.perf_counter()
    for file_index,path in enumerate(files,1):
        progress(f"[{file_index}/{len(files)}] {path.name}")
        try:
            records=read_records(path)
        except Exception as exc:
            errors.append({"filename":str(path),"error":str(exc)});progress(f"  ERROR: {exc}");continue
        for record in records:
            images=cfg.get("images",{})
            override={**cfg.get("defaults",{}),**images.get(path.name,{}),**images.get(record.meta["key"],{})}
            result,diagnostics=analyze_record(record,override,sensitivity)
            unique_key=result["image_sha256"]
            if unique_key in seen:
                result["exact_duplicate_of"]=seen[unique_key]
                result["info_flags"].append("EXACT_DUPLICATE_RASTER")
            else: seen[unique_key]=result["key"]
            number=len(results)+1
            folder=f"images/{number:03d}_{unique_key[:10]}"
            result["output_folder"]=folder
            save_diagnostics(record,result,diagnostics,output/folder)
            results.append(result)
            angle=(result.get("fit") or {}).get("circle",{}).get("theta_deg")
            progress(f"  {result['status']}: circle="+(f"{angle:.2f} deg" if angle is not None else "NA"))
    versions={}
    for package in ("PyMuPDF","numpy","scipy","scikit-image","Pillow","matplotlib"):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]="not installed"
    manifest={"program_version":__version__,"python":sys.version,"platform":platform.platform(),
              "utc_time":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.perf_counter()-started,
              "input":str(Path(input_path).resolve()),"n_input_files":len(files),"n_records":len(results),
              "dependencies":versions,"configuration":cfg,"sensitivity_enabled":sensitivity,
              "errors":errors,"units":"native pixels and degrees; no physical-length calibration"}
    write_outputs(results,output,manifest);write_json(output/"errors.json",errors)
    progress(f"Done: {len(results)} records; {len(errors)} input errors. Open {output / 'report.html'}")
    return results,manifest


def annotate(args):
    # Do not import plotting code that selects Agg before the interactive backend.
    import matplotlib.pyplot as plt
    records=read_records(Path(args.source))
    selected=[r for r in records if r.meta["page"]==args.page and r.meta["image_index"]==args.image]
    if not selected: raise ValueError("Requested page/image not found. Indices start at 1.")
    record=selected[0]
    fig,ax=plt.subplots(figsize=(11,8));ax.imshow(record.rgb)
    ax.set_title("Step 1: click two opposite ROI corners. Include the drop and substrate, exclude the needle.")
    print("Click 2 ROI corners in the image window. Close the window to cancel.")
    corners=plt.ginput(2,timeout=0)
    if len(corners)!=2: plt.close(fig);raise ValueError("Annotation cancelled.")
    x0,x1=sorted([p[0] for p in corners]);y0,y1=sorted([p[1] for p in corners])
    roi=[max(0,int(x0)),max(0,int(y0)),min(record.rgb.shape[1],int(x1)+1),min(record.rgb.shape[0],int(y1)+1)]
    ax.set_xlim(roi[0],roi[2]);ax.set_ylim(roi[3],roi[1])
    ax.set_title("Step 2: click two points on the TRUE substrate line (left and right).")
    fig.canvas.draw_idle();points=plt.ginput(2,timeout=0);plt.close(fig)
    if len(points)!=2: raise ValueError("Annotation cancelled.")
    from .geometry import Baseline
    Baseline.from_points(points,True)
    path=Path(args.config);cfg=load_config(path) if path.exists() else {"defaults":{},"images":{}}
    cfg.setdefault("images",{})[record.meta["key"]]={"roi":roi,"baseline":[list(p) for p in points],
                                                    "baseline_validated":True,
                                                    "image_sha256":record.meta["image_sha256"]}
    write_json(path,cfg)
    print(f"Saved {record.meta['key']} to {path}. Rerun batch with --config {path}")


def main():
    parser=argparse.ArgumentParser(description="Re-fit contact angles from native PDF droplet images.")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("batch",help="Batch reanalysis; no cloud service or OCR required")
    p.add_argument("--input",required=True,help="PDF/image file or a directory")
    p.add_argument("--output",required=True,help="Separate result directory")
    p.add_argument("--config",help="Optional JSON overrides")
    p.add_argument("--no-sensitivity",action="store_true",help="Quick check only; not recommended for reporting")
    p=sub.add_parser("annotate",help="Interactively choose ROI and a physical substrate line")
    p.add_argument("source");p.add_argument("--page",type=int,default=1);p.add_argument("--image",type=int,default=1)
    p.add_argument("--config",default="overrides.json")
    args=parser.parse_args()
    try:
        if args.command=="batch": batch(args.input,args.output,args.config,not args.no_sensitivity)
        else: annotate(args)
    except Exception as exc:
        parser.exit(1,f"ERROR: {exc}\n")

if __name__=="__main__": main()
