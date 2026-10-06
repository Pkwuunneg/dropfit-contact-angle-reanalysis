"""PDF raster/text extraction and filename-based sample identity."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import hashlib
import io
import re
import numpy as np
from PIL import Image
import fitz

SUPPORTED = {".pdf",".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

@dataclass
class Record:
    rgb: np.ndarray
    meta: dict


def identity(filename: str) -> dict:
    stem = Path(filename).stem
    parts = re.split(r"[-_]+",stem)
    liquid = parts[0] if parts else "unknown"
    material = parts[1] if len(parts)>1 else "unknown"
    method = next((p.lower() for p in parts[2:] if p.lower() in ("tan","theta")),"unspecified")
    acquisition = "-".join(p for p in parts if p.lower() not in ("tan","theta"))
    return {"liquid":liquid,"material":material,"filename_method":method,
            "filename_acquisition_key":acquisition}


def reported_angles(text: str) -> dict:
    result = {}
    for key,label in (("left","左"),("right","右"),("mean","平均")):
        match = re.search(label+r"接触角\s*[:：]\s*([+-]?\d+(?:\.\d+)?)",text)
        result[key+"_deg"] = float(match.group(1)) if match else None
    return result


def image_hash(rgb: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(rgb.shape).encode());h.update(rgb.tobytes())
    return h.hexdigest()


def read_records(path: Path) -> list[Record]:
    """One record per sizeable embedded raster occurrence.

    Embedded image extraction is preferred over page rendering. A rendered
    fallback is explicitly marked and requires manual ROI/baseline review.
    """
    path = Path(path)
    records=[]
    base = {"filename":path.name,**identity(path.name)}
    if path.suffix.lower() != ".pdf":
        with Image.open(path) as im:
            rgb=np.asarray(im.convert("RGB")).copy()
        meta={**base,"page":1,"image_index":1,"extraction":"source_image",
              "key":path.name+"::p1::i1","reported_angles":{},"extraction_warnings":[]}
        meta.update({"native_width":rgb.shape[1],"native_height":rgb.shape[0],"image_sha256":image_hash(rgb)})
        return [Record(rgb,meta)]
    with fitz.open(path) as doc:
        if doc.needs_pass:
            raise ValueError("Password-protected PDF is not supported without decryption.")
        for page_index,page in enumerate(doc,1):
            text=page.get_text("text")
            candidates=[x for x in page.get_image_info(xrefs=True)
                        if x.get("width",0)>=150 and x.get("height",0)>=60]
            if not candidates:
                # Rasterise only when no suitable embedded bitmap is available.
                pix=page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
                rgb=np.asarray(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")).copy()
                meta={**base,"page":page_index,"image_index":1,"key":f"{path.name}::p{page_index}::i1",
                      "extraction":"rendered_page_fallback","reported_angles":reported_angles(text),
                      "page_text":text,"extraction_warnings":["RENDERED_PAGE_MANUAL_REVIEW_REQUIRED"]}
                meta.update({"native_width":rgb.shape[1],"native_height":rgb.shape[0],"image_sha256":image_hash(rgb)})
                records.append(Record(rgb,meta));continue
            for image_index,info in enumerate(candidates,1):
                xref=info.get("xref",0)
                warnings=[]
                if xref:
                    embedded=doc.extract_image(xref)
                    if embedded.get("smask",0):
                        pix=fitz.Pixmap(doc,xref);mask=fitz.Pixmap(doc,embedded["smask"])
                        pix=fitz.Pixmap(pix,mask)
                        rgba=Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")
                        background=Image.new("RGBA",rgba.size,"white");background.alpha_composite(rgba)
                        rgb=np.asarray(background.convert("RGB")).copy()
                    else:
                        rgb=np.asarray(Image.open(io.BytesIO(embedded["image"])).convert("RGB")).copy()
                    extraction="embedded_raster"
                else:
                    pix=page.get_pixmap(matrix=fitz.Matrix(2,2),clip=fitz.Rect(info["bbox"]),alpha=False)
                    rgb=np.asarray(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")).copy()
                    extraction="rendered_image_fallback";warnings.append("RENDERED_IMAGE_MANUAL_REVIEW_REQUIRED")
                a,b,c,d,e,f=info["transform"]
                sx=np.hypot(a,b)/info["width"];sy=np.hypot(c,d)/info["height"]
                aspect=float(sx/sy) if sy else None
                if aspect is None or abs(aspect-1)>.01:
                    warnings.append("PDF_DISPLAY_ASPECT_NOT_ISOTROPIC")
                if abs(a*c+b*d)>1e-4*max(1.,np.hypot(a,b)*np.hypot(c,d)):
                    warnings.append("PDF_DISPLAY_SHEAR")
                reported=reported_angles(text) if len(candidates)==1 else {}
                if len(candidates)>1:
                    warnings.append("MULTIPLE_IMAGES_ANGLE_TEXT_NOT_ASSOCIATED")
                meta={**base,"page":page_index,"image_index":image_index,
                      "key":f"{path.name}::p{page_index}::i{image_index}",
                      "xref":xref,"extraction":extraction,"pdf_bbox":list(info["bbox"]),
                      "display_pixel_scale_ratio_xy":aspect,"reported_angles":reported,
                      "page_text":text,"extraction_warnings":warnings,
                      "native_width":rgb.shape[1],"native_height":rgb.shape[0],"image_sha256":image_hash(rgb)}
                records.append(Record(rgb,meta))
    return records


def input_files(path: Path, output: Path | None=None) -> list[Path]:
    path=Path(path)
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED: raise ValueError("Unsupported input extension.")
        return [path]
    if not path.is_dir(): raise FileNotFoundError(path)
    files=[]
    for p in sorted(path.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED: continue
        if output and p.resolve().is_relative_to(output.resolve()): continue
        files.append(p)
    return files
