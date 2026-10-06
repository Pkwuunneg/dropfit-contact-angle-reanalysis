"""Native-pixel silhouette extraction with explicit annotation exclusion."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, label, binary_fill_holes
from skimage.measure import find_contours
from .geometry import Baseline


class ImageQualityError(ValueError):
    def __init__(self, code, message, diagnostics=None):
        super().__init__(message)
        self.code = code
        self.diagnostics = diagnostics or {}


def extract_contour(rgb: np.ndarray, baseline: Baseline, *, level: float=110,
                    exclude_band: float=6., color_delta: float=60., roi=None) -> dict:
    """Segment a single dark sessile drop, close only its excluded bottom band.

    Color overlays are not inpainted. A binary base closure is used ONLY to
    identify the outer silhouette around bright internal highlights. Every
    fitted point must also be an actual grey-level crossing in the source;
    artificial closure/fill boundaries and the bottom band are excluded.
    """
    height,width = rgb.shape[:2]
    if roi is None:
        roi = [0,0,width,height]
    x0,y0,x1,y1 = map(int,roi)
    if not (0<=x0<x1<=width and 0<=y0<y1<=height):
        raise ImageQualityError("INVALID_ROI","ROI falls outside the native raster.")
    # Maximum channel suppresses highly saturated blue/cyan/magenta annotations.
    # Residual JPEG colour contamination cannot be perfectly undone.
    gray = gaussian_filter(rgb.max(axis=2).astype(float),.55)
    yy,xx = np.indices(gray.shape)
    c,s = np.cos(baseline.angle),np.sin(baseline.angle)
    vgrid = s*(xx-baseline.origin[0])-c*(yy-baseline.origin[1])
    region = (xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)
    mask = (gray<level)&(vgrid>0)&region
    labels,n = label(mask)
    if not n:
        raise ImageQualityError("NO_DROP","No dark liquid component above the baseline.")
    sizes = np.bincount(labels.ravel());sizes[0] = 0
    selected = labels == int(sizes.argmax())
    sy,sx = np.where(selected)
    native_height = float(np.max(vgrid[selected]))
    diagnostics = {"height_observed_px":native_height,"component_area_px2":int(selected.sum())}
    if native_height < 12:
        raise ImageQualityError("UNRESOLVED_HEIGHT",
                                "Droplet height is below 12 native pixels; no reliable finite angle.",diagnostics)
    if np.min(vgrid[selected]) > exclude_band+1:
        raise ImageQualityError("DETACHED_OR_WRONG_COMPONENT",
                                "Selected component does not reach the substrate; possibly suspended drop/needle.",diagnostics)
    if sx.min()<=x0 or sx.max()>=x1-1 or sy.min()<=y0:
        raise ImageQualityError("CLIPPED_DROP","Droplet touches a side/top ROI boundary; reselect ROI.",diagnostics)
    # Close the cap across the substrate only inside an excluded segmentation band.
    closure_top = max(1.,exclude_band-1.)
    closure = region & (vgrid>=0)&(vgrid<=closure_top)&(xx>=sx.min()-8)&(xx<=sx.max()+8)
    filled = binary_fill_holes(selected|closure)
    contours = find_contours(filled.astype(float),.5)
    if not contours:
        raise ImageQualityError("NO_CONTOUR","No outer contour found.",diagnostics)
    def area(q):
        y,x=q.T
        return abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    contour = max(contours,key=area)[:,::-1]
    refined = []
    # A marching-squares point lies on a horizontal or vertical pixel-grid edge.
    # Subpixel interpolation is accepted only for a real source-intensity crossing.
    for x,y in contour:
        if abs(y-round(y)) < 1e-6:
            iy,ix = int(round(y)),int(np.floor(x))
            if ix<0 or ix+1>=width or iy<0 or iy>=height: continue
            a,b = gray[iy,ix],gray[iy,ix+1]
            if (a-level)*(b-level)>0 or abs(b-a)<1e-10: continue
            refined.append([ix+(level-a)/(b-a),float(iy)])
        else:
            ix,iy = int(round(x)),int(np.floor(y))
            if iy<0 or iy+1>=height or ix<0 or ix>=width: continue
            a,b = gray[iy,ix],gray[iy+1,ix]
            if (a-level)*(b-level)>0 or abs(b-a)<1e-10: continue
            refined.append([float(ix),iy+(level-a)/(b-a)])
    points = np.asarray(refined,float)
    if points.size == 0:
        raise ImageQualityError("NO_SOURCE_EDGES","No usable source-intensity crossings.",diagnostics)
    uv = baseline.uv(points)
    above = uv[:,1]>exclude_band
    points,uv = points[above],uv[above]
    if len(points)<20:
        raise ImageQualityError("TOO_FEW_EDGES","Too few edges outside the contact band.",diagnostics)
    ix = np.clip(np.rint(points[:,0]).astype(int),0,width-1)
    iy = np.clip(np.rint(points[:,1]).astype(int),0,height-1)
    delta = rgb.max(axis=2).astype(float)-rgb.min(axis=2).astype(float)
    used = delta[iy,ix]<=color_delta
    diagnostics.update({"height_observed_px":float(np.max(uv[:,1])),
                        "n_candidate_points":len(points),"n_used_points":int(used.sum()),
                        "masked_fraction":float(1-used.mean()),"level":level,
                        "exclude_band_px":exclude_band,"color_delta":color_delta})
    if used.sum()<30:
        raise ImageQualityError("ANNOTATION_OCCLUSION","Fewer than 30 unmasked contour points.",diagnostics)
    return {"points_xy":points,"points_uv":uv,"used":used,"diagnostics":diagnostics}
