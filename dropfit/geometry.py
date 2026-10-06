"""Geometry in a substrate-aligned (u, v) frame; v points into the liquid.

No software-reported contact angle is accepted by any fitting function.
Lengths are native-image pixels, not millimetres.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Baseline:
    origin: tuple[float, float]
    angle: float
    source: str = "manual"
    validated: bool = False
    residual_px: float | None = None
    span_px: float | None = None

    @classmethod
    def from_points(cls, points, validated: bool = False):
        a, b = np.asarray(points, dtype=float)
        if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a,b]).all():
            raise ValueError("Baseline must contain two finite [x,y] points.")
        if a[0] > b[0]:
            a, b = b, a
        if np.linalg.norm(b-a) < 10 or abs(b[0]-a[0]) < 5:
            raise ValueError("Choose two well separated, approximately horizontal baseline points.")
        angle = float(np.arctan2(b[1]-a[1], b[0]-a[0]))
        if abs(angle) > np.pi/4:
            raise ValueError("Rotate the source image first: baseline tilt exceeds 45 degrees.")
        return cls(tuple((a+b)/2), angle, "manual", validated, span_px=float(np.linalg.norm(b-a)))

    def uv(self, xy):
        p = np.asarray(xy, dtype=float)-np.asarray(self.origin)
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.column_stack((c*p[:,0]+s*p[:,1], s*p[:,0]-c*p[:,1]))

    def xy(self, uv):
        p = np.asarray(uv, dtype=float)
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.column_stack((c*p[:,0]+s*p[:,1], s*p[:,0]-c*p[:,1]))+np.asarray(self.origin)

    def shifted(self, distance_px: float):
        """Positive displacement moves the baseline towards the liquid."""
        c, s = np.cos(self.angle), np.sin(self.angle)
        origin = np.asarray(self.origin) + distance_px*np.array([s, -c])
        return Baseline(tuple(origin), self.angle, self.source, self.validated,
                        self.residual_px, self.span_px)


def detect_green_baseline(rgb: np.ndarray, roi=None) -> Baseline:
    """Recognize the instrument's green annotation, NOT the physical substrate.

    Requires a long, almost horizontal, robustly fitted set of green pixels.
    Failure is explicit: there is no arbitrary mid-image baseline fallback.
    """
    r,g,b = rgb.astype(float).transpose(2,0,1)
    mask = (g-r > 25) & (g-b > 20) & (g > 40)
    if roi is not None:
        x0,y0,x1,y1 = map(int, roi)
        valid = np.zeros(mask.shape, bool); valid[y0:y1,x0:x1] = True
        mask &= valid
    yy,xx = np.where(mask)
    if len(xx) < 35 or np.ptp(xx) < 40:
        raise ValueError("No reliable green baseline. Use the annotate command.")
    xc = float(np.median(xx))
    slope, intercept = np.polyfit(xx-xc, yy, 1)
    fit = least_squares(lambda p:p[0]*(xx-xc)+p[1]-yy,
                        [slope,intercept], loss="soft_l1", f_scale=.6)
    good = np.abs(fit.fun) < 1.5
    if good.sum() < 30 or good.mean() < .45 or np.ptp(xx[good]) < 40:
        raise ValueError("Green pixels do not define a unique baseline; review manually.")
    refined = least_squares(lambda p:p[0]*(xx[good]-xc)+p[1]-yy[good],
                            fit.x, loss="soft_l1", f_scale=.5)
    slope,intercept = refined.x
    if abs(slope) > .4:
        raise ValueError("Detected green line is too steep; review manually.")
    return Baseline((xc,float(intercept)), float(np.arctan(slope)),
                    "report_green_line", False,
                    float(np.sqrt(np.mean(refined.fun**2))), float(np.ptp(xx[good])))


def fit_circle(points: np.ndarray) -> dict:
    """Robust geometric-distance circle fit, followed by a baseline intersection."""
    p = np.asarray(points, float)
    if len(p) < 20:
        raise ValueError("Fewer than 20 usable contour points.")
    u,v = p.T
    a = np.column_stack((2*u,2*v,np.ones(len(p))))
    z = np.linalg.lstsq(a,u*u+v*v,rcond=None)[0]
    radius2 = z[2]+z[0]**2+z[1]**2
    if radius2 <= 0:
        raise ValueError("Degenerate initial circle.")
    fit = least_squares(lambda q:np.hypot(u-q[0],v-q[1])-q[2],
                        [z[0],z[1],np.sqrt(radius2)],
                        bounds=([-np.inf,-np.inf,.5],[np.inf,np.inf,np.inf]),
                        loss="soft_l1",f_scale=1.,max_nfev=3000)
    uc,vc,r = map(float,fit.x)
    if not fit.success or not (-1 < -vc/r < 1):
        raise ValueError("Fitted circle does not intersect the baseline.")
    theta = float(np.degrees(np.arccos(-vc/r)))
    halfwidth = float(np.sqrt(r*r-vc*vc))
    return {"u_center_px":uc,"v_center_px":vc,"radius_px":r,
            "theta_deg":theta,"left_contact_u_px":uc-halfwidth,
            "right_contact_u_px":uc+halfwidth,
            "rmse_px":float(np.sqrt(np.mean(fit.fun**2))),
            "max_abs_residual_px":float(np.max(np.abs(fit.fun))),
            "jacobian_condition":float(np.linalg.cond(fit.jac))}


def fit_local_parametric(points: np.ndarray, circle: dict,
                         fraction: float = .4, degree: int = 2) -> dict:
    """Fit u(q),v(q) separately near each contact; intersect fitted v(q) with v=0.

    The global circle supplies only a smooth angular coordinate q. Both local
    coordinates are free polynomials: local contours are NOT constrained to
    that circle. Polynomial extrapolation/window dependence is audited.
    """
    p = np.asarray(points,float);u,v = p.T
    uc,vc,r = [circle[k] for k in ("u_center_px","v_center_px","radius_px")]
    theta = np.radians(circle["theta_deg"])
    phi = np.arctan2(u-uc,v-vc)
    arc = (theta-np.abs(phi))*r
    window = max(12., fraction*r*theta)
    sides = {}
    for name,sign in (("left",-1),("right",1)):
        use = (sign*(u-uc)>0) & (arc < window) & (arc > -.25*window)
        q, x, y = arc[use]/window, u[use], v[use]
        if len(q) < max(12,4*(degree+1)) or np.ptp(q) < .25 or np.ptp(y) < 5:
            raise ValueError(f"{name} local arc is insufficiently resolved.")
        coeffs = []
        for values in (x,y):
            start = np.polynomial.polynomial.polyfit(q,values,degree)
            fit = least_squares(lambda c:np.polynomial.polynomial.polyval(q,c)-values,
                                start,loss="soft_l1",f_scale=1.,max_nfev=1000)
            if not fit.success:
                raise ValueError("Local polynomial fit failed.")
            coeffs.append(fit.x)
        cu,cv = coeffs
        roots = np.polynomial.polynomial.polyroots(cv)
        valid = [float(z.real) for z in roots if abs(z.imag)<1e-7 and -.5<z.real<.5]
        if not valid:
            raise ValueError(f"{name} local polynomial has no nearby baseline intersection.")
        q0 = min(valid,key=abs)
        du = float(np.polynomial.polynomial.polyval(q0,np.polynomial.polynomial.polyder(cu)))
        dv = float(np.polynomial.polynomial.polyval(q0,np.polynomial.polynomial.polyder(cv)))
        angle = float(np.degrees(np.arctan2(dv,-sign*du)))
        if dv <= 0 or not 0 < angle < 180:
            raise ValueError(f"{name} tangent has a nonphysical orientation.")
        resu = np.polynomial.polynomial.polyval(q,cu)-x
        resv = np.polynomial.polynomial.polyval(q,cv)-y
        sides[name] = {"theta_deg":angle,
                       "contact_u_px":float(np.polynomial.polynomial.polyval(q0,cu)),
                       "q_contact":q0,"q_min_observed":float(q.min()),
                       "q_max_observed":float(q.max()),"window_px":window,
                       "coeff_u":cu.tolist(),"coeff_v":cv.tolist(),
                       "n_points":len(q),"tangent_uv":[du,dv],
                       "fit_rmse_px":float(np.sqrt(np.mean(resu**2+resv**2)))}
    width = sides["right"]["contact_u_px"]-sides["left"]["contact_u_px"]
    if width <= 10:
        raise ValueError("Local contact points have an invalid order or spacing.")
    sides["mean_deg"] = (sides["left"]["theta_deg"]+sides["right"]["theta_deg"])/2
    sides["asymmetry_deg"] = abs(sides["left"]["theta_deg"]-sides["right"]["theta_deg"])
    sides["base_width_px"] = width
    sides["degree"] = degree; sides["fraction"] = fraction
    return sides
