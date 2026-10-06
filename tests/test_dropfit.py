"""Reproducible synthetic tests; no experimental angles are used as targets."""
from __future__ import annotations
import unittest
import numpy as np
from scipy.ndimage import gaussian_filter
from dropfit.geometry import Baseline,fit_circle,fit_local_parametric,detect_green_baseline
from dropfit.analysis import analyze_record
from dropfit.io import Record,identity,image_hash,reported_angles


def synthetic_drop(theta_deg=60., tilt_deg=0., detached=False, colour_overlay=False):
    height,width=620,1400
    theta=np.radians(theta_deg)
    radius=max(160.,45./(1.-np.cos(theta)))
    base=Baseline((width/2,440.),np.radians(tilt_deg),"manual",True)
    yy,xx=np.indices((height,width))
    uv=base.uv(np.column_stack((xx.ravel(),yy.ravel())))
    u,v=uv.T;vc=-radius*np.cos(theta)
    if detached:
        inside=u*u+(v-115)**2<45**2
    else:
        inside=(u*u+(v-vc)**2<radius**2)&(v>=0)
    gray=np.where(inside.reshape(height,width),20.,250.)
    gray=gaussian_filter(gray,.65)
    rgb=np.repeat(np.rint(gray).astype(np.uint8)[...,None],3,axis=2)
    if colour_overlay:
        # Simulates a rasterised instrument baseline; not an input measurement.
        v2=v.reshape(height,width);u2=u.reshape(height,width)
        line=(np.abs(v2)<.6)&(np.abs(u2)<radius*np.sin(theta))
        rgb[line]=[0,140,0]
    return rgb,base


def as_record(rgb,mean=None):
    return Record(rgb,{"filename":"synthetic-test-1.png","key":"synthetic-test-1.png::p1::i1",
                       **identity("synthetic-test-1.png"),"native_width":rgb.shape[1],"native_height":rgb.shape[0],
                       "image_sha256":image_hash(rgb),"reported_angles":{"mean_deg":mean},
                       "extraction_warnings":[]})


def manual_override(base):
    return {"baseline":base.xy([[-400,0],[400,0]]).tolist(),"baseline_validated":True}


class GeometryTests(unittest.TestCase):
    def test_coordinate_round_trip(self):
        b=Baseline((80,120),np.radians(12))
        points=np.array([[30.,20.],[80.,150.],[200.,250.]])
        np.testing.assert_allclose(b.xy(b.uv(points)),points,atol=1e-10)

    def test_baseline_shift_sign(self):
        b=Baseline((100,100),0)
        p=np.array([[100.,90.]])
        self.assertAlmostEqual(b.shifted(1).uv(p)[0,1],9.)

    def test_exact_circle_including_obtuse_angles(self):
        for theta in (15,30,60,85,110,140,160):
            t=np.radians(theta);phi=np.linspace(-.97*t,.97*t,500)
            points=np.column_stack((120*np.sin(phi),-120*np.cos(t)+120*np.cos(phi)))
            self.assertAlmostEqual(fit_circle(points)["theta_deg"],theta,places=7)

    def test_noisy_circle(self):
        rng=np.random.default_rng(724)
        t=np.radians(75);phi=np.linspace(-.94*t,.94*t,500)
        points=np.column_stack((160*np.sin(phi),-160*np.cos(t)+160*np.cos(phi)))
        points+=rng.normal(0,.15,points.shape)
        self.assertLess(abs(fit_circle(points)["theta_deg"]-75),.1)

    def test_local_tangents_support_obtuse_angles(self):
        for theta in (30,60,110,140):
            t=np.radians(theta);phi=np.linspace(-.98*t,.98*t,900)
            points=np.column_stack((180*np.sin(phi),-180*np.cos(t)+180*np.cos(phi)))
            circle=fit_circle(points)
            local=fit_local_parametric(points,circle,fraction=.3,degree=3)
            self.assertLess(abs(local["mean_deg"]-theta),1.)


class ImageTests(unittest.TestCase):
    def test_raster_circles(self):
        for theta in (15,30,60,85,110,140):
            rgb,base=synthetic_drop(theta)
            r,_=analyze_record(as_record(rgb),manual_override(base),sensitivity=False)
            self.assertIsNotNone(r["fit"],r.get("error"))
            self.assertLess(abs(r["fit"]["circle"]["theta_deg"]-theta),1.)

    def test_tilt(self):
        rgb,base=synthetic_drop(60,7)
        r,_=analyze_record(as_record(rgb),manual_override(base),False)
        self.assertLess(abs(r["fit"]["circle"]["theta_deg"]-60),1.)

    def test_green_annotation_baseline(self):
        rgb,base=synthetic_drop(60,3,colour_overlay=True)
        detected=detect_green_baseline(rgb)
        self.assertLess(abs(detected.angle-base.angle),np.radians(.1))
        self.assertFalse(detected.validated)
        self.assertEqual(detected.source,"report_green_line")

    def test_blank_image_is_na(self):
        rgb=np.full((200,300,3),255,np.uint8)
        r,_=analyze_record(as_record(rgb),{"baseline":[[30,150],[270,150]]},False)
        self.assertEqual(r["status"],"NA")
        self.assertIsNone(r["reportable_theta_deg"])

    def test_unknown_baseline_is_na(self):
        rgb,_=synthetic_drop(60)
        r,_=analyze_record(as_record(rgb),{},False)
        self.assertEqual(r["status"],"NA")

    def test_detached_drop_is_na(self):
        rgb,base=synthetic_drop(detached=True)
        r,_=analyze_record(as_record(rgb),manual_override(base),False)
        self.assertEqual(r["status"],"NA")
        self.assertIn("DETACHED_OR_WRONG_COMPONENT",r["flags"])

    def test_thin_film_is_na(self):
        rgb=np.full((200,500,3),250,np.uint8);rgb[147:151,80:420]=20
        r,_=analyze_record(as_record(rgb),{"baseline":[[30,150],[470,150]]},False)
        self.assertEqual(r["status"],"NA")
        self.assertIn("UNRESOLVED_HEIGHT",r["flags"])

    def test_reported_value_cannot_change_fit(self):
        rgb,base=synthetic_drop(60);override=manual_override(base)
        first,_=analyze_record(as_record(rgb,15),override,False)
        second,_=analyze_record(as_record(rgb,170),override,False)
        self.assertEqual(first["fit"],second["fit"])
        self.assertEqual(first["status"],second["status"])

    def test_override_hash_checked(self):
        rgb,base=synthetic_drop(60);override={**manual_override(base),"image_sha256":"wrong"}
        r,_=analyze_record(as_record(rgb),override,False)
        self.assertIn("STALE_OVERRIDE_IMAGE_HASH",r["flags"])


class MetadataTests(unittest.TestCase):
    def test_filename_is_primary(self):
        obj=identity("酒精-Al-1-theta.pdf")
        self.assertEqual(obj["liquid"],"酒精");self.assertEqual(obj["material"],"Al")
        self.assertEqual(obj["filename_method"],"theta")

    def test_parse_reported_angles_only(self):
        result=reported_angles("固体名称:玻璃 液体名称:水 左接触角:21.39 度 右接触角:22.55 度 平均接触角:21.97 度")
        self.assertEqual(result,{"left_deg":21.39,"right_deg":22.55,"mean_deg":21.97})

    def test_exact_pixel_duplicate_hash(self):
        rgb=np.zeros((10,20,3),np.uint8);other=rgb.copy()
        self.assertEqual(image_hash(rgb),image_hash(other))
        other[2,3,0]=1;self.assertNotEqual(image_hash(rgb),image_hash(other))

if __name__=="__main__":unittest.main(verbosity=2)
