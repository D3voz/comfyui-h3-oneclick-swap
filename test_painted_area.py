import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as F


# Load only the area nodes so the tensor tests do not initialize ComfyUI's GPU runtime.
source = ast.parse(Path(__file__).with_name("__init__.py").read_text(encoding="utf-8"))
names = {"H3PaintedEditArea", "H3AddFitCheck", "H3ObjectPlacementMask", "H3AddMovingArea", "placement_boxes", "box_pixels", "placement_person", "placement_pose"}
nodes = [item for item in source.body if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name in names]


def center_crop(images, width, height, method, crop):
    h, w = images.shape[-2:]
    if w / h > width / height:
        cropped = round(h * width / height)
        left = (w - cropped) // 2
        images = images[..., left:left + cropped]
    else:
        cropped = round(w * height / width)
        top = (h - cropped) // 2
        images = images[..., top:top + cropped, :]
    return F.interpolate(images, size=(height, width), mode=method, align_corners=False)


namespace = dict(torch=torch, np=np, json=json, math=math, F=F,
                 comfy=SimpleNamespace(utils=SimpleNamespace(common_upscale=center_crop), model_management=SimpleNamespace(intermediate_device=lambda: torch.device("cpu"))),
                 unpack_masks=lambda packed: torch.from_numpy(np.unpackbits(packed.numpy(), axis=-1, bitorder="little")).bool(),
                 save_temp_json=lambda prefix, data: data)
exec(compile(ast.Module(body=nodes, type_ignores=[]), "<edit-area>", "exec"), namespace)
PaintedArea = namespace["H3PaintedEditArea"]
FitCheck = namespace["H3AddFitCheck"]
MovingArea = namespace["H3AddMovingArea"]


def rectangle(frame, x, y, width, height):
    runs = []
    for row in range(y, y + height):
        runs.extend((row * 100 + x, width))
    return dict(frame=frame, width=100, height=100, runs=runs)


class PaintedAreaTests(unittest.TestCase):
    def test_translation_scale_missing_track_and_frame_offset(self):
        tracked = torch.zeros(5, 100, 100)
        tracked[0, 40:50, 30:50] = 1
        tracked[1, 20:40, 40:80] = 1
        tracked[3, 40:50, 30:50] = 1
        tracked[4] = tracked[3]
        paint = json.dumps([rectangle(12, 20, 35, 40, 20)])
        result = PaintedArea().build(tracked, paint, 12, 100, 100)[0]
        self.assertTrue(torch.all(result >= tracked))
        self.assertEqual(float(result[2].sum()), 0)
        self.assertGreater(float(result[0, 36:39, 22:28].min()), 0.9)
        self.assertGreater(float(result[1, 12:18, 24:35].min()), 0.9)
        self.assertEqual(float(result[1, :8].sum()), 0)
        self.assertTrue(torch.equal(result[3], result[4]))

    def test_keyframes_interpolate_in_bbox_coordinates(self):
        tracked = torch.zeros(5, 100, 100)
        tracked[:, 40:50, 40:60] = 1
        paint = json.dumps([rectangle(0, 20, 30, 20, 20), rectangle(4, 60, 30, 20, 20)])
        result = PaintedArea().build(tracked, paint, 0, 100, 100)[0]
        self.assertGreater(float(result[0, 32:38, 22:38].min()), 0.9)
        self.assertGreater(float(result[2, 32:38, 42:58].min()), 0.9)
        self.assertGreater(float(result[4, 32:38, 62:78].min()), 0.9)

    def test_empty_anchor_and_no_paint_preserve_tracks(self):
        tracked = torch.zeros(2, 100, 100)
        tracked[1, 40:50, 40:60] = 1
        paint = json.dumps([rectangle(0, 20, 30, 20, 20)])
        self.assertTrue(torch.equal(PaintedArea().build(tracked, paint, 0, 100, 100)[0], tracked))
        self.assertTrue(torch.equal(PaintedArea().build(tracked, "[]", 0, 100, 100)[0], tracked))

    def test_center_crop_aligns_paint_and_tracks(self):
        tracked = torch.zeros(1, 100, 200)
        tracked[0, 40:50, 90:110] = 1
        paint = json.dumps([rectangle(0, 20, 30, 60, 30)])
        result = PaintedArea().build(tracked, paint, 0, 100, 100)[0]
        self.assertEqual(tuple(result.shape), (1, 200, 200))
        self.assertGreater(float(result[0, 65:75, 45:155].min()), 0.9)

    def test_invalid_runs(self):
        tracked = torch.ones(1, 100, 100)
        for runs in [[1], [9999, 2], [-1, 2], [5, 3, 4, 1]]:
            paint = json.dumps([dict(frame=0, width=100, height=100, runs=runs)])
            with self.assertRaises(ValueError):
                PaintedArea().build(tracked, paint, 0, 100, 100)


class FitCheckTests(unittest.TestCase):
    def setUp(self):
        self.masks = torch.zeros(8, 1, 100, 104, dtype=torch.bool)
        self.masks[:, 0, 20:80, 60:70] = True
        self.masks[4:, 0, 40:50, 65:100] = True
        self.box = dict(x=0.3, y=0.1, width=0.18, height=0.8)

    def check(self, box):
        packed = torch.from_numpy(np.packbits(self.masks.numpy(), axis=-1, bitorder="little"))
        placement = box if isinstance(box, list) else [dict(frame=0, **box)]
        result = FitCheck().check(torch.zeros(8, 100, 104, 3), {"packed_masks": packed}, json.dumps(placement))
        return result["ui"]["files"][0]

    def test_suggested_box_fits_an_asymmetric_routine(self):
        summary = self.check(self.box)
        self.assertTrue(summary["outside"])
        self.assertIsNotNone(summary["needed"])
        self.assertEqual(self.check(summary["needed"])["outside"], [])

    def test_normalized_pixel_boxes_keep_their_original_dimensions(self):
        self.assertEqual(namespace["box_pixels"]((5 / 104, 57 / 100, 99 / 104, 15 / 100), 100, 104), (57, 72, 5, 104))

    def test_overflow_beyond_the_video_frame_is_counted(self):
        self.masks[:] = False
        self.masks[:, 0, 20:80, 60:70] = True
        self.masks[:, 0, 40:42, 10:65] = True
        summary = self.check(dict(x=0, y=0.1, width=0.3, height=0.8))
        self.assertEqual(summary["outside"], [dict(start=0, end=8, sides=["left"])])

    def test_suggested_boxes_fit_near_both_frame_edges(self):
        for mirrored in (False, True):
            for x in (0, 0.7):
                with self.subTest(mirrored=mirrored, x=x):
                    summary = self.check(dict(x=x, y=0.1, width=0.3, height=0.8))
                    box = summary["needed"]
                    self.assertIsNotNone(box)
                    self.assertGreaterEqual(box["x"], 0)
                    self.assertLessEqual(box["x"] + box["width"], 1)
                    self.assertEqual(self.check(box)["outside"], [])
            self.masks = self.masks.flip(-1)

    def test_wide_routine_gets_a_shorter_box_that_fits(self):
        self.masks[:] = False
        self.masks[:, 0, 40:60, 5:100] = True
        summary = self.check(self.box)
        self.assertTrue(summary["outside"])
        self.assertLess(summary["needed"]["height"], self.box["height"])
        self.assertEqual(self.check(summary["needed"])["outside"], [])

    def test_moving_boxes_get_one_fixed_box_that_fits(self):
        summary = self.check([dict(frame=0, **self.box), dict(frame=7, x=0.7, y=0.3, width=0.2, height=0.6)])
        self.assertEqual(self.check(summary["needed"])["outside"], [])

    def test_vertical_motion_fits_the_box_height(self):
        self.masks[4:] = False
        self.masks[4:, 0, 0:60, 60:70] = True
        summary = self.check(self.box)
        self.assertEqual(summary["outside"], [])
        self.assertIsNotNone(summary["needed"])
        self.assertEqual(self.check(summary["needed"])["outside"], [])

    def test_moving_area_contains_the_pose_when_arms_leave_the_box(self):
        images = torch.zeros(8, 100, 104, 3)
        packed = torch.from_numpy(np.packbits(self.masks.numpy(), axis=-1, bitorder="little"))
        placement = json.dumps([dict(frame=0, **self.box)])
        mask = MovingArea().build(images, placement, {"packed_masks": packed})[0]
        boxes = namespace["placement_boxes"](placement, 0, 8)
        person, center, floor, tall = namespace["placement_person"]({"packed_masks": packed}, boxes)
        grid = torch.meshgrid(torch.arange(100, dtype=torch.float32), torch.arange(104, dtype=torch.float32), indexing="ij")
        for f, box in enumerate(boxes):
            ghost = namespace["placement_pose"](person[f], namespace["box_pixels"](box, 100, 104), center, floor, tall, grid)
            self.assertTrue(torch.all(mask[f][ghost] == 1))
        self.assertGreater(float(mask[4].sum()), float(mask[0].sum()))
        self.assertTrue(torch.any(mask[4, :, 51:] > 0))

    def test_moving_area_keeps_fixed_boxes_without_a_person_or_during_track_gaps(self):
        images = torch.zeros(8, 100, 104, 3)
        placement = json.dumps([dict(frame=0, **self.box)])
        fixed = namespace["H3ObjectPlacementMask"]().build(images, placement)[0]
        self.assertTrue(torch.equal(MovingArea().build(images, placement, {"packed_masks": None})[0], fixed))
        self.masks[3] = False
        packed = torch.from_numpy(np.packbits(self.masks.numpy(), axis=-1, bitorder="little"))
        mask = MovingArea().build(images, placement, {"packed_masks": packed})[0]
        self.assertTrue(torch.equal(mask[3], fixed[3]))

    def test_moving_preview_shows_the_generation_boundary(self):
        images = torch.zeros(8, 100, 104, 3)
        packed = torch.from_numpy(np.packbits(self.masks.numpy(), axis=-1, bitorder="little"))
        tracks = {"packed_masks": packed}
        placement = json.dumps([dict(frame=0, **self.box)])
        mask = MovingArea().build(images, placement, tracks)[0]
        result = FitCheck().check(images, tracks, placement, moving_area=True)
        summary = result["ui"]["files"][0]
        self.assertTrue(summary["moving"])
        self.assertEqual(summary["outside"], [])
        self.assertIsNone(summary["needed"])
        for f in range(8):
            rows, cols = mask[f].any(1).nonzero().flatten(), mask[f].any(0).nonzero().flatten()
            self.assertTrue(torch.allclose(result["result"][0][f, rows[0], cols[0]], torch.tensor([1.0, 0.85, 0.1])))


if __name__ == "__main__":
    unittest.main()
