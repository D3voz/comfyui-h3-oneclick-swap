import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as F


# Load only this node so the tensor tests do not initialize ComfyUI's GPU runtime.
source = ast.parse(Path(__file__).with_name("__init__.py").read_text(encoding="utf-8"))
node = next(item for item in source.body if isinstance(item, ast.ClassDef) and item.name == "H3PaintedEditArea")


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
                 comfy=SimpleNamespace(utils=SimpleNamespace(common_upscale=center_crop)))
exec(compile(ast.Module(body=[node], type_ignores=[]), "<painted-area>", "exec"), namespace)
PaintedArea = namespace["H3PaintedEditArea"]


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


if __name__ == "__main__":
    unittest.main()
