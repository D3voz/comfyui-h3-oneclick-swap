import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F


source = ast.parse(Path(__file__).with_name("__init__.py").read_text(encoding="utf-8"))
names = {"build_scene_prompt", "H3GreyMannequins", "H3SceneSwapPlan"}
parts = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
scope = dict(H3PictureSwapPlan=object, torch=torch, F=F, FPS=24, SHOTS_PLACEHOLDER="{SHOTS}", LIP_SYNC="lip-sync", NO_LIP_SYNC="no singing", SOUND="sound",
             SHOTS_PROMPT="Shots for %s", SETTING_SHOTS=" Use picture setting.",
             numbered_lines=lambda text: [line for line in text.splitlines() if line.strip()],
             ordinal_people=lambda count: ["the main person"], split_person=lambda person: (person, ""), neutral_words=lambda role: role,
             plan_clips=lambda *args: ([(124, 0), (124, 39)], 1024, 576), run_hash=lambda *args: "test",
             RESOLUTIONS={"576p": 576}, plan_json=lambda kind, prompts, clips, overlap, steps, seed: prompts,
             SHARP_BLUR=1, SOFT_BLUR=8, gaussian_blur=lambda x, radius: x,
             comfy=SimpleNamespace(model_management=SimpleNamespace(get_torch_device=lambda: torch.device("cpu"))))
exec(compile(ast.Module(body=parts, type_ignores=[]), "<scene>", "exec"), scope)


class SceneTests(unittest.TestCase):
    def test_motion_guide_removes_surroundings_and_preserves_the_mannequin(self):
        image = torch.rand(2, 12, 16, 3)
        mask = torch.zeros(2, 12, 16)
        mask[:, 2:10, 6:10] = 1
        node = scope["H3GreyMannequins"]()
        original = node.hide(image, mask, mask, 0)[0]
        scene = node.hide(image, mask, mask, 0, neutral_background=True)[0]
        self.assertTrue(torch.equal(original[mask == 0], image[mask == 0]))
        self.assertTrue(torch.equal(scene[mask == 1], original[mask == 1]))
        self.assertTrue(torch.all(scene[mask == 0] == 0.5))

    def test_scene_plan_carries_picture_and_motion_prompts_across_clips(self):
        plan = scope["H3SceneSwapPlan"]().build(None, None, 6, 39, "576p", 6, 0, "the dancer", no_lip_sync=True)
        prompts, width, height, name, shots = plan
        self.assertEqual((width, height, name), (1024, 576, "scene_test"))
        self.assertEqual(len(prompts), 2)
        self.assertNotIn("already-generated", prompts[0])
        self.assertIn("already-generated", prompts[1])
        self.assertIn("Use picture setting", shots)

    def test_scene_prompt_assigns_appearance_and_motion_to_the_correct_sources(self):
        prompt = scope["build_scene_prompt"]("Red top.", "the dancer", 124, 39, False)
        self.assertIn("environment and illumination fill the entire frame", prompt)
        self.assertIn("grey figure supplies body poses", prompt)
        self.assertIn("1.625 seconds continue", prompt)
        self.assertIn("no singing", prompt)
        self.assertNotIn("source lighting", prompt)


if __name__ == "__main__":
    unittest.main()
