import ast
from pathlib import Path
import unittest


source = ast.parse(Path(__file__).with_name("__init__.py").read_text(encoding="utf-8"))
parts = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "clear_far_extras"]
scope = dict(MIN_SUBJECT_SHARE=0.1)
exec(compile(ast.Module(body=parts, type_ignores=[]), "<extras>", "exec"), scope)


def run(*appearances):
    segs = [{"shot": shot, "area": area} for shot, area, _ in appearances]
    labels = [{"role": role} for _, _, role in appearances]
    scope["clear_far_extras"](segs, labels)
    return [label["role"] for label in labels]


class FarExtrasTests(unittest.TestCase):
    def test_small_people_named_as_a_subject_go_back_to_other(self):
        self.assertEqual(run((0, 215009, 2), (0, 193548, 1), (0, 3989, 2), (0, 2143, 2)), [2, 1, 0, 0])

    def test_two_subjects_of_similar_size_both_stay(self):
        self.assertEqual(run((0, 195013, 1), (0, 169211, 2), (0, 181566, 0)), [1, 2, 0])

    def test_each_shot_is_measured_on_its_own(self):
        self.assertEqual(run((0, 200000, 1), (0, 180000, 2), (1, 9000, 1), (1, 8000, 2)), [1, 2, 1, 2])

    def test_a_big_person_left_alone_does_not_clear_small_subjects(self):
        self.assertEqual(run((0, 200000, 0), (0, 9000, 1), (0, 8000, 2)), [0, 1, 2])

    def test_a_shot_with_no_subject_is_unchanged(self):
        self.assertEqual(run((0, 3000, 0), (0, 2000, 0)), [0, 0])


if __name__ == "__main__":
    unittest.main()
