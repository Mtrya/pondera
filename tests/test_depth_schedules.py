import unittest

from pondera.training.schedules import exit_teacher_depths


class ExitTeacherDepthTests(unittest.TestCase):
    """Expected maps are hand-derived from the linear spread definition."""

    def test_final_only_supervises_only_the_last_exit(self):
        self.assertEqual(
            exit_teacher_depths(4, "final_only", 16), [None, None, None, 16]
        )
        self.assertEqual(exit_teacher_depths(1, "final_only", 16), [16])

    def test_uniform_supervises_every_exit_at_the_strongest_depth(self):
        self.assertEqual(exit_teacher_depths(3, "uniform", 16), [16, 16, 16])

    def test_progressive_spreads_linearly_with_round_half_up(self):
        # 1 + 15*k/(K-1): K=6 lands exactly on 1,4,7,10,13,16.
        self.assertEqual(
            exit_teacher_depths(6, "progressive", 16), [1, 4, 7, 10, 13, 16]
        )
        # K=3 midpoint is 1+7.5=8.5, rounding half up gives 9.
        self.assertEqual(exit_teacher_depths(3, "progressive", 16), [1, 9, 16])
        # K=2 has only the endpoints.
        self.assertEqual(exit_teacher_depths(2, "progressive", 16), [1, 16])
        # A single exit degenerates to the strongest target.
        self.assertEqual(exit_teacher_depths(1, "progressive", 16), [16])
        # Shallower teacher: 1 + 7*k/2 -> 1, 4.5->5, 8.
        self.assertEqual(exit_teacher_depths(3, "progressive", 8), [1, 5, 8])

    def test_invalid_arguments_are_rejected(self):
        for args in [(0, "uniform", 16), (2, "uniform", 0), (2, "curriculum", 16)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                exit_teacher_depths(*args)


if __name__ == "__main__":
    unittest.main()
