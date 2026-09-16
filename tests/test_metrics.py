import unittest

from vlm.eval import char_error_rate, exact_match, normalized_exact_match, normalize_text


class MetricTests(unittest.TestCase):
    def test_exact_match_preserves_case_and_whitespace(self):
        self.assertTrue(exact_match("MAIN ST", "MAIN ST"))
        self.assertFalse(exact_match("main st", "MAIN ST"))
        self.assertFalse(exact_match("MAIN ST ", "MAIN ST"))

    def test_normalization_policies(self):
        self.assertEqual(normalize_text("  MAIN\n ST\t "), "main st")
        self.assertTrue(normalized_exact_match(" MAIN   ST ", "main st"))
        self.assertFalse(normalized_exact_match("MAIN ST", "main st", lowercase=False))
        self.assertFalse(normalized_exact_match("Main St.", "main st"))
        self.assertTrue(normalized_exact_match("Main St.", "main st", strip_punctuation=True))

    def test_character_error_rate_counts_edits_against_target_length(self):
        cases = [
            ("cat", "cat", 0.0),
            ("cut", "cat", 1 / 3),
            ("cart", "cat", 1 / 3),
            ("ct", "cat", 1 / 3),
            ("", "cat", 1.0),
            ("abcdef", "a", 5.0),
            ("cafe", "café", 0.25),
            ("", "", 0.0),
            ("cat", "", 1.0),
        ]
        for prediction, target, expected in cases:
            with self.subTest(prediction=prediction, target=target):
                self.assertAlmostEqual(char_error_rate(prediction, target), expected)


if __name__ == "__main__":
    unittest.main()
