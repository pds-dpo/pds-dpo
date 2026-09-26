import unittest
from hallucination_eval.rating_parser import rating


class RatingTests(unittest.TestCase):
    def test_regular(self):
        for i in range(7):
            self.assertEqual(rating(f"Analysis: text.\nRating: {i}, description"), i)

    def test_explicit_final_overrides_tentative(self):
        self.assertEqual(rating("Rating: 4, if ignoring a false claim\nHowever, it is hallucinated.\n\nFinal Rating: 1, somewhat informative, with hallucination"), 1)

    def test_no_best_score_cherry_picking(self):
        self.assertEqual(rating("Rating: 1, tentative\nFinal Rating: 4, somewhat informative, no hallucination"), 4)

    def test_markdown_final(self):
        self.assertEqual(rating("Rating: 1 in a hypothetical case\n**Final Rating: 4**"), 4)

    def test_conflicting_final(self):
        with self.assertRaises(ValueError):
            rating("Final Rating: 1\nFinal Rating: 4")

    def test_conflicting_without_final(self):
        for content in ("Rating: 4, perhaps rating: 5", "Rating: 1\nRating: 4"):
            with self.assertRaises(ValueError):
                rating(content)

    def test_invalid_final_must_not_fall_back(self):
        for value in ("7", "-1", "4.5", "4/6", "1 or 4", "N/A", "1, or rating: 4"):
            with self.assertRaises(ValueError):
                rating("Rating: 4\nFinal Rating: " + value)

    def test_missing(self):
        with self.assertRaises(ValueError):
            rating("looks good")

    def test_inline_final_is_not_authoritative(self):
        with self.assertRaises(ValueError):
            rating("One could give final rating: 4 hypothetically.\nRating: 1")


if __name__ == "__main__":
    unittest.main()
