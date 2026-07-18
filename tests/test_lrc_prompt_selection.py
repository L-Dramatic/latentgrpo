import unittest

from research.latent_reasoning_contract_benchmark.prompt_selection import (
    extract_final_answer,
    select_records,
)


class LrcPromptSelectionTest(unittest.TestCase):
    def test_extract_final_answer_normalizes_commas(self):
        self.assertEqual(extract_final_answer("work\n#### 1,250"), "1250")

    def test_selection_is_label_blind_and_deterministic(self):
        rows = [
            {"question": "q1", "answer": "a\n#### 1"},
            {"question": "q2", "answer": "b\n#### 2"},
            {"question": "q3", "answer": "c\n#### 3"},
            {"question": "q4", "answer": "d\n#### 4"},
        ]
        first = select_records(rows, salt="fixed", calibration_count=1, confirmation_count=1)
        mutated = [{**row, "answer": "changed\n#### 9"} for row in rows]
        second = select_records(mutated, salt="fixed", calibration_count=1, confirmation_count=1)
        self.assertEqual(
            [record["dataset_index"] for record in first],
            [record["dataset_index"] for record in second],
        )
        self.assertEqual([record["split"] for record in first], ["calibration", "confirmation"])

    def test_bad_answer_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "delimiter"):
            extract_final_answer("no final marker")


if __name__ == "__main__":
    unittest.main()
