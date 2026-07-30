import unittest

import pandas as pd

from analysis.split_protocols import (
    assert_no_interval_overlap,
    chronological_purged_split,
)


class SplitProtocolTests(unittest.TestCase):
    def setUp(self):
        self.metadata = pd.DataFrame(
            {
                "participant_id": ["p1"] * 31,
                "condition": ["c1"] * 31,
                "source_dataset": ["simulations"] * 31,
                "start_index": list(range(0, 465, 15)),
                "end_index": list(range(30, 495, 15)),
            }
        )

    def test_split_has_no_cross_partition_overlap(self):
        assignment = chronological_purged_split(
            self.metadata, train_fraction=0.6, validation_fraction=0.2, purge_samples=15
        )
        assert_no_interval_overlap(self.metadata, assignment)
        self.assertIn("train", set(assignment))
        self.assertIn("validation", set(assignment))
        self.assertIn("test", set(assignment))

    def test_rejects_invalid_fractions(self):
        with self.assertRaises(ValueError):
            chronological_purged_split(
                self.metadata, train_fraction=0.9, validation_fraction=0.2
            )


if __name__ == "__main__":
    unittest.main()
