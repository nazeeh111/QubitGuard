"""Independent small shared-shot accounting and admission controls."""
import math
import unittest
import numpy as np

from qubitguard.statistics import paired


class SharedShotContract(unittest.TestCase):
    def test_exact_counts_and_symmetry(self):
        truth = [0, 1, 1, 0, 0, 1, 1, 0]
        first = [0, 1, 0, 0, 0, 1, 1, 0]
        second = [1, 0, 1, 1, 1, 1, 1, 0]
        expected_wins = sum(x == t and y != t for x, y, t in zip(first, second, truth))
        expected_losses = sum(x != t and y == t for x, y, t in zip(first, second, truth))
        discordant = expected_wins + expected_losses
        exact_p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(expected_wins, expected_losses) + 1)) / 2**discordant)
        result = paired(first, second, truth)
        self.assertEqual((result['wins'], result['losses']), (expected_wins, expected_losses))
        self.assertAlmostEqual(result['delta_ler'], (expected_losses - expected_wins) / len(truth))
        self.assertAlmostEqual(result['mcnemar_p'], exact_p)
        reversed_result = paired(second, first, truth)
        self.assertEqual((reversed_result['wins'], reversed_result['losses']), (expected_losses, expected_wins))
        self.assertAlmostEqual(reversed_result['delta_ler'], -result['delta_ler'])
        self.assertEqual(paired(truth, truth, truth)['mcnemar_p'], 1.0)

    def test_seeded_binary_and_bool_equivalence(self):
        def fixture(seed):
            rng = np.random.default_rng(seed)
            return tuple(rng.integers(0, 2, size=31) for _ in range(3))
        first = fixture(31415)
        repeated = fixture(31415)
        self.assertEqual(paired(*first), paired(*repeated))
        self.assertEqual(paired(*first), paired(*(x.astype(bool) for x in first)))

    def test_rejects_broadcast_empty_and_nonbinary(self):
        cases = [([0], [0, 1], [0, 1]), ([0, 1], [0], [0, 1]),
                 ([[0], [1]], [0, 1], [0, 1]), ([], [], []),
                 (0, [0], [0]), ([0, 1], [0, 1], [[0], [1]]),
                 ([0, 2], [0, 1], [0, 1]), ([0, float('nan')], [0, 1], [0, 1]),
                 ([0, float('inf')], [0, 1], [0, 1]), (['0', '1'], [0, 1], [0, 1])]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                paired(*case)
