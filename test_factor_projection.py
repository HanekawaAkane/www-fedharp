import sys
import unittest
from pathlib import Path

import torch


CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

from factor_projection import project_B_to_new_A  # noqa: E402


class FactorProjectionTests(unittest.TestCase):
    def test_exact_transport_for_same_row_space(self):
        torch.manual_seed(7)
        old_a = torch.randn(4, 12)
        old_b = torch.randn(9, 4)
        # A_new differs by an invertible rank-space transform, so an exact B
        # transport exists.
        q, _ = torch.linalg.qr(torch.randn(4, 4))
        new_a = q.T @ old_a
        projected, stats = project_B_to_new_A(old_a, old_b, new_a, ridge=1e-8)
        old_update = old_b @ old_a
        new_update = projected @ new_a
        self.assertLess(torch.linalg.norm(new_update - old_update).item(), 1e-4)
        self.assertGreater(stats["relative_reduction"], 0.99)

    def test_projection_reduces_residual(self):
        torch.manual_seed(11)
        old_a = torch.randn(4, 12)
        old_b = torch.randn(9, 4)
        new_a = torch.randn(4, 12)
        projected, stats = project_B_to_new_A(old_a, old_b, new_a, ridge=1e-5)
        self.assertEqual(tuple(projected.shape), tuple(old_b.shape))
        self.assertLessEqual(stats["residual_after"], stats["residual_before"] + 1e-5)

    def test_invalid_shapes(self):
        with self.assertRaises(ValueError):
            project_B_to_new_A(torch.randn(4, 12), torch.randn(9, 3), torch.randn(4, 12))
        with self.assertRaises(ValueError):
            project_B_to_new_A(torch.randn(4, 12), torch.randn(9, 4), torch.randn(3, 12))


if __name__ == "__main__":
    unittest.main()
