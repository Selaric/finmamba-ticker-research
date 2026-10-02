import unittest

import torch

from genRelation import calculate_relations, stock_cor_matrix


class StockRelationTests(unittest.TestCase):
    def test_vectorized_matrix_matches_pairwise_correlations(self) -> None:
        torch.manual_seed(17)
        features = torch.randn(50, 6, 4)
        window = features[11:31].permute(1, 2, 0)

        for method in ("spearman", "pcc"):
            relation = stock_cor_matrix(
                features,
                lookback=20,
                day=30,
                method=method,
            )
            pairwise = torch.stack(
                [
                    calculate_relations(
                        window[stock_index],
                        window,
                        n=20,
                        method=method,
                    )
                    for stock_index in range(window.size(0))
                ]
            )

            torch.testing.assert_close(relation, pairwise, rtol=1e-5, atol=1e-5)
            self.assertTrue(torch.isfinite(relation).all())
            self.assertTrue(torch.allclose(relation, relation.T))
            self.assertTrue(torch.allclose(relation.diag(), torch.ones(6)))


if __name__ == "__main__":
    unittest.main()