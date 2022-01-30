from asyncio.log import logger
from absl.testing import absltest
from absl.testing import parameterized


import pysimpfer.simpfer

import numpy as np


def naive_rmips(Q, P, item_idx, k=5):
    n_users = Q.shape[0]
    hit = set()
    rs =[]
    for i in range(n_users):
        ret = np.dot(P, Q[i])
        rs.append(ret)
        ret = np.argsort(-ret)[:k]
        if item_idx in ret:
            hit.add(i)
    return hit


class pysimpferTest(absltest.TestCase):

    Q = np.random.normal(0, 1.5, size=(1000, 5))
    P = np.random.normal(0, 1.5, size=(15, 5))
    # Q[Q<0] = -Q[Q<0]
    # P[P<0] = -P[P<0]

    def test_init(self):
        model = pysimpfer.simpfer.Simpfer(self.Q, self.P)
        K = 10
        n_items = self.P.shape[0]
        for item_idx in range(n_items):
            A = sorted(model.rmips_v1(item_idx, K))
            B = sorted(naive_rmips(self.Q, self.P, item_idx, K))
            self.assertEqual(A, B)

    def test_equality(self):
        model = pysimpfer.simpfer.Simpfer(self.Q, self.P)
        K = 10
        n_items = self.P.shape[0]
        for item_idx in range(n_items):
            A = sorted(model.rmips_v1(item_idx, K))
            B = sorted(naive_rmips(self.Q, self.P, item_idx, K))
            self.assertEqual(A, B)

    def test_equality2(self):
        model = pysimpfer.simpfer.Simpfer(self.Q, self.P)
        K = 10
        n_items = self.P.shape[0]
        for item_idx in range(10):
            A = sorted(model.rmips_v2(item_idx, K))
            B = sorted(naive_rmips(self.Q, self.P, item_idx, K))
            self.assertEqual(A, B, msg=f"Failed at {item_idx}")


if __name__ == '__main__':
    absltest.main(testLoader=pysimpferTest())
