import unittest

import sys

sys.path.append("../../../")
import Jiezi.Graph.builder as builder


class Test3(unittest.TestCase):

    def test_get_total_neighbor(self):

        cnt = builder.CNT(4, 2, 1, a_cc=1.44, nonideal=False)
        cnt.construct()

        result = cnt.get_total_neighbor()

        self.assertEqual(result[1], [48])
        self.assertEqual(result[2], [30, 48, 29])


