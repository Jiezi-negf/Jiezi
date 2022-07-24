import unittest

import sys

sys.path.append("../../../")
import Jiezi.Graph.builder as builder


class Test5(unittest.TestCase):

    def test_get_length(self):

        cnt = builder.CNT(4, 2, 1, a_cc=1.44, nonideal=False)
        cnt.construct()

        result = cnt.get_length()

        self.assertAlmostEqual(result, 11.429645663799029, 3)
