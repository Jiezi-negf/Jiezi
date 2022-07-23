import unittest

import sys

sys.path.append("../../../")
import Jiezi.Graph.builder as builder


class Test4(unittest.TestCase):

    def test_get_mass_desity(self):

        cnt = builder.CNT(4, 2, 1, a_cc=1.44, nonideal=False)
        cnt.construct()

        result = cnt.get_mass_desity()

        self.assertEqual(result, 7.044791638581044)





