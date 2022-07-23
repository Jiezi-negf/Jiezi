import unittest

import sys

sys.path.append("../../../")
import Jiezi.Graph.builder as builder


class Test6(unittest.TestCase):

    def test_get_layertolayer(self):

        cnt = builder.CNT(4, 2, 1, a_cc=1.44, nonideal=False)
        cnt.construct()

        result = cnt.get_layertolayer()

        self.assertEqual(result[0], (42, 57))
        self.assertEqual(result[7], (27, 97))



