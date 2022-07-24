import unittest

import sys

sys.path.append("../../../")
import Jiezi.Graph.builder as builder


class Test1(unittest.TestCase):

    def test_get_coordinate(self):
        # print("")
        # print("1", self._testMethodName)

        cnt = builder.CNT(4, 2, 1, a_cc=1.44, nonideal=False)
        cnt.construct()

        result = cnt.get_coordinate()[2]
        self.assertEqual(result[0], 0.46740477071675973, 3)
        self.assertEqual(result[1], -2.0478341033216036, 3)
        self.assertEqual(result[2], 2.449209785099792, 3)

