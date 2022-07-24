import unittest

import sys

sys.path.append("../../../")
from Jiezi.Physics import hamilton
from Jiezi.Graph import builder


class Test2(unittest.TestCase):

    def test_hamilton_set(self):
        cnt = builder.CNT(n=4, m=2, Trepeat=5, nonideal=False)
        cnt.construct()
        H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
        H.build_H()
        H.build_S(base_overlap=0.018)
        self.assertEqual(H.get_Hii()[2].get_value()[0][0].real, -0.28)
        self.assertEqual(H.get_Hii()[2].get_value()[0][47].real, -2.97)
