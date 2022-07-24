import unittest

import sys

sys.path.append("../../../")
import Jiezi.Graph.builder as builder
from Jiezi.Visualization.Visualization_Graph import visual


class Test1(unittest.TestCase):

    def test_atom_position(self):

        cnt = builder.CNT(4, 2, 1, a_cc=1.44, nonideal=False)
        cnt.construct()
        x, y, z, connection = cnt.data_plot()

        self.assertAlmostEqual(x[0], 1.2861840596014625e-16, 3)
        self.assertAlmostEqual(y[0], -2.1004979729615982, 3)
        self.assertAlmostEqual(z[0], 0.0, 3)

        self.assertAlmostEqual(x[1], 0.46740477071675973, 3)
        self.assertAlmostEqual(y[1], -2.0478341033216036, 3)
        self.assertAlmostEqual(z[1], 2.449209785099792, 3)
