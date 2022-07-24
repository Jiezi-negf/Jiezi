import unittest

import sys

sys.path.append("../../../")
import Jiezi.Graph.builder as builder
from Jiezi.Visualization.Visualization_Graph import visual


class Test2(unittest.TestCase):

    def test_atom_connection(self):
        cnt = builder.CNT(4, 2, 1, a_cc=1.44, nonideal=False)
        cnt.construct()
        x, y, z, connection = cnt.data_plot()

        self.assertAlmostEqual(connection[0], 
                               ([1.2861840596014625e-16, -0.4674047707167568],
                                [-2.1004979729615982, -2.047834103321604],
                                [0.0, 1.3606721028332185]), 3)

        self.assertAlmostEqual(connection[1], 
                               ([0.46740477071675973, 1.642235444121658, 0.46740477071675973, 
                                -0.4674047707167568, 0.46740477071675973, -3.377858365919409e-16], 
                                [-2.0478341033216036, -1.3096390649664982, -2.0478341033216036, 
                                -2.047834103321604, -2.0478341033216036, -2.1004979729615982], 
                                [2.449209785099792, 2.177075364533149, 2.449209785099792, 
                                1.3606721028332185, 2.449209785099792, 3.8098818879330096]), 3)




