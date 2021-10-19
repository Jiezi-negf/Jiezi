# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from . import cell
from . import extend
from . import visual
from mayavi import mlab

""" CNT class """


def nonideal():
    print("system has some non-ideal factors")


class CNT:
    __name__ = "Carbon Nanotube"

    def __init__(self, n: int, m: int, Trepeat: int, a_cc=1.4, *, nonideal=False):
        assert 0 <= m <= n, "Condition 0 <= m <= n does not fill!"
        assert Trepeat >= 0, "Repeatation must be positive!"

        self.__n = n
        self.__m = m
        self.__radius = 0.0
        self.__Trepeat = Trepeat
        self.__a_cc = a_cc
        # the suffix "number" means it contains number
        # the suffix "index" means there is only (p,q) but not number
        # {1: (0, 0), 2: (1, -1),...}
        self.__a_set_cell = {}
        # {29: (1, -2), 30: (1, -1),...}
        self.__b_set_cell = {}

        # [[(4, 1)], [(4, 1), (1, -2), (1, -1)],...]
        self.__a_link_index = []
        self.__b_link_index = []

        # {(0, 0): [(4, 1)], (1, -1): [(4, 1), (1, -2), (1, -1)],...}
        self.__a_link_map_index = {}
        self.__b_link_map_index = {}

        # {1: [48], 2: [48, 29, 30],...}
        self.__total_link_number = {}

        # {1: [x,y,z],2:[]...}
        self.__coord_a = {}
        self.__coord_b = {}
        self.__planar_a = []
        self.__planar_b = []

        # array
        self.__hamilton_cell = np.zeros((2, 2))
        self.__hamilton_hopping = np.zeros((2, 2))
        self.__nonideal = nonideal

    def construct(self):
        shape_para = cell.shape_parameter(self.__n, self.__m, self.__a_cc)
        circumstance = shape_para[0]
        self.__radius = shape_para[1]
        (t_1, t_2) = shape_para[2]
        # both set_a and set_b are [(),(),...]
        set_a, set_b = cell.atom_ocean(self.__n, self.__m, self.__a_cc)
        set_a, set_b, a_right, b_right = cell.screen(set_a, set_b, self.__n, self.__m, t_1, t_2)
        self.__a_set_cell, self.__b_set_cell, self.__a_link_index, self.__b_link_index, self.__a_link_map_index, self.__b_link_map_index = \
            cell.neighbor(set_a, set_b, a_right, b_right, self.__n, self.__m)

        self.__a_set_number, self.__b_set_number, self.__total_link_number, self.__hamilton_cell, \
        self.__hamilton_hopping = extend.define_hamiltion(self.__a_set_cell, self.__b_set_cell, self.__a_link_index, self.__b_link_index,
                                                          self.__n, self.__m, t_1, t_2, self.__Trepeat, 1, 2)

        self.__coord_a, self.__coord_b = extend.coordinate(self.__a_set_number, self.__b_set_number, self.__n, self.__m,
                                                           circumstance, self.__a_cc, self.__radius)
        if self.__nonideal:
            nonideal()

    def data_print(self):
        print("A in first cell:")
        print(self.__a_set_cell)
        print("B in first cell:")
        print(self.__b_set_cell)
        print("A in whole tube:")
        print(self.__a_set_number)
        print("B in whole tube:")
        print(self.__b_set_number)
        print("neighbors of A in one cell are:")
        print(self.__a_link_map_index)
        print("neighbors of B in one cell are:")
        print(self.__b_link_map_index)
        print("total link relation is:")
        print(self.__total_link_number)
        print("coordinate of A in 3D space:")
        print(self.__coord_a)
        print("coordinate of B in 3D space:")
        print(self.__coord_b)
        print(self.__hamilton_cell)
        print(self.__hamilton_hopping)

    def data_plot(self):
        visual.visual(self.__coord_a, self.__coord_b, self.__total_link_number)
        mlab.show()
