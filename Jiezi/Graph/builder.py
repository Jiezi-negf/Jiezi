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


""" CNT class """

def nonideal():
    print("system has some non-ideal factors")


class CNT:
    __name__ = "Carbon Nanotube"

    def __init__(self, n:int, m:int, Trepeat:int, a_cc=1.4, *, nonideal=False):
        assert m >= 0 and n >= m, "Condition 0 <= m =< n does not fill!"
        assert Trepeat >= 0, "Repeatation must be positive!"

        self.__n = n
        self.__m = m
        self.__radius = 0.0
        self.__Trepeat = Trepeat
        self.__a_cc = a_cc
        # the suffix "number" means it contains number
        # the suffix "index" means there is only (p,q) but not number
        # {1: (0, 0), 2: (1, -1),...}
        self.__a_set_number = {}
        # {29: (1, -2), 30: (1, -1),...}
        self.__b_set_number = {}

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
        set_a, set_b, self.__a_link_index, self.__b_link_index, self.__a_link_map_index, self.__b_link_map_index = \
            cell.neighbor(set_a, set_b, a_right, b_right, self.__n, self.__m)
        self.__a_set_number, self.__b_set_number, self.__total_link_number, self.__hamilton_cell, \
        self.__hamilton_hopping = extend.define_hamiltion(set_a, set_b, self.__a_link_index, self.__b_link_index,
                                                          self.__n, self.__m, t_1, t_2, 1, 2)
        count = len(self.__total_link_number)
        self.__coord_a, self.__coord_b, self.__planar_a, self.__planar_b = extend.coordinate(self.__a_set_number,
                                                                                             self.__b_set_number,
                                                                                             self.__n, self.__m,
                                                                                             circumstance, t_1, t_2,
                                                                                             count,
                                                                                             self.__Trepeat,
                                                                                             self.__a_cc)
        if self.__nonideal:
            nonideal()

    def data_print(self):
        print("a:")
        print(self.__a_set_number)
        print("b:")
        print(self.__b_set_number)
        print("neighbors of A are:")
        print(self.__a_link_map_index)
        print("neighbors of B are:")
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
        a_x = []
        a_y = []
        b_x = []
        b_y = []
        for point in self.__planar_a:
            a_x.append(point[0])
            a_y.append(point[1])
        for point in self.__planar_b:
            b_x.append(point[0])
            b_y.append(point[1])
        plt.scatter(a_x, a_y, c="red")
        plt.scatter(b_x, b_y, c="blue")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


