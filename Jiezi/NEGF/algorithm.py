# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


from Jiezi.Physics.common import *
from Jiezi.Graph import builder
from Jiezi.Physics import hamilton


def algorithm(chirality_n, chirality_m, T_repeat, a_cc, onsite, hopping, nonideal, base_overlap):
    cnt = builder.CNT(chirality_n, chirality_m, T_repeat, a_cc=a_cc, onsite=onsite, hopping=hopping,
                      nonideal=nonideal)
    cnt.construct()
    H_cell = cnt.get_hamilton_cell()
    H_hopping = cnt.get_hamilton_hopping()
    nn = H_cell.shape[0]
    hopping_value = cnt.get_hopping_value()
    H = hamilton.hamilton(H_cell, H_hopping, nn, T_repeat)
    H.build_H()
    H.build_S(hopping_value, base_overlap=base_overlap)
    Hii = H.get_Hii()
    Hi1 = H.get_Hi1()
    Sii = H.get_Sii()
# import unittest
# from fractions import Fraction
#
# class TestSum(unittest.TestCase):
#     def test_list_int(self):
#         """
#         Test that it can sum a list of integers
#         """
#         data = [1, 2, 3]
#         result = sum(data)
#         self.assertEqual(result, 6)
#
#     def test_list_fraction(self):
#         """
#         Test that it can sum a list of fractions
#         """
#         data = [Fraction(1, 4), Fraction(1, 4), Fraction(2, 5)]
#         result = sum(data)
#         self.assertEqual(result, 1)
#
# if __name__ == '__main__':
#     unittest.main()
