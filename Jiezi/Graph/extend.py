# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


"""
extend the cell along translation vector to form the whole structure
1, change the index of the cell atoms to form the index of second layer atoms;
2, complete the coupling relation of the first layer and second layer;
3, transform the index based on a_1 and a_2 vector to the real coordinate;
4, roll up the planar structure to 3-D space to form the real CNT
"""
import numpy as np
import math

np.set_printoptions(threshold=np.inf)


def pre_define_hamilton(a, b, neighbor_a, neighbor_b, n, m, t_1, t_2, T_repeat):
    """
    Function1: numbering A atom and B atom: combine (p, q) with its number
    the order is first A then B, thus the number of B is greater than A
    Function2: compute the neighbor of every atom
    Principle: first compute cell, then the whole tube
    :param a: list, set of the index of atom A in one cell, which is like [(),(),(),...]
    :param b: list, set of the index of atom B in one cell, which is like [(),(),(),...]
    :param neighbor_a: list, the neighbor of A, [[(),(),()],[(),()],...]
    :param neighbor_b: list, the neighbor of B, [[(),(),()],[(),()],...]
    :param n:
    :param m:
    :param t_1:
    :param t_2:
    :param T_repeat: number of layers
    :return: a_total_number: dict, save the number of A atom of whole tube, which is like {1:(), 2:(), ...}
             b_total_number: dict, save the number of B atom of whole tube, which is like {29:(), 30:(), ...}
             total_neighbor: dict, save the neighbor's number of atoms within single cell, which is like {1:[2,3,4],...}
             total_link: dict, save the neighbor's number of every atom in whole tube, which is like {1:[2,3,4],...}
             layertolayer: list, adjacent atoms between two layers, [(45, 57),...]
             nn: integer, the size of single hamilton matrix is (nn * nn)
    """
    a_number = {}
    b_number = {}
    total_neighbor = {}
    # numbering a and b, a first, then b. The subscript is from 1!
    for index in range(len(a)):
        a_number[index + 1] = a[index]
    for index in range(len(b)):
        b_number[index + 1 + len(a)] = b[index]

    # numbering the neighbor relation of total atoms in one cell
    # total_neighbor is like {1:[2,3,4],2:[1,5,6],3:...}
    for index in range(len(neighbor_a)):
        sub = neighbor_a[index]
        key = []
        for element in sub:
            key.append(get_key(b_number, element))
        total_neighbor[index + 1] = key
    for index in range(len(neighbor_b)):
        sub = neighbor_b[index]
        key = []
        for element in sub:
            key.append(get_key(a_number, element))
        total_neighbor[len(neighbor_a) + index + 1] = key

    # the size of the cell hamilton array is nn * nn
    nn = len(neighbor_a) + len(neighbor_b)

    # next layer
    # compute the number and index of the next layer atoms
    a_number_next = {}
    b_number_next = {}
    for number, index in a_number.items():
        a_number_next[number + nn] = (index[0] + t_1, index[1] + t_2)
    for number, index in b_number.items():
        b_number_next[number + nn] = (index[0] + t_1, index[1] + t_2)

    # find linked atoms between two layers
    layertolayer = []
    # according to the "rule", use new A to find old B, use new B to find old A
    # new indicates the next layer, old is the initial layer
    for number, index in a_number_next.items():
        p_a, q_a = index
        rule = [(p_a, q_a - 1), (p_a, q_a), (p_a - 1, q_a),
                (p_a + n, q_a - 1 + m), (p_a + n, q_a + m), (p_a - 1 + n, q_a + m),
                (p_a - n, q_a - 1 - m), (p_a - n, q_a - m), (p_a - 1 - n, q_a - m)]
        for sub in rule:
            if find_value(b_number, sub):
                layertolayer.append((get_key(b_number, sub), number))

    for number, index in b_number_next.items():
        p_b, q_b = index
        rule = [(p_b + 1, q_b), (p_b, q_b), (p_b, q_b + 1),
                (p_b + 1 + n, q_b + m), (p_b + n, q_b + m), (p_b + n, q_b + 1 + m),
                (p_b + 1 - n, q_b - m), (p_b - n, q_b - m), (p_b - n, q_b + 1 - m)]
        for sub in rule:
            if find_value(a_number, sub):
                layertolayer.append((get_key(a_number, sub), number))

    # compute the relation of number and index(p,q) in the whole tube which has T_repeat same layers
    a_total_number = {}
    b_total_number = {}
    for layer in range(T_repeat):
        for number, index in a_number.items():
            a_total_number[number + layer * nn] = (index[0] + layer * t_1, index[1] + layer * t_2)
        for number, index in b_number.items():
            b_total_number[number + layer * nn] = (index[0] + layer * t_1, index[1] + layer * t_2)

    # compute the relationship about neighbors after extending
    total_link = {}
    for layer in range(T_repeat):
        # total_neighbor is like: {1: [48], 2: [48, 29, 30],...}
        for number, member in total_neighbor.items():
            temp = []
            for index in member:
                index += nn * layer
                temp.append(index)
            total_link[number + nn * layer] = temp
        # add the atoms between linked layers
        # layer1 to layer2 is layertolayer
        # layertolayer is like: [(42, 57), (56, 57),...]
        if layer == 0:
            continue
        else:
            for num_1, num_2 in layertolayer:
                n_1 = num_1 + (layer - 1) * nn
                n_2 = num_2 + (layer - 1) * nn
                total_link[n_1].append(n_2)
                total_link[n_2].append(n_1)

    return a_total_number, b_total_number, total_neighbor, total_link, layertolayer, nn


def coordinate(coord_a, coord_b, n, m, circumstance, acc, radius):
    """
    convert the index (p, q) of every atom in the whole tube with T_repeat same cells to real 3D space coordinate(x,y,z)
    :param coord_a: dict, number and index
    :param coord_b: dict, number and index
    :param n:
    :param m:
    :param circumstance:
    :param acc: a_cc
    :return: the coordinate of A and B in 3D space
             the coordinate of A and B in 3D space
    """

    # transfer the index(p, q) to real coordinate [x,y,z]
    result_a = {}
    result_b = {}
    for number, index in coord_a.items():
        x = (index[0] + index[1]) * (3 / 2 * acc)
        y = (index[0] - index[1]) * (math.sqrt(3) / 2 * acc)
        # rotate
        theta = math.acos(math.sqrt(3) / 2 * (n + m) / math.sqrt(m ** 2 + n ** 2 + m * n))
        u = x * math.cos(theta) + y * math.sin(theta)
        v = - x * math.sin(theta) + y * math.cos(theta)
        # roll up
        alpha = u / circumstance * 2 * math.pi
        x_3d = math.cos(alpha - math.pi / 2) * radius
        y_3d = math.sin(alpha - math.pi / 2) * radius
        z_3d = v
        result_a[number] = [x_3d, y_3d, z_3d]

    for number, index in coord_b.items():
        x = (index[0] + index[1] + 2 / 3) * (3 / 2 * acc)
        y = (index[0] - index[1]) * (math.sqrt(3) / 2 * acc)
        # rotate
        theta = math.acos(math.sqrt(3) / 2 * (n + m) / math.sqrt(m ** 2 + n ** 2 + m * n))
        u = x * math.cos(theta) + y * math.sin(theta)
        v = - x * math.sin(theta) + y * math.cos(theta)
        # roll up
        alpha = u / circumstance * 2 * math.pi
        x_3d = math.cos(alpha - math.pi / 2) * radius
        y_3d = math.sin(alpha - math.pi / 2) * radius
        z_3d = v
        result_b[number] = [x_3d, y_3d, z_3d]
    return result_a, result_b


def find_value(dict, value):
    """
    if the value exists in a dict, return true
    :param dict: dict
    :param value: value
    :return: True or False
    """
    for k, v in dict.items():
        if v == value:
            return True


def get_key(dict, value):
    """
    give the value, return its key
    :param dict: dict
    :param value: value
    :return: key
    """
    for k, v in dict.items():
        if v == value:
            return k
