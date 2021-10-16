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


def define_hamiltion(a, b, neighbor_a, neighbor_b, n, m, t_1, t_2, onsite, hopping):
    """
    numbering A atom and B atom: combine (p, q) with its number
    the order is first A then B, thus the number of B is greater than A
    :param a: list, set of atom A, which is like [(),(),(),...]
    :param b: list, set of atom B, which is like [(),(),(),...]
    :param neighbor_a: list, the neighbor of A, [[(),(),()],[(),()],...]
    :param neighbor_b: list, the neighbor of B, [[(),(),()],[(),()],...]
    :param n:
    :param m:
    :param t_1:
    :param t_2:
    :param onsite:
    :param hopping:
    :return: a_number: dict, save the number of A atom, which is like {1:(), 2:(), ...}
             b_number: dict, save the number of B atom, which is like {29:(), 30:(), ...}
             total_neighbor: dict, save the neighbor's number of every atom,which is like {1:[2,3,4],...}
             cell_hamilton: np.array[nn,nn], the hamilton matrix of initial layer's hamilton
             hopping_hamilton: np.array[nn,nn], the hamilton matrix between the initial and second layer
    """
    a_number = {}
    b_number = {}
    total_neighbor = {}
    # numbering a and b, a first, then b. The subscript is from 1!
    for index in range(len(a)):
        a_number[index + 1] = a[index]
    for index in range(len(b)):
        b_number[index + 1 + len(a)] = b[index]
    # numbering the neighbor relation of total atoms
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

    # put the hopping and onsite value in the cell hamilton array
    # the size of the cell hamilton array is nn * nn
    nn = len(neighbor_a) + len(neighbor_b)
    cell_hamilton = np.zeros((nn, nn))
    # onsite value is the diagonal element
    for i in range(nn):
        cell_hamilton[i, i] = onsite
    for row, value in total_neighbor.items():
        for column in value:
            cell_hamilton[row - 1, column - 1] = hopping
            cell_hamilton[column - 1, row - 1] = hopping

    # next layer
    # compute the number and index of the next layer atoms
    a_number_next = {}
    b_number_next = {}
    for number, index in a_number.items():
        a_number_next[number + nn] = (index[0] + t_1, index[1] + t_2)
    for number, index in b_number.items():
        b_number_next[number + nn] = (index[0] + t_1, index[1] + t_2)
    print("a_number:", a_number)
    print("b_number:", b_number)
    print("a_number_next:", a_number_next)
    print("b_number_next:", b_number_next)

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
    print("layer", layertolayer)
    # construct hopping hamilton matrix
    hopping_hamilton = np.zeros((nn, nn))
    # layertolayer list follows the rule that initial(small) layer number first, next(big) layer number then
    for element in layertolayer:
        # this hopping matrix is the right-top one, the left-bottom one is its conjugate
        hopping_hamilton[element[0] - 1, element[1] - nn - 1] = hopping
    return a_number, b_number, total_neighbor, cell_hamilton, hopping_hamilton


def coordinate(a_set, b_set, n, m, circumstance, t_1, t_2, count, T_repeat, acc):
    """

    :param a_set: dict, number and index
    :param b_set: dict, number and index
    :param n:
    :param m:
    :param circumstance:
    :param t_1:
    :param t_2:
    :param count: total number of atoms in each layer
    :param T_repeat: total number of layers
    :param acc: a_cc
    :return: the coordinate of A and B in 3D space
             the coordinate of A and B in 3D space
    """
    # compute the relation of number and index(p,q) in the whole tube which has T_repeat same layers
    coord_a = {}
    coord_b = {}
    planar_a = []
    planar_b = []
    for layer in range(T_repeat):
        for number, index in a_set.items():
            coord_a[number + layer * count] = (index[0] + layer * t_1, index[1] + layer * t_2)
        for number, index in b_set.items():
            coord_b[number + layer * count] = (index[0] + layer * t_1, index[1] + layer * t_2)
    # transfer the index(p, q) to real coordinate [x,y,z]
    for number, index in coord_a.items():
        x = (index[0] + index[1]) * (3 / 2 * acc)
        y = (index[0] - index[1]) * (math.sqrt(3) / 2 * acc)
        # rotate
        theta = math.acos(math.sqrt(3) / 2 * (n + m) / math.sqrt(m ** 2 + n ** 2 + m * n))
        u = x * math.cos(theta) + y * math.sin(theta)
        v = - x * math.sin(theta) + y * math.cos(theta)
        planar_a.append([u, v])
        # roll up
        alpha = u / circumstance * 2 * math.pi
        x_3d = math.cos(alpha - math.pi / 2)
        y_3d = math.sin(alpha - math.pi / 2)
        z_3d = v
        coord_a[number] = [x_3d, y_3d, z_3d]

    for number, index in coord_b.items():
        x = (index[0] + index[1] + 2 / 3) * (3 / 2 * acc)
        y = (index[0] - index[1]) * (math.sqrt(3) / 2 * acc)
        # rotate
        theta = math.acos(math.sqrt(3) / 2 * (n + m) / math.sqrt(m ** 2 + n ** 2 + m * n))
        u = x * math.cos(theta) + y * math.sin(theta)
        v = - x * math.sin(theta) + y * math.cos(theta)
        planar_b.append([u, v])
        # roll up
        alpha = u / circumstance * 2 * math.pi
        x_3d = acc * math.cos(alpha - math.pi / 2)
        y_3d = acc * math.sin(alpha - math.pi / 2)
        z_3d = v
        coord_b[number] = [x_3d, y_3d, z_3d]
    return coord_a, coord_b, planar_a, planar_b


def find_value(dict, value):
    for k, v in dict.items():
        if v == value:
            return True


def get_key(dict, value):
    for k, v in dict.items():
        if v == value:
            return k
