# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


"""
construct the cell
1, compute the initial field OAB'B based on the chirality parameter;
2, build the ocean of atoms in 2-D plane;
3, screen the atoms in the OAB'B;
4, define the neighbors of every atom;
5, numbering;
"""


import math


def shape_parameter(n, m, a_cc):
    """
    compute some parameters of the shape of CNT
    :param n: chirality first para
    :param m: chirality first para
    :return: shape_para = (circumference, radius, translation vector T,length of T)
    """
    # compute the circumference and radius of the CNT
    circumference = math.sqrt(3) * a_cc * math.sqrt(n**2 + m**2 + m*n)
    radius = circumference / (2 * math.pi)
    # compute the index of the translation vector T
    # dr is the greatest common divisor of (2m+n) and (2n+m)
    dr = greatest_comm_divisor((2*n+m), (2*m+n))
    T = ((2*m+n)//dr, -(2*n+m)//dr)
    # compute the length of vector T
    T_length = math.sqrt(3) / dr * circumference
    shape_para = (circumference, radius, T, T_length)
    return shape_para


def atom_ocean(n, m, a_cc):
    """
    build the ocean of two type atoms for the following step: screening
    :param n: chirality first para
    :param m: chirality second para
    :return: ocean of atom set a and set b
             a, b = ([(0, 0), (0, 1),...],[(0, 0), (0, 1),...])
    """
    a = []
    b = []
    shape_para = shape_parameter(n, m, a_cc)
    # translation vector is T(t1,t2)
    (t_1, t_2) = shape_para[2]
    # O:(0,0)     A:(n,m)
    # B:(t1,t2) B’:(n+t1,m+t2)
    first_index_min = min(0, n, t_1, n + t_1)
    first_index_max = max(0, n, t_1, n + t_1)
    second_index_min = min(0, m, t_2, m + t_2)
    second_index_max = max(0, m, t_2, m + t_2)
    for first_index in range(first_index_min, first_index_max + 1):
        for second_index in range(second_index_min, second_index_max + 1):
            a.append((first_index, second_index))
            b.append((first_index, second_index))
    return a, b


def screen(set_a, set_b, n, m, t_1, t_2):
    """
    screen the atoms in the ocean. remain the atoms within OAB'B and those which on the bottom line,
    left line. those which on the right are saved in a_right and b_right for the following step.
    :param set_a: A type atom in ocean
    :param set_b: B type atom in ocean
    :param n:
    :param m:
    :param t_1:
    :param t_2:
    :return: a[(),(),...] b[(),(),...] the atoms in OAB'B but not contain the top atoms
             a_right[(),(),...] b_right[(),(),...] right atoms
    """
    a = []
    b = []
    a_right = []
    b_right = []

    for p, q in set_a:
        p_1 = p
        q_1 = q
        p_2 = p-n-t_1
        q_2 = q-m-t_2
        # allow the atom on left, right, bottom. but not on the top
        if 2*(t_1*p_1+t_2*q_1)+t_1*q_1+t_2*p_1>=0 and 2*(n*p_1+m*q_1)+n*q_1+m*p_1>=0 \
                and -2*(n*p_2+m*q_2)-n*q_2-m*p_2>=0 and -2*(t_1*p_2+t_2*q_2)-t_1*q_2-t_2*p_2>0:
            a.append((p, q))
            if -2*(n*p_2+m*q_2)-n*q_2-m*p_2 == 0:
                a_right.append((p, q))
    for p, q in set_b:
        p_1 = 3*p+1
        q_1 = 3*q+1
        p_2 = 3*(p-n-t_1)+1
        q_2 = 3*(q-m-t_2)+1
        # allow the atom on left, right, bottom. but not on the top
        if 2*(t_1*p_1+t_2*q_1)+t_1*q_1+t_2*p_1>=0 and 2*(n*p_1+m*q_1)+n*q_1+m*p_1>=0 \
                and -2*(n*p_2+m*q_2)-n*q_2-m*p_2>=0 and -2*(t_1*p_2+t_2*q_2)-t_1*q_2-t_2*p_2>0:
            b.append((p, q))
            if -2*(n*p_2+m*q_2)-n*q_2-m*p_2 == 0:
                b_right.append((p, q))

    return a, b, a_right, b_right


def neighbor(set_a, set_b, a_right, b_right, n, m):
    """
    receive list a and b from the output of function "screen" as list set_a, set_b
    define the relationship of neighbors of every atom
    delete and replace the identical atom because of the rolling up (on the left and right line)
    :param set_a: A type atoms in OAB'B (the output of function "screen")
    :param set_b: B type atoms in OAB'B (the output of function "screen")
    :param a_right: A type atoms on the right line (the output of function "screen")
    :param b_right: B type atoms on the right line (the output of function "screen")
    :param n:
    :param m:
    :return: set_a, set_b: [(),(),()...]
             list of atom A and B in cell (inter, bottom, left)
             a_link, b_link: [[(),(),()],[(),(),()],[(),()]...]
             list of the neighbors of A and B
    """
    # set_a is like [(0, 1),(0, 2),(1, 0),...]
    a_link = []
    b_link = []
    a_map = {}
    b_map = {}
    for p_a, q_a in set_a:
        # sublink is a list, the biggest length of which is 3, [(,),(,),(,)]
        sublink = []
        # set the rule of a' neighbor
        # a(p,q) -> b(p,q-1)
        # a(p,q) -> b(p,q)
        # a(p,q) -> b(p-1,q)
        rule = [(p_a, q_a-1), (p_a-n, q_a-1-m), (p_a+n, q_a-1+m),
                (p_a, q_a), (p_a-n, q_a-m), (p_a+n, q_a+m),
                (p_a-1, q_a), (p_a-1-n, q_a-m), (p_a-1+n, q_a+m)]
        for sub_rule in rule:
            if set_b.count(sub_rule):
                sublink.append(sub_rule)
        a_link.append(sublink)

    for p_b, q_b in set_b:
        # sublink is a list, the biggest length of which is 3, [(,),(,),(,)]
        sublink = []
        # set the rule of b' neighbor
        # b(p,q) -> a(p+1,q)
        # b(p,q) -> a(p,q)
        # b(p,q) -> a(p,q+1)
        rule = [(p_b + 1, q_b), (p_b + 1 - n, q_b - m), (p_b + 1 + n, q_b + m),
                (p_b, q_b), (p_b - n, q_b - m), (p_b + n, q_b + m),
                (p_b, q_b + 1), (p_b - n, q_b + 1 - m), (p_b + n, q_b + 1 + m)]
        for sub_rule in rule:
            if set_a.count(sub_rule):
                sublink.append(sub_rule)
        b_link.append(sublink)

    # now we should deal with the atoms on the right line
    # A type
    for p, q in a_right:
        # delete A type atom on the right, delete it in A set
        index = set_a.index((p, q))
        set_a.remove((p, q))
        # delete the corresponding neighbor set
        a_link.pop(index)
        # In b_link, replace A type atom on the right with corresponding left one:
        # (p, q) -> (p-n, q-m)
        for sub_b in b_link:
            if sub_b.count((p, q)):
                index = sub_b.index((p, q))
                sub_b[index] = (p-n, q-m)
                # some sub_b may has repetitive (p, q), so transfer the sub_b to set, then transfer it back
                b_link[b_link.index(sub_b)] = list(set(sub_b))
    # B type
    for p, q in b_right:
        # delete B type atom on the right, delete it in B set
        index = set_b.index((p, q))
        set_b.remove((p, q))
        # delete the corresponding neighbor set
        b_link.pop(index)
        # In a_link, replace B type atom on the right with corresponding left one:
        # (p, q) -> (p-n, q-m)
        for sub_a in a_link:
            if sub_a.count((p, q)):
                index = sub_a.index((p, q))
                sub_a[index] = (p-n, q-m)
                # some sub_a may has repetitive (p, q), so transfer the sub_a to set, then transfer it back
                a_link[a_link.index(sub_a)] = list(set(sub_a))
    # combine the atom index set with their neighbors index set
    for count in range(len(set_a)):
        a_map[set_a[count]] = a_link[count]
    for count in range(len(set_b)):
        b_map[set_b[count]] = b_link[count]
    return set_a, set_b, a_link, b_link, a_map, b_map


# Euclidean rolling Division method to compute greatest common divisor of n and m
# use recursive method to finish it
def greatest_comm_divisor(n, m):
    if not m:
        return n
    else:
        return greatest_comm_divisor(m, n % m)
