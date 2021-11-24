# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
import math

E_list = []


def bose(E, BOSE=1, TEMP=1):
    return 1 / (math.exp(E / BOSE / TEMP) - 1)


def fermi(x):
    return 1 / (1 + math.exp(x))


def heaviside(x):
    if x >= 0:
        return 1
    else:
        return 0