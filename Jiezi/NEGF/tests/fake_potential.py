# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
import numpy as np
from matplotlib import pyplot as plt
from Jiezi.Physics.common import *
import math


def fake_potential(z, z_max):
    if z <= z_max / 2.0:
        y = -1.0 / (np.exp(-3.0 * (z - z_max / 2.0 + 0.1 * z_max)) + 1.0)
    else:
        y = -2.0 / (np.exp(3.0 * (z - z_max / 2.0 - 0.1 * z_max)) + 1.0) + 1.0
    return y


# p1=3
# p2=10
# x_list = np.arange(0, 10, 0.1)
# y_list1 = []
# y_list2 = []
# for x in x_list:
#     y_list1.append(-1*(math.exp(-x/p1)-1))
#     y_list2.append(-1 * (math.exp(-x / p2) - 1))
# plt.plot(x_list, y_list1, label="small", color="r")
# plt.plot(x_list, y_list2, label="big", color="b")
# plt.legend()
# plt.show()
