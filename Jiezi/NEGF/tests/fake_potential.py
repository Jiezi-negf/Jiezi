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


def fake_potential(z, z_max):
    if z <= z_max / 2:
        y = -1 / (np.exp(-3 * (z - z_max / 2 + 0.1 * z_max)) + 1)
    else:
        y = -2 / (np.exp(3 * (z - z_max / 2 - 0.1 * z_max)) + 1) + 1
    return 0


# x_list = np.arange(0, 20, 0.1)
# y_list = []
# for x in x_list:
#     y_list.append(fake_potential(x, 20))
# plt.plot(x_list, y_list)
# plt.show()
