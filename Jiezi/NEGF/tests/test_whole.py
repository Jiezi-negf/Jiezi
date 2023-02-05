# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
phi = np.array([-0.9061290952177729, -0.725644237474218, -0.353570888581873,
    -0.03431132325018673, -0.008218083355513177, -0.007777431064805044,
    -0.02484885960325484, -0.31983470082398735, -0.7369388859306175, -0.9612375027163028]) * -1
x_axis = np.arange(1, 11, 1)
plt.plot(x_axis, phi)
plt.show()
second_derivative = []
for i in range(1, 9):
    second_derivative.append((phi[i + 1] + phi[i - 1] - 2 * phi[i]) / (4.17 ** 2))
print(second_derivative)
print(1)