# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
import matplotlib.pyplot as plt
import numpy as np

# # this .py file will plot the iteration process of phi
# X = np.random.rand(5**2).reshape(5, 5)
# X = np.triu(X)
# X += X.T - np.diag(X.diagonal())
# a = np.random.rand(5).reshape(5, 1)
# b = np.random.rand(5).reshape(5, 1)
# c = b - np.dot(b.T, a)/np.dot(a.T, a) * a
# print(np.dot(np.dot(a.T, X), c))

# file_path = "/home/zjy/Jiezi/Jiezi/Files/normal/process0/layerPhi"
file_path = "/home/zjy/Jiezi/Jiezi/Files/grapheneContact/process0/layerPhi"
with open(file_path, "r") as f:
    lines = f.readlines()
for i in range(len(lines)):
    line = list(map(float, lines[i][1:-2].split(", ")))
    x = np.arange(0, len(line), 1)
    plt.plot(x, line, color=(0, 0.1+i*0.1, 0))
    # if i == len(lines) - 1:
    #     plt.plot(x, line, color='g', linewidth=4)
    # if i == len(lines) - 2:
    #     plt.plot(x, line, color='b', linewidth=2)
    # else:
    #     plt.plot(x, line, color=(1, 0, 0), linewidth=2, alpha=i*0.1)
plt.show()
