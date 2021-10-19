# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ===========================================================================

import numpy as np
from mayavi import mlab


def visual(coordinate_a, coordinate_b, total_link):
    """
    visualize the atoms and the connection among them
    :param coordinate_a: dict, {1:([,,],2;[,,]...}
    :param coordinate_b: dict, {1:([,,],2;[,,]...}
    :param total_link: relationship of neighbors
    :return: points and lines connecting the atom and its neighbors
    """
    x = []
    y = []
    z = []
    points = {}
    # "points" is a dict, which can store the coordinate (x,y,z) of every atom
    # points: {1:[, , ,], 2:[, , ,],...}
    for key, value in coordinate_a.items():
        points[key] = value
    for key, value in coordinate_b.items():
        points[key] = value
    for i in range(len(points)):
        x.append(points[i + 1][0])
        y.append(points[i + 1][1])
        z.append(points[i + 1][2])
    mlab.points3d(x, y, z, resolution=16, scale_factor=0.8)

    for start, neighbor in total_link.items():
        xx = []
        yy = []
        zz = []
        for member in neighbor:
            xx.append(points[start][0])
            xx.append(points[member][0])
            yy.append(points[start][1])
            yy.append(points[member][1])
            zz.append(points[start][2])
            zz.append(points[member][2])
        mlab.plot3d(xx, yy, zz)


# mlab.figure(bgcolor=(1,1,1))
# surf = mlab.surf(z,colormap="cool")
# lut = surf.module_manager.scalar_lut_manager.lut.table.to_array()
# lut[:, -1] = np.linspace(254,255,256)
# surf.module_manager.scalar_lut_manager.lut.table = lut
