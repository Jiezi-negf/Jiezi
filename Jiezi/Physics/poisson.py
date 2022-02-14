# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  -Laplace(u) = f    in the unit square
            u = u_D  on the boundary

  u_D = 1 + x^2 + 2y^2
    f = -6
"""
import numpy as np
from dolfin import *
import matplotlib.pyplot as plt


def poisson(x_min, x_max, num_interval, u_s, u_d, f_value):
    # Create mesh and define function space
    mesh = IntervalMesh(num_interval, x_min, x_max)
    V = FunctionSpace(mesh, 'P', 1)
    coor = mesh.coordinates()

    # Define boundary condition
    class SB_left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], x_min) and on_boundary

    class SB_right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], x_max) and on_boundary

    sub_boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    left = SB_left()
    right = SB_right()
    sub_boundaries.set_all(0)
    left.mark(sub_boundaries, 1)
    right.mark(sub_boundaries, 2)

    bc_left = DirichletBC(V, u_s, sub_boundaries, 1)
    bc_right = DirichletBC(V, u_d, sub_boundaries, 2)
    bcs = [bc_left, bc_right]

    # define source term f
    class myfunc(UserExpression):
        def eval(self, values, x):
            for i in range(len(coor[0]) - 1):
                if between(x[0], (coor[0][i], coor[0][i + 1])):
                    values[0] = f_value[i]

    f = myfunc()

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v)) * dx
    L = f * v * dx

    # Compute solution
    u = Function(V)
    solve(a == L, u, bcs)
    # TODO: projection should be done here, only return values on interested points
    # Plot solution and mesh
    return np.array(u.vector())
