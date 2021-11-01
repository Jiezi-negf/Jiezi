# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import numpy as np
import abc


class vector(abc.ABC):
    @abc.abstractmethod
    def get_size(self):
        pass

    @abc.abstractmethod
    def set_value(self):
        pass

    @abc.abstractmethod
    def get_value(self):
        pass

    @abc.abstractmethod
    def imaginary(self):
        pass

    @abc.abstractmethod
    def real(self):
        pass

    @abc.abstractmethod
    def trans(self):
        pass

    @abc.abstractmethod
    def conjugate(self):
        pass

    @abc.abstractmethod
    def dagger(self):
        pass

    @abc.abstractmethod
    def nega(self):
        pass

    @abc.abstractmethod
    def copy(self):
        pass

    @abc.abstractmethod
    def print(self):
        pass


class matrix(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_size(self):
        pass

    @abc.abstractmethod
    def set_value(self):
        pass

    @abc.abstractmethod
    def get_value(self):
        pass

    @abc.abstractmethod
    def imaginary(self):
        pass

    @abc.abstractmethod
    def real(self):
        pass

    @abc.abstractmethod
    def trans(self):
        pass

    @abc.abstractmethod
    def conjugate(self):
        pass

    @abc.abstractmethod
    def dagger(self):
        pass

    @abc.abstractmethod
    def nega(self):
        pass

    @abc.abstractmethod
    def identity(self):
        pass

    @abc.abstractmethod
    def tre(self):
        pass

    @abc.abstractmethod
    def det(self):
        pass

    @abc.abstractmethod
    def copy(self):
        pass

    @abc.abstractmethod
    def print(self):
        pass

