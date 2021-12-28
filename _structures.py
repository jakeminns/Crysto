import numpy as np
from _structure_interface import IStruct


class Cubic(IStruct):

    def __init__(self):
        super().__init__(6)

    def addCellParams(self, random=True, a=None):
        if random:
            a = self.randomCellParam()
        self.a = a
        self.b = a
        self.c = a
        self.gamma = 90.0
        self.alpha = 90.0
        self.beta = 90.0

    def addSpaceGroup(self, random=True, sg=None):
        if random:
            sg = np.random.randint(low=194, high=230)
        self.sg = sg

    def cell_bragg(self, h, k, l):
        """cubic bragg equation"""
        a, b, c, alpha, beta, gamma = (
            self.a,
            self.b,
            self.c,
            self.alpha,
            self.beta,
            self.gamma,
        )
        const2 = 1.0 / (a**2.0)
        hkl = np.power(h, 2.0) + np.power(k, 2.0) + np.power(l, 2.0)
        return np.multiply(const2, hkl)


class Hexagonal(IStruct):

    def __init__(self):
        super().__init__(5)

    def addCellParams(self, random=True, a=None, c=None):
        if random:
            a = self.randomCellParam()
            c = self.randomCellParam()

        self.a = a
        self.b = a
        self.c = c

        self.gamma = 120.0
        self.alpha = 90.0
        self.beta = 90.0

    def addSpaceGroup(self, random=True, sg=None):
        if random:
            sg = np.random.randint(low=167, high=194)
        self.sg = sg

    def cell_bragg(self, h, k, l):
        """trigonal bragg equation"""
        a, b, c, alpha, beta, gamma = (
            self.a,
            self.b,
            self.c,
            self.alpha,
            self.beta,
            self.gamma,
        )
        rad = np.pi / 180.0
        const2 = 1.0 / (a**2.0)
        hkl = (4.0 / 3.0) * (np.power(h, 2.0) + np.power(k, 2.0) +
                             np.power(np.multiply(h, k), 2.0)) + (np.multiply(
                                 (a / c)**2.0, np.power(l, 2.0)))
        return np.multiply(const2, hkl)


class Trigonal(IStruct):

    def __init__(self):
        super().__init__(4)

    def addCellParams(self, random=True, a=None, c=None):
        if random:
            a = self.randomCellParam()
            c = self.randomCellParam()

        self.a = a
        self.b = a
        self.c = c

        self.gamma = 120.0
        self.alpha = 90.0
        self.beta = 90.0

    def addSpaceGroup(self, random=True, sg=None):
        if random:
            sg = np.random.randint(low=142, high=167)
        self.sg = sg

    def cell_bragg(self, h, k, l):
        """trigonal bragg equation"""
        a, b, c, alpha, beta, gamma = (
            self.a,
            self.b,
            self.c,
            self.alpha,
            self.beta,
            self.gamma,
        )
        rad = np.pi / 180.0
        const2 = 1.0 / (a**2.0)
        hkl = (4.0 / 3.0) * (np.power(h, 2.0) + np.power(k, 2.0) +
                             np.power(np.multiply(h, k), 2.0)) + (np.multiply(
                                 (a / c)**2.0, np.power(l, 2.0)))
        return np.multiply(const2, hkl)


class Tetragonal(IStruct):

    def __init__(self):
        super().__init__(3)

    def addCellParams(self, random=True, a=None, c=None):
        if random:
            a = self.randomCellParam()
            c = self.randomCellParam()

        self.a = a
        self.b = a
        self.c = c

        self.gamma = 90.0
        self.alpha = 90.0
        self.beta = 90.0

    def addSpaceGroup(self, random=True, sg=None):
        if random:
            sg = np.random.randint(low=74, high=142)
        self.sg = sg

    def cell_bragg(self, h, k, l):
        """tetragonal bragg equation"""
        a, b, c, alpha, beta, gamma = (
            self.a,
            self.b,
            self.c,
            self.alpha,
            self.beta,
            self.gamma,
        )
        const2 = 1.0 / (a**2.0)
        hkl = (np.power(h, 2.0) + np.power(k, 2.0) + (np.multiply(
            (a / c)**2.0, np.power(l, 2.0))))
        return np.multiply(const2, hkl)


class Orthorhombic(IStruct):

    def __init__(self):
        super().__init__(2)

    def addCellParams(self, random=True, a=None, b=None, c=None):
        if random:
            a = self.randomCellParam()
            b = self.randomCellParam()
            c = self.randomCellParam()

        self.a = a
        self.b = b
        self.c = c

        self.gamma = 90.0
        self.alpha = 90.0
        self.beta = 90.0

    def addSpaceGroup(self, random=True, sg=None):
        if random:
            sg = np.random.randint(low=15, high=74)
        self.sg = sg

    def cell_bragg(self, h, k, l):
        """orthorhombic bragg equation"""
        a, b, c, alpha, beta, gamma = (
            self.a,
            self.b,
            self.c,
            self.alpha,
            self.beta,
            self.gamma,
        )
        return (np.divide(np.power(h, 2.0), a**2.0) +
                np.divide(np.power(k, 2.0), b**2.0) +
                np.divide(np.power(l, 2.0), c**2.0))


class Monoclinic(IStruct):

    def __init__(self):
        super().__init__(1)

    def addCellParams(self, random=True, a=None, b=None, c=None, beta=None):
        if random:
            a = self.randomCellParam()
            b = self.randomCellParam()
            c = self.randomCellParam()
            beta = self.randomCellAngle()

        self.a = a
        self.b = b
        self.c = c

        self.gamma = 90.0
        self.alpha = 90.0
        self.beta = beta

    def addSpaceGroup(self, random=True, sg=None):
        if random:
            sg = np.random.randint(low=2, high=15)
        self.sg = sg

    def cell_bragg(self, h, k, l):
        """Monoclinic bragg equation"""
        a, b, c, alpha, beta, gamma = (
            self.a,
            self.b,
            self.c,
            self.alpha,
            self.beta,
            self.gamma,
        )

        rad = np.pi / 180.0
        h_ = np.divide(
            np.power(h, 2.0),
            np.multiply((a**2.0), np.power(np.sin(beta * rad), 2.0)))
        k_ = np.divide(np.power(k, 2.0), b**2.0)
        l_ = np.divide(
            np.power(l, 2.0),
            np.multiply((c**2.0), np.power(np.sin(beta * rad), 2.0)))
        a_ = np.divide(
            (2.0 * np.multiply(h, l)) * np.cos(rad * beta),
            (a * c * (np.power(np.sin(beta * rad), 2.0))),
        )
        return h_ + k_ + l_ - a_


class Triclinic(IStruct):

    def __init__(self):
        super().__init__(0)

    def addCellParams(self,
                      random=True,
                      a=None,
                      b=None,
                      c=None,
                      alpha=None,
                      beta=None,
                      gamma=None):
        if random:
            a = self.randomCellParam()
            b = self.randomCellParam()
            c = self.randomCellParam()
            alpha = self.randomCellAngle()
            beta = self.randomCellAngle()
            gamma = self.randomCellAngle()

        self.a = a
        self.b = b
        self.c = c

        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def addSpaceGroup(self, random=True, sg=None):
        if random:
            sg = np.random.randint(low=0, high=2)
        self.sg = sg

    def cell_bragg(self, h, k, l):
        """Triclinic bragg equation"""
        rad = np.pi / 180.0

        a, b, c, alpha, beta, gamma = (
            self.a,
            self.b,
            self.c,
            self.alpha,
            self.beta,
            self.gamma,
        )

        V = (a * b * c *
             (1 + 2 * np.cos(rad * alpha) * np.cos(rad * beta) *
              np.cos(rad * gamma) - (np.cos(alpha * rad)**2.0) -
              (np.cos(beta * rad)**2.0) - (np.cos(gamma * rad)**2.0))**0.5)

        aS = (1 / V) * b * c * np.sin(rad * alpha)
        cosAS = (np.cos(beta * rad) * np.cos(gamma * rad) - np.cos(
            alpha * rad)) / (np.sin(rad * beta) * np.sin(rad * gamma))

        bS = (1 / V) * a * c * np.sin(rad * beta)
        cosBS = (np.cos(gamma * rad) * np.cos(alpha * rad) - np.cos(
            beta * rad)) / (np.sin(rad * gamma) * np.sin(rad * alpha))

        cS = (1 / V) * a * b * np.sin(rad * gamma)
        cosGS = (np.cos(alpha * rad) * np.cos(beta * rad) - np.cos(
            gamma * rad)) / (np.sin(rad * alpha) * np.sin(rad * beta))

        return (np.multiply(np.power(h, 2.0), aS**2.0) +
                np.multiply(np.power(k, 2.0), bS**2.0) +
                np.multiply(np.power(l, 2.0), cS**2.0) +
                (2.0 * np.multiply(k, l) * bS * cS * cosAS) +
                (2.0 * np.multiply(l, h) * cS * aS * cosBS) +
                (2.0 * np.multiply(l, h) * cS * aS * cosBS) +
                (2.0 * np.multiply(h, k) * aS * bS * cosGS))
