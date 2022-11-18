import numpy as np


class Distance:
    @staticmethod
    def calculateCosineDistance(x, y):
        dot_product = np.dot(x, y)

        return dot_product / (np.linalg.norm(x) * np.linalg.norm(y))

    @staticmethod
    def calculateMinkowskiDistance(x, y, p=2):
        x_minus_y = np.abs(x - y) ** p
        summation = np.sum(x_minus_y)

        return summation ** (1 / p)

    @staticmethod
    def calculateMahalanobisDistance(x, y, S_minus_1):
        x_minus_y = x - y
        dot1 = np.dot(x_minus_y.T, S_minus_1)
        dot2 = np.dot(dot1, x_minus_y)

        return dot2 ** .5
