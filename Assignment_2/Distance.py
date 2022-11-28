import numpy as np


class Distance:
    @staticmethod
    def calculateCosineDistance(x, y):
        """
        Calculates the cosine distance between the given samples.
        :param x: 1D Numpy Array (Data sample)
        :param y: 1D Numpy Array (Data sample)
        :return: Float
        """
        dot_product = np.dot(x, y)  # Take the dot product of sample x and sample y

        return dot_product / (np.linalg.norm(x) * np.linalg.norm(y))  # Divide the dot product to the multiplication
                                                                      # of their norms

    @staticmethod
    def calculateMinkowskiDistance(x, y, p=2):
        """
        Calculates the minkowski distance between the given samples.
        :param x:   1D Numpy Array (Data sample)
        :param y:   1D Numpy Array  (Data sample)
        :param p:   p value in the formula (Default value=2)
        :return:    Float
        """
        x_minus_y = np.abs(x - y) ** p  # Calculates the absolute difference of each element between the numpy arrays
                                        # x and y and takes the power of p

        summation = np.sum(x_minus_y)   # Sums up the elements of the resulting array

        return summation ** (1 / p)     # Takes the power of 1/p and returns

    @staticmethod
    def calculateMahalanobisDistance(x, y, S_minus_1):
        """
        Calculates the mahalanobis distance between the given samples.
        :param x:   1D Numpy Array (Data sample)
        :param y:   1D Numpy Array (Data sample)
        :param S_minus_1:   2D Numpy Array (Inverse of the covariance matrix)
        :return:    Float
        """
        x_minus_y = x - y   # Element-wise subtraction
        dot1 = np.dot(x_minus_y.T, S_minus_1)   # Takes the dot product of (x-y)^T and S^-1
        dot2 = np.dot(dot1, x_minus_y)          # Takes the overall dot product

        return dot2 ** .5       # Returns the square-root
