import numpy as np
import math
from Distance import Distance

class KMeansPlusPlus:
    def __init__(self, dataset, K=2):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        # each cluster is represented with an integer index
        # self.clusters stores the data points of each cluster in a dictionary
        self.clusters = {i: [] for i in range(K)}
        # self.cluster_centers stores the cluster mean vectors for each cluster in a dictionary
        self.cluster_centers = {i: None for i in range(K)}
        # you are free to add further variables and functions to the class

    def calculateNearestDistance(self, sample):

        min_distance = 99999999

        for i in range(self.K):
            distance = ((sample[0] - self.cluster_centers[i][0]) ** 2 + (sample[1] - self.cluster_centers[i][1]) ** 2) ** .5
            if min_distance > distance:
                min_distance = distance

        return min_distance

    def calculateLoss(self):
        """Loss function implementation of Equation 1"""

        total_sum = 0

        for k in self.clusters.keys():

            if len(self.clusters[k]) != 0:
                cluster = np.array(self.clusters[k])
                mean = np.array(self.cluster_centers[k])
                total_sum += np.sum((np.linalg.norm(cluster - mean, axis=1)) ** 2)

        return total_sum

    def run(self):
        """Kmeans++ algorithm implementation"""

        initial_mean = self.dataset[np.random.choice(self.dataset.shape[0], 1, replace=False), :]

        for i in range(self.K):
            self.cluster_centers[i] = initial_mean[0].tolist()



        return self.cluster_centers, self.clusters, self.calculateLoss()
