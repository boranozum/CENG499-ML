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
            if self.cluster_centers[i] is not None:

                center = np.array(self.cluster_centers[i])
                distance = np.sum((sample-center)**2)**.5

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

    def findDistance(self, x1, x2):

        # return ((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)**.5

        cluster = np.array(x2)
        s = (x1-cluster)**2

        return s.sum()**.5

    def assignToCluster(self, row):
        self.clusters[row[-1]].append(row[:-1])

    def run(self):
        """Kmeans++ algorithm implementation"""

        initial_mean = self.dataset[np.random.choice(self.dataset.shape[0], 1, replace=False), :]
        self.cluster_centers[0] = initial_mean[0].tolist()

        for i in range(1, self.K):

            d = np.apply_along_axis(self.calculateNearestDistance, 1, self.dataset)
            d_squared = d**2
            d_squared /= np.sum(d_squared)
            next_selected_center = self.dataset[np.random.choice(self.dataset.shape[0], 1, replace=False, p=d_squared)]

            self.cluster_centers[i] = next_selected_center.tolist()

        last_mean = np.array(list(self.cluster_centers.values()))
        current_mean = np.zeros((self.K, self.dataset.shape[1]))

        while not (current_mean == last_mean).all():

            last_mean = np.array(list(self.cluster_centers.values()))

            current_clusters = np.apply_along_axis(self.findDistance, 1, self.dataset, self.cluster_centers[0])
            self.clusters[0] = []

            for i in range(1, self.K):
                temp = np.apply_along_axis(self.findDistance, 1, self.dataset, self.cluster_centers[i])
                current_clusters = np.column_stack((current_clusters, temp))
                self.clusters[i] = []

            minimum_distances = np.argmin(current_clusters, axis=1)

            np.apply_along_axis(self.assignToCluster, 1, np.column_stack((self.dataset, minimum_distances)))

            for i in range(self.K):

                if len(self.clusters[i]) != 0:
                    cluster = np.array(self.clusters[i])
                    mean = np.mean(cluster, axis=0)
                    self.cluster_centers[i] = mean

            current_mean = np.array(list(self.cluster_centers.values()))

        return self.cluster_centers, self.clusters, self.calculateLoss()
