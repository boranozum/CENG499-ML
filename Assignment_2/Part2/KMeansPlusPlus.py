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
        """
        For the given data sample, computes the euclidean distance to the cluster centers and return the
        minimum of these distances.
        :param sample: 1D Numpy Array (Data sample)
        :return: Float
        """

        min_distance = 99999999

        # Iterates through every cluster
        for i in range(self.K):

            # There can be unassigned cluster centers since the algorithm is in the process of initializing them
            if self.cluster_centers[i] is not None:

                center = np.array(self.cluster_centers[i])  # Gets the current center and convert to numpy array
                distance = np.sum((
                                              sample - center) ** 2) ** .5  # Gets the squared differences of each feature and takes the sqrt of the summation

                if min_distance > distance:  # Checks if the computed distance is currenly minimum
                    min_distance = distance

        return min_distance

    def calculateLoss(self):
        """Loss function implementation of Equation 1"""

        total_sum = 0  # The value that will be returned

        # Iterate through the clusters
        for k in self.clusters.keys():

            # For each cluster that is not empty
            if len(self.clusters[k]) != 0:
                cluster = np.array(self.clusters[k])  # Converts the array to a numpy array for faster computation
                mean = np.array(
                    self.cluster_centers[k])  # Converts the mean vector of the particular cluster center to numpy array

                # Takes the norm of difference vector element-wise between data sample in the cluster and the mean
                # vector and adds the squared sum to the total sum
                total_sum += np.sum((np.linalg.norm(cluster - mean, axis=1)) ** 2)

        return total_sum

    def findDistance(self, x1, x2):

        """
        Computes the euclidean distance between the data samples.
        :param x1: 1D Numpy array (data sample)
        :param x2: 1D Numpy array (data sample)
        :return: Float
        """

        cluster = np.array(x2)  # Convert the cluster array to numpy array for faster computation
        s = (x1 - cluster) ** 2  # Takes element-wise squared value of the differences

        return s.sum() ** .5  # Returns the square root

    def assignToCluster(self, row):
        """
        Assigns the data sample to a cluster.
        :param row: 1D Numpy array (With last elements as the index of the destination cluster)
        """
        self.clusters[row[-1]].append(
            row[:-1])  # Extracts the last element from the array and appends the data sample to the cluster array

    def run(self):
        """Kmeans++ algorithm implementation"""

        # Randomly chooses a data sample from the dataset as the initial cluster center and assigns it
        initial_mean = self.dataset[np.random.choice(self.dataset.shape[0], 1, replace=False), :]
        self.cluster_centers[0] = initial_mean[0].tolist()

        # For the remaining cluster centers...
        for i in range(1, self.K):
            # For every data sample, calculates the distance to the nearest cluster center
            # Note that this applies to the data samples that are chosen to be cluster center, but the corresponding
            # row will be 0, making the squared probability to be chosen as cluster center 0
            d = np.apply_along_axis(self.calculateNearestDistance, 1, self.dataset)

            d_squared = d ** 2  # Takes the squared of each element
            d_squared /= np.sum(d_squared)  # Divides each element to the squared sum to get a probability value

            # Again randomly chooses the next cluster center with a specified probabilities and assigns them
            next_selected_center = self.dataset[np.random.choice(self.dataset.shape[0], 1, replace=False, p=d_squared)]
            self.cluster_centers[i] = next_selected_center[0].tolist()

        # Remaining part of the code is identical to Kmeans since the algorithm after the initialization is similar with Kmeans#

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
