from Distance import Distance
import numpy as np

class KMeans:
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

    def calculateLoss(self):
        """Loss function implementation of Equation 1"""

        total_sum = 0

        for k in self.clusters.keys():

            cluster = np.array(self.clusters[k])
            mean = np.array(self.cluster_centers[k])

            total_sum += np.sum((np.linalg.norm(cluster-mean,axis=1))**2)

        return total_sum

    def findDistance(self, x1, x2):

        return ((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)**.5

    def assignToCluster(self, row):
        self.clusters[row[2]].append(row[:2])


    def run(self):
        """Kmeans algorithm implementation"""

        initial_cluster_means = self.dataset[np.random.choice(self.dataset.shape[0], self.K, replace=False), :]

        for i in range(self.K):
            self.cluster_centers[i] = initial_cluster_means[i].tolist()

        last_mean = np.array(list(self.cluster_centers.values()))
        current_mean = np.zeros((self.K, 2))

        k = 1

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

                cluster = np.array(self.clusters[i])
                mean = np.mean(cluster, axis=0)
                self.cluster_centers[i] = mean

            current_mean = np.array(list(self.cluster_centers.values()))

            print(k)

            k += 1

        return self.cluster_centers, self.clusters, self.calculateLoss()



