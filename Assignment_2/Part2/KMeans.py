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

        total_sum = 0       # The value that will be returned

        # Iterate through the clusters
        for k in self.clusters.keys():

            # For each cluster that is not empty
            if len(self.clusters[k]) != 0:
                cluster = np.array(self.clusters[k])    # Converts the array to a numpy array for faster computation
                mean = np.array(self.cluster_centers[k])    # Converts the mean vector of the particular cluster center to numpy array

                # Takes the norm of difference vector element-wise between data sample in the cluster and the mean
                # vector and adds the squared sum to the total sum
                total_sum += np.sum((np.linalg.norm(cluster-mean, axis=1))**2)

        return total_sum

    def findDistance(self, x1, x2):
        """
        Computes the euclidean distance between the data samples.
        :param x1: 1D Numpy array (data sample)
        :param x2: 1D Numpy array (data sample)
        :return: Float
        """

        cluster = np.array(x2)      # Convert the cluster array to numpy array for faster computation
        s = (x1-cluster)**2         # Takes element-wise squared value of the differences

        return s.sum()**.5      # Returns the square root

    def assignToCluster(self, row):
        """
        Assigns the data sample to a cluster.
        :param row: 1D Numpy array (With last elements as the index of the destination cluster)
        """
        self.clusters[row[-1]].append(row[:-1])     # Extracts the last element from the array and appends the data sample to the cluster array


    def run(self):
        """Kmeans algorithm implementation"""

        # Randomly chooses three different data sample from the dataset as the cluster center
        initial_cluster_means = self.dataset[np.random.choice(self.dataset.shape[0], self.K, replace=False), :]

        # Assigns these centers to cluster_means
        for i in range(self.K):
            self.cluster_centers[i] = initial_cluster_means[i].tolist()

        # Two numpy arrays to keep track of the calculated mean and the previous mean to check for convergence
        # If these values become equal, the loop will terminate
        last_mean = np.array(list(self.cluster_centers.values()))
        current_mean = np.zeros((self.K, self.dataset.shape[1]))

        while not (current_mean == last_mean).all():

            # Get the last mean from previously updated cluster centers
            last_mean = np.array(list(self.cluster_centers.values()))

            # The following code segment clears the clusters and calculates every distance between data samples
            # and the cluster centers and creates a 2D numpy array (rows-> data samples, cols-> distance to cluster means)
            current_clusters = np.apply_along_axis(self.findDistance, 1, self.dataset, self.cluster_centers[0])
            self.clusters[0] = []
            for i in range(1, self.K):
                temp = np.apply_along_axis(self.findDistance, 1, self.dataset, self.cluster_centers[i])
                current_clusters = np.column_stack((current_clusters, temp))
                self.clusters[i] = []

            # Finds the indexes which represents cluster means where the distance is minimum
            minimum_distances = np.argmin(current_clusters, axis=1)

            # Assign each data sample to the cluster which is at the minimum distance
            np.apply_along_axis(self.assignToCluster, 1, np.column_stack((self.dataset, minimum_distances)))

            # This for loop calculates the cluster centers with the updated clusters
            for i in range(self.K):

                if len(self.clusters[i]) != 0:
                    cluster = np.array(self.clusters[i])
                    mean = np.mean(cluster, axis=0)
                    self.cluster_centers[i] = mean

            # Gets the currently updated means
            current_mean = np.array(list(self.cluster_centers.values()))

        return self.cluster_centers, self.clusters, self.calculateLoss()



