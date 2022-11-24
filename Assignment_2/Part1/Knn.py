import numpy as np
from Distance import Distance

class KNN:
    def __init__(self, dataset, data_label, similarity_function, similarity_function_parameters=None, K=1):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics
        :param K: how many neighbors to consider, integer
        """
        self.K = K
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters

    def majorityVote(self, indices):

        labels = self.dataset_label[indices]

        values, counts = np.unique(labels, return_counts=True)
        ind = np.argmax(counts)

        return values[ind]

    def predict(self, instance):

        params = [self.similarity_function, 1, self.dataset, instance]

        if type(self.similarity_function_parameters) is list:
            params += self.similarity_function_parameters
        elif self.similarity_function_parameters is not None:
            params += [self.similarity_function_parameters]

        similarity_vec = np.apply_along_axis(*params)

        neighbour_indices = np.argpartition(similarity_vec, self.K)[:self.K]

        if self.similarity_function == Distance.calculateCosineDistance:
            neighbour_indices = np.argpartition(similarity_vec, -self.K)[-self.K:]

        return self.majorityVote(neighbour_indices)



