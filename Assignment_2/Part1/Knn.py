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
        """
        Computes the label that is the majority of nearest neighbour labels and returns it as the predicted label.
        :param indices: 1D Numpy Array (Indices of the computed nearest neighbours in the dataset)
        :return: int (label)
        """

        labels = self.dataset_label[indices]    # Gets the labels of the neighbours with given indices

        values, counts = np.unique(labels, return_counts=True)      # Counts the occurrences of each label in the array
        ind = np.argmax(counts)     # Finds the index of the label that occurs the most among the neighbour labels

        return values[ind]      # Returns the label

    def predict(self, instance, ):
        """
        Predicts a label for the given instance.
        :param instance: 1D Numpy Array (Data sample)
        :return: int (label)
        """

        params = [self.similarity_function, 1, self.dataset, instance]  # Parameter array that will be given to
                                                                        # np.apply_along_axis for concurrent distance
                                                                        # calculation between the given sample and the
                                                                        # train dataset

        # Parameter adjustment based on the similarity function
        if type(self.similarity_function_parameters) is list:
            params += self.similarity_function_parameters
        elif self.similarity_function_parameters is not None:
            params += [self.similarity_function_parameters]

        similarity_vec = np.apply_along_axis(*params)       # Calculates every distance between the given instance
                                                            # and the samples in the dataset with the similarity function
                                                            # and stores in a 1D numpy array

        neighbour_indices = np.argpartition(similarity_vec, self.K)[:self.K]    # In the array that contains the distances
                                                                                # gathers the indices of the top K min values

        # Situation is reversed in case of the calculation with cosine distance since cosine_distance = 1-cosine_similarity
        if self.similarity_function == Distance.calculateCosineDistance:
            neighbour_indices = np.argpartition(similarity_vec, -self.K)[-self.K:]

        # Finds the majority of the labels between the neighbours and returns
        return self.majorityVote(neighbour_indices)



