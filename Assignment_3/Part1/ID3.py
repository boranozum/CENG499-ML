import numpy as np
# In the decision tree, non-leaf nodes are going to be represented via TreeNode
class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}

# In the decision tree, leaf nodes are going to be represented via TreeLeafNode
class TreeLeafNode:
    def __init__(self, data, label):
        self.data = data
        self.labels = label

class DecisionTree:
    def __init__(self, dataset: list, labels, features, criterion="information gain"):
        """
        :param dataset: array of data instances, each data instance is represented via an Python array
        :param labels: array of the labels of the data instances
        :param features: the array that stores the name of each feature dimension
        :param criterion: depending on which criterion ("information gain" or "gain ratio") the splits are to be performed
        """
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None

        # further variables and functions can be added...


    def calculate_entropy__(self, dataset, labels):
        """
        :param dataset: array of the data instances
        :param labels: array of the labels of the data instances
        :return: calculated entropy value for the given dataset
        """
        entropy_value = 0.0

        """
        Entropy calculations
        """
        # count the occurrences of each label

        # Column-wise concatenation of dataset and labels
        d = np.column_stack((dataset, labels))

        # Get the unique labels and their counts
        unique, counts = np.unique(d[:, -1], return_counts=True)

        # Calculate the probabilities of each label and multiply them with their log2 values
        counts = (-counts / np.sum(counts)) * np.log2(counts / np.sum(counts))

        # Sum the values to get the entropy
        entropy_value = np.sum(counts)

        return entropy_value

    def calculate_average_entropy__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an average entropy value is calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute an average entropy value is going to be calculated...
        :return: the calculated average entropy value for the given attribute
        """
        average_entropy = 0.0
        """
            Average entropy calculations
        """

        # Column-wise concatenation of dataset and labels
        d = np.column_stack((dataset, labels))

        # Get the attribute index of the attribute parameter in the features array
        attribute_index = np.where(np.array(self.features) == attribute)[0][0]

        # Get the unique values of the attribute
        column = d[:, attribute_index]
        unique, counts = np.unique(column, return_counts=True)

        # Normalize the counts to get the probabilities
        counts = counts/np.sum(counts)

        # For each unique value of the attribute
        for i in range(len(unique)):
            value = unique[i]

            # Get the rows that have the value
            filtered_dataset = d[d[:, attribute_index] == value]

            # Calculate the unique labels and their counts for the rows that have the value
            unique1, counts1 = np.unique(filtered_dataset[:,-1], return_counts=True)

            # Calculate the probabilities of each label and multiply them with their log2 values
            counts1 = counts1/np.sum(counts1)
            counts1 = (-1)*counts1*np.log2(counts1)

            # Sum the values to get the entropy
            average_entropy += counts[i]*np.sum(counts1)

        return average_entropy

    def calculate_information_gain__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an information gain score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the information gain score is going to be calculated...
        :return: the calculated information gain score
        """
        """
            Information gain calculations
        """
        # Information gain = entropy(parent) - average entropy(children)
        return self.calculate_entropy__(dataset, labels) - self.calculate_average_entropy__(dataset, labels, attribute)

    def calculate_intrinsic_information__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances on which an intrinsic information score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the intrinsic information score is going to be calculated...
        :return: the calculated intrinsic information score
        """
        """
            Intrinsic information calculations for a given attribute
        """
        # Get the attribute index of the attribute parameter in the features array
        attribute_index = np.where(np.array(self.features) == attribute)[0][0]

        # Column-wise concatenation of dataset and labels
        d = np.column_stack((dataset, labels))

        # Get the rows that have the value
        column = d[:, attribute_index]

        # Get the unique values of the attribute
        unique, counts = np.unique(column, return_counts=True)
        counts = counts / np.sum(counts)

        # Calculate the probabilities of each label and multiply them with their log2 values
        return -np.sum(counts*np.log2(counts))
    def calculate_gain_ratio__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances with which a gain ratio is going to be calculated
        :param labels: array of labels of those instances
        :param attribute: for which attribute the gain ratio score is going to be calculated...
        :return: the calculated gain ratio score
        """
        """
            Your implementation
        """

        # Gain ratio = information gain / intrinsic information
        return self.calculate_information_gain__(dataset, labels, attribute)/self.calculate_intrinsic_information__(dataset, labels, attribute)


    def ID3__(self, dataset, labels, used_attributes):
        """
        Recursive function for ID3 algorithm
        :param dataset: data instances falling under the current  tree node
        :param labels: labels of those instances
        :param used_attributes: while recursively constructing the tree, already used labels should be stored in used_attributes
        :return: it returns a created non-leaf node or a created leaf node
        """
        """
            Your implementation
        """

        # For each level of the tree, find the best attribute to split on among the remaining attributes
        max_gain = -100000
        max_attribute = None
        for attribute in self.features:
            if attribute not in used_attributes:
                if self.criterion == "information gain":
                    gain = self.calculate_information_gain__(dataset, labels, attribute)
                elif self.criterion == "gain ratio":
                    gain = self.calculate_gain_ratio__(dataset, labels, attribute)
                if gain > max_gain:
                    max_gain = gain
                    max_attribute = attribute

        # Create a node with the best attribute
        node = TreeNode(max_attribute)

        # Append the attribute to the used attributes
        used_attributes.append(max_attribute)

        # Get the attribute index of the best attribute in the features array
        attribute_index = np.where(np.array(self.features) == max_attribute)[0][0]

        # Column-wise concatenation of dataset and labels
        d = np.column_stack((dataset, labels))

        # Get the rows that have the value and the unique values of the attribute
        column = d[:, attribute_index]
        unique, counts = np.unique(column, return_counts=True)

        # For each unique value of the attribute
        for i in range(len(unique)):
            value = unique[i]

            # Get the rows that have the value
            filtered_dataset = d[d[:, attribute_index] == value]

            # Get the labels of the rows that have the value
            filtered_labels = filtered_dataset[:,-1]
            unique1, counts1 = np.unique(filtered_labels, return_counts=True)
            unique1=unique1.astype(int)

            # If all the labels are the same, create a leaf node with that label
            if len(unique1) == 1:
                 leaf = TreeLeafNode(None, unique1[0])
                 node.subtrees[value] = leaf
            # If not, recursively call the ID3 function to create a subtree
            else:
                node.subtrees[value] = self.ID3__(filtered_dataset, filtered_labels, used_attributes)

        return node
    def predict(self, x):
        """
        :param x: a data instance, 1 dimensional Python array 
        :return: predicted label of x
        
        If a leaf node contains multiple labels in it, the majority label should be returned as the predicted label
        """
        predicted_label = None
        """
            Your implementation
        """
        # Get the root node
        node = self.root

        # While the node is not a leaf node
        while isinstance(node, TreeNode):
            # Get the attribute index of the attribute in the features array
            attribute_index = np.where(np.array(self.features) == node.attribute)[0][0]

            # Get the value of the attribute in the data instance
            value = x[attribute_index]

            # Continue with the subtree that has the value
            node = node.subtrees[value]

        # Return the label of the leaf node
        predicted_label = node.labels
        return predicted_label

    def train(self):
        self.root = self.ID3__(self.dataset, self.labels, [])
        print("Training completed")