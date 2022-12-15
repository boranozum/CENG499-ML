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
        d = np.column_stack((dataset, labels))
        unique, counts = np.unique(d[:, -1], return_counts=True)
        counts = (-counts / np.sum(counts)) * np.log2(counts / np.sum(counts))
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

        d= np.column_stack((dataset, labels))
        attribute_index = np.where(np.array(self.features) == attribute)[0][0]
        column = d[:, attribute_index]
        unique, counts = np.unique(column, return_counts=True)
        counts = counts/np.sum(counts)

        for i in range(len(unique)):
            value = unique[i]
            filtered_dataset = d[d[:, attribute_index] == value]
            unique1, counts1 = np.unique(filtered_dataset[:,-1], return_counts=True)
            counts1 = counts1/np.sum(counts1)
            counts1 = (-1)*counts1*np.log2(counts1)
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
        attribute_index = np.where(np.array(self.features) == attribute)[0][0]
        d = np.array(dataset)
        column = d[:, attribute_index]
        unique, counts = np.unique(column, return_counts=True)
        counts = counts / np.sum(counts)

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

        max_gain = -100000
        max_attribute = None
        for attribute in self.features:
            if attribute not in used_attributes:
                if self.criterion == "information gain":
                    gain = self.calculate_information_gain__(dataset, labels, attribute)
                elif self.criterion == "gain ratio":
                    gain = self.calculate_gain_ratio__(dataset, labels, attribute)
                else:
                    raise Exception("criterion should be either information gain or gain ratio")
                if gain > max_gain:
                    max_gain = gain
                    max_attribute = attribute

        node = TreeNode(max_attribute)
        used_attributes.append(max_attribute)
        attribute_index = np.where(np.array(self.features) == max_attribute)[0][0]
        d = np.column_stack((dataset, labels))
        column = d[:, attribute_index]
        unique, counts = np.unique(column, return_counts=True)
        for i in range(len(unique)):
            value = unique[i]
            filtered_dataset = d[d[:, attribute_index] == value]
            filtered_labels = filtered_dataset[:,-1]
            unique1, counts1 = np.unique(filtered_labels, return_counts=True)
            unique1=unique1.astype(int)
            if len(unique1) == 1:
                 leaf = TreeLeafNode(None, unique1[0])
                 node.subtrees[value] = leaf
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
        node = self.root
        while isinstance(node, TreeNode):
            attribute_index = np.where(np.array(self.features) == node.attribute)[0][0]
            value = x[attribute_index]
            node = node.subtrees[value]

        predicted_label = node.labels

        return predicted_label

    def train(self):
        self.root = self.ID3__(self.dataset, self.labels, [])
        print("Training completed")