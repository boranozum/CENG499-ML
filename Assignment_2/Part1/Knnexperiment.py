import pickle
from Knn import KNN
from Distance import Distance
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


dataset, labels = pickle.load(open("../data/part1_dataset.data", "rb"))

# Hyperparameter configuration array that is used for the grid search
configs = [
    {
        'K': 5,
        'similarity_function': Distance.calculateCosineDistance,
        'similarity_function_parameters': None,
    },
    {
        'K': 10,
        'similarity_function': Distance.calculateCosineDistance,
        'similarity_function_parameters': None,
    },
    {
        'K': 30,
        'similarity_function': Distance.calculateCosineDistance,
        'similarity_function_parameters': None,
    },
    {
        'K': 5,
        'similarity_function': Distance.calculateMinkowskiDistance,
        'similarity_function_parameters': 2,
    },
    {
        'K': 10,
        'similarity_function': Distance.calculateMinkowskiDistance,
        'similarity_function_parameters': 2,
    },
    {
        'K': 30,
        'similarity_function': Distance.calculateMinkowskiDistance,
        'similarity_function_parameters': 2,
    },
    {
        'K': 5,
        'similarity_function': Distance.calculateMahalanobisDistance,
        'similarity_function_parameters': None,
    },
    {
        'K': 10,
        'similarity_function': Distance.calculateMahalanobisDistance,
        'similarity_function_parameters': None,
    },
    {
        'K': 30,
        'similarity_function': Distance.calculateMahalanobisDistance,
        'similarity_function_parameters': None,
    }
]

# StratifiedKFold instance for splitting the dataset into partitions with equal class proportions.
# Shuffles the dataset at each split
skf = StratifiedKFold(n_splits=10, shuffle=True)

# Will store the average accuracy scores for each configuration and will be used for
# calculating confidence intervals
config_accuracies = []

j = 1
for config in configs:
    iteration_accuracy = []

    # Each configuration will be run 5 times
    for i in range(1, 6):

        cv_accuracy = 0
        for train_index, test_index in skf.split(dataset, labels):

            # In case of the mahalanobis distance calculations, finds the inverse of the covariance matrix
            if config['similarity_function'] == Distance.calculateMahalanobisDistance:
                S = np.cov(dataset.T)       # Takes the covariance matrix of the dataset
                S_minus_1 = np.linalg.inv(S)    # Calculates the inverse of the covariance matrix
                config['similarity_function_parameters'] = S_minus_1    # Sets as the parameter for the similarity function

            # Initializes the model instance for every configuration
            knn_model = KNN(dataset[train_index],
                            labels[train_index],
                            similarity_function=config['similarity_function'],
                            similarity_function_parameters=config['similarity_function_parameters'],
                            K=config['K'])

            # For every sample in the test partition in kfold cv, predicts a label and stores it in a 1D numpy array
            predicted = np.apply_along_axis(knn_model.predict, 1, dataset[test_index])

            # Gets the actual labels of the selected test samples
            label_for_predicted = labels[test_index]

            # Calculates the accuracy of the test partition predictions
            accuracy = accuracy_score(label_for_predicted, predicted)

            # Sums up these accuracies to take average later
            cv_accuracy += accuracy*100

            print('Configuration: %d | Iteration: %d | Accuracy: %.2f' % (j,i,(accuracy*100)))

        # Stores the average accuracy of the iteration
        iteration_accuracy.append(cv_accuracy/10)

    # Maps each iteration accuracy array to the configurations
    config_accuracies.append(iteration_accuracy)

    j += 1

j = 1
max_mean = 0
selected_index = 0

# For loop for calculating and printing confidence intervals of each configuration
for acc in config_accuracies:

    n = np.array(acc)

    mean = np.mean(acc)

    # Find the maximum mean among the configuration accuracy means
    if mean > max_mean:
        max_mean = mean
        selected_index = j-1

    print(('Confidence interval of configuration %d : %.2f ' + u'\u00B1' + ' %.2f') % (j, np.mean(n), 1.96*np.std(n)/(len(n)**.5)))

    j+=1

print('Configuration %d yielded the best accuracy score' % (selected_index+1))
