import pickle
from Knn import KNN
from Distance import Distance
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


dataset, labels = pickle.load(open("../data/part1_dataset.data", "rb"))

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

confidence_intervals = []

skf = StratifiedKFold(n_splits=10, shuffle=True)

config_accuracies = []

j = 1
for config in configs:
    iteration_accuracy = []
    for i in range(1, 6):
        if config['similarity_function'] == Distance.calculateMahalanobisDistance:
            S = np.cov(dataset.T)
            S_minus_1 = np.linalg.inv(S)
            config['similarity_function_parameters'] = S_minus_1

        cv_accuracy = 0
        for train_index, test_index in skf.split(dataset, labels):
            knn_model = KNN(dataset[train_index],
                            labels[train_index],
                            similarity_function=config['similarity_function'],
                            similarity_function_parameters=config['similarity_function_parameters'],
                            K=config['K'])

            predicted = np.apply_along_axis(knn_model.predict, 1, dataset[test_index])

            label_for_predicted = labels[test_index]

            accuracy = accuracy_score(label_for_predicted, predicted)

            cv_accuracy += accuracy*100

            print('Configuration: %d | Iteration: %d | Accuracy: %.2f' % (j,i,(accuracy*100)))

        iteration_accuracy.append(cv_accuracy/10)

    config_accuracies.append(iteration_accuracy)

    j += 1

j = 1
max_mean = 0
selected_index = 0
for acc in config_accuracies:

    n = np.array(acc)

    mean = np.mean(acc)

    if mean > max_mean:
        max_mean = mean
        selected_index = j-1

    print(('Confidence interval of configuration %d : %.2f ' + u'\u00B1' + ' %.2f') % (j, np.mean(n), 1.96*np.std(n)/(len(n)**.5)))

    j+=1

print('Configuration %d yielded the best accuracy score' % (selected_index+1))
