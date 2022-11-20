import pickle
from Knn import KNN
from Assignment_2.Distance import Distance
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold

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

skf = StratifiedKFold(n_splits=10)

for config in configs:

    x_train, y_train, x_test, y_test = train_test_split(dataset, labels, test_size=0.4, random_state=0, stratify=labels)

    if config['similarity_function'] == Distance.calculateMahalanobisDistance:
        S_minus_1 = np.linalg.inv(np.cov(x_train))


