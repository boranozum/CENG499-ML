import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

dataset, labels = pickle.load(open("../data/part2_dataset1.data", "rb"))

configurations = [
    {
        "kernel": "sigmoid",
        "C": 1,
    },
    {
        "kernel": "sigmoid",
        "C": 10,
    },
    {
        "kernel": "rbf",
        "C": 1,
    },
    {
        "kernel": "rbf",
        "C": 10,
    },
]

classifiers = []

for configuration in configurations:
    clf = SVC(kernel=configuration["kernel"], C=configuration["C"]).fit(dataset, labels)
    classifiers.append(clf)

f, ax = plt.subplots(2, 2, figsize=(10, 10))
for i in range(2):
    for j in range(2):
        clf = classifiers[i*2+j]
        display = DecisionBoundaryDisplay.from_estimator(clf, dataset, ax=ax[i][j], alpha=0.7,
                                                         response_method="predict", xlabel=labels[0], ylabel=labels[1])
        display.ax_.scatter(dataset[:, 0], dataset[:, 1], c=labels, edgecolors='k')
        ax[i][j].set_title(f"Kernel: {clf.kernel}, C: {clf.C}")
plt.show()

