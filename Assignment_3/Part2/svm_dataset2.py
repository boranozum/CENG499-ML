import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC

dataset, labels = pickle.load(open("../data/part2_dataset2.data", "rb"))

scaler = StandardScaler()

skf = StratifiedKFold(n_splits=10, shuffle=True)

processed_dataset = scaler.fit_transform(dataset, labels)

configurations = [
    {
        'kernel': ['rbf', 'sigmoid'],
        'C': [1, 10]
    }
]

mean_scores = [None for _ in range(4)]

for i in range(5):
    clf = SVC()
    grid = GridSearchCV(clf, configurations, cv=skf, scoring="accuracy")
    grid.fit(processed_dataset, labels)
    for j in range(4):
        if mean_scores[j] is None:
            result = {"params": grid.cv_results_["params"][j], "mean_test_score": [grid.cv_results_["mean_test_score"][j]]}
            mean_scores[j] = result
        else:
            mean_scores[j]["mean_test_score"].append(grid.cv_results_["mean_test_score"][j])

for i in range(4):
    # calculate confidence interval
    mean_value = np.mean(mean_scores[i]["mean_test_score"])
    std_value = np.std(mean_scores[i]["mean_test_score"])
    confidence_interval = 1.96 * std_value / np.sqrt(5)
    print(f"Mean score: {mean_value}, Confidence interval: {confidence_interval}, Parameters: {mean_scores[i]['params']}")

