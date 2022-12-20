import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC

dataset, labels = pickle.load(open("../data/part2_dataset2.data", "rb"))

# Standardize the dataset
scaler = StandardScaler()
processed_dataset = scaler.fit_transform(dataset, labels)

# CV instance with 10 folds and shuffle
skf = StratifiedKFold(n_splits=10, shuffle=True)

# Configurations to be tested
configurations = [
    {
        'kernel': ['rbf', 'sigmoid'],
        'C': [1, 10]
    }
]

# Result array for each configuration
mean_scores = [None for _ in range(4)]

# Run the classifier for each configuration 5 times and store the results
for i in range(5):
    # Create a classifier for each configuration
    clf = SVC()

    # Run a grid search for each configuration with 10-fold stratified CV using the accuracy metric
    grid = GridSearchCV(clf, configurations, cv=skf, scoring="accuracy")
    grid.fit(processed_dataset, labels)

    # Store the results
    for j in range(4):
        if mean_scores[j] is None:
            result = {"params": grid.cv_results_["params"][j], "mean_test_score": [grid.cv_results_["mean_test_score"][j]]}
            mean_scores[j] = result
        else:
            mean_scores[j]["mean_test_score"].append(grid.cv_results_["mean_test_score"][j])


# Calculate the confidence interval for each configuration and print the results
for i in range(4):
    mean_value = np.mean(mean_scores[i]["mean_test_score"])
    std_value = np.std(mean_scores[i]["mean_test_score"])
    confidence_interval = 1.96 * std_value / np.sqrt(5)
    print(f"Mean score: {mean_value}, Confidence interval: {confidence_interval}, Parameters: {mean_scores[i]['params']}")

