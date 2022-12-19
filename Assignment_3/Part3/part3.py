import numpy as np
from DataLoader import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def calculate_f1_score(model, x_test, y_test):
    y_pred = model.predict(x_test)
    tp = np.sum(np.logical_and(y_pred == 1, y_test == 1))
    fp = np.sum(np.logical_and(y_pred == 1, y_test == 0))
    fn = np.sum(np.logical_and(y_pred == 0, y_test == 1))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score

data_path = "../data/credit.data"

dataset, labels = DataLoader.load_credit_with_onehot(data_path)

configs = {
    'knn': {
        'n_neighbors': [3, 5],
        'metric': ['euclidean', 'manhattan'],
    },
    'svm': {
        'kernel': ['rbf', 'sigmoid'],
        'C': [1, 10]
    },
    'dt': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5]
    },
    'rf': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5]
    }
}

results = {
    'knn': {
        "hyperparameter_scores": {},
        "mean_test_score": [],
        "f1_score": []
    },
    'svm': {
        "hyperparameter_scores": {},
        "mean_test_score": [],
        "f1_score": []
    },
    'dt': {
        "hyperparameter_scores": {},
        "mean_test_score": [],
        "f1_score": []
    },
    'rf': {
        "hyperparameter_scores": {},
        "mean_test_score": [],
        "f1_score": []
    }
}
fold = 1
outer_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5)
for train_index, test_index in outer_cv.split(dataset, labels):
    X_train, X_test = dataset[train_index], dataset[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
    # min-max normalization
    X_train_normalized = -1 + (X_train - np.min(X_train)) * (1 - (-1)) / (np.max(X_train) - np.min(X_train))
    X_test_normalized = -1 + (X_test - np.min(X_train)) * (1 - (-1)) / (np.max(X_train) - np.min(X_train))

    # knn hyperparameter tuning
    grid = GridSearchCV(KNeighborsClassifier(), configs['knn'], cv=inner_cv, scoring='accuracy')
    grid.fit(X_train_normalized, y_train)
    print(f"Fold {fold} - KNN - Inner CV completed")
    for i in range(4):
        if str(grid.cv_results_["params"][i]) not in results['knn']['hyperparameter_scores']:
            results['knn']['hyperparameter_scores'][str(grid.cv_results_["params"][i])] = [grid.cv_results_["mean_test_score"][i]]
        else:
            results['knn']['hyperparameter_scores'][str(grid.cv_results_["params"][i])].append(grid.cv_results_["mean_test_score"][i])

    # knn evaluation
    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(X_train_normalized, y_train)
    results['knn']['mean_test_score'].append(knn.score(X_test_normalized, y_test))
    results['knn']['f1_score'].append(calculate_f1_score(knn, X_test_normalized, y_test))
    print(f"Fold {fold} - KNN - Evaluation completed")
    # svm hyperparameter tuning
    grid = GridSearchCV(SVC(), configs['svm'], cv=inner_cv, scoring='accuracy')
    grid.fit(X_train_normalized, y_train)
    print(f"Fold {fold} - SVM - Inner CV completed")
    for i in range(4):
        if str(grid.cv_results_["params"][i]) not in results['knn']['hyperparameter_scores']:
            results['svm']['hyperparameter_scores'][str(grid.cv_results_["params"][i])] = [
                grid.cv_results_["mean_test_score"][i]]
        else:
            results['svm']['hyperparameter_scores'][str(grid.cv_results_["params"][i])].append(
                grid.cv_results_["mean_test_score"][i])

    # svm evaluation
    svm = SVC(**grid.best_params_)
    svm.fit(X_train_normalized, y_train)
    results['svm']['mean_test_score'].append(svm.score(X_test_normalized, y_test))
    results['svm']['f1_score'].append(calculate_f1_score(svm, X_test_normalized, y_test))
    print(f"Fold {fold} - SVM - Evaluation completed")
    # dt hyperparameter tuning
    grid = GridSearchCV(DecisionTreeClassifier(), configs['dt'], cv=inner_cv, scoring='accuracy')
    grid.fit(X_train_normalized, y_train)
    print(f"Fold {fold} - DT - Inner CV completed")
    for i in range(4):
        if str(grid.cv_results_["params"][i]) not in results['knn']['hyperparameter_scores']:
            results['dt']['hyperparameter_scores'][str(grid.cv_results_["params"][i])] = [
                grid.cv_results_["mean_test_score"][i]]
        else:
            results['dt']['hyperparameter_scores'][str(grid.cv_results_["params"][i])].append(
                grid.cv_results_["mean_test_score"][i])

    # dt evaluation
    dt = DecisionTreeClassifier(**grid.best_params_)
    dt.fit(X_train_normalized, y_train)
    results['dt']['mean_test_score'].append(dt.score(X_test_normalized, y_test))
    results['dt']['f1_score'].append(calculate_f1_score(dt, X_test_normalized, y_test))
    print(f"Fold {fold} - DT - Evaluation completed")
    # rf hyperparameter tuning
    config_means = []
    for i in range(5):
        grid = GridSearchCV(RandomForestClassifier(), configs['rf'], cv=inner_cv, scoring='accuracy')
        grid.fit(X_train_normalized, y_train)
        config_means.append(grid.cv_results_['mean_test_score'])
        print(f'Fold {fold} - RF - Inner CV {i} completed')

    config_means = np.mean(config_means, axis=0)
    for i in range(4):
        if str(grid.cv_results_["params"][i]) not in results['knn']['hyperparameter_scores']:
            results['rf']['hyperparameter_scores'][str(grid.cv_results_["params"][i])] = [
                grid.cv_results_["mean_test_score"][i]]
        else:
            results['rf']['hyperparameter_scores'][str(grid.cv_results_["params"][i])].append(
                grid.cv_results_["mean_test_score"][i])

    print(f"Fold {fold} - RF - Inner CV completed")
    # rf evaluation
    best_rf_config = grid.cv_results_['params'][np.argmax(np.array(config_means))]
    rf = RandomForestClassifier(**best_rf_config)
    rf.fit(X_train_normalized, y_train)
    results['rf']['mean_test_score'].append(rf.score(X_test_normalized, y_test))
    results['rf']['f1_score'].append(calculate_f1_score(rf, X_test_normalized, y_test))
    print(f"Fold {fold} - RF - Evaluation completed")
    fold += 1
for model in results:

    print(model)
    print('hyperparameter_scores')
    for config in results[model]['hyperparameter_scores']:
        # calculate mean and confidence interval
        mean_value = np.mean(results[model]['hyperparameter_scores'][config])
        std_value = np.std(results[model]['hyperparameter_scores'][config])
        confidence_interval = 1.96 * std_value / np.sqrt(len(results[model]['hyperparameter_scores'][config]))
        print(f"{config} - {mean_value} +/- {confidence_interval}")

    print('mean_test_score')
    mean_value = np.mean(results[model]['mean_test_score'])
    std_value = np.std(results[model]['mean_test_score'])
    confidence_interval = 1.96 * std_value / np.sqrt(len(results[model]['mean_test_score']))
    print(f"{mean_value} +/- {confidence_interval}")

    print('f1_score')

    mean_value = np.mean(results[model]['f1_score'])
    std_value = np.std(results[model]['f1_score'])
    confidence_interval = 1.96 * std_value / np.sqrt(len(results[model]['f1_score']))
    print(f"{mean_value} +/- {confidence_interval}")