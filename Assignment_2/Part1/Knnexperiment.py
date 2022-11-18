import pickle
from Distance import Distance
from Part1.Knn import KNN
from sklearn.model_selection import train_test_split

dataset, labels = pickle.load(open("../data/part1_dataset.data", "rb"))

x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.1, random_state=0, stratify=labels)

