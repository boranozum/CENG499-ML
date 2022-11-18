import pickle
from Distance import Distance
from Part1.KNN import KNN


dataset, labels = pickle.load(open("../data/part1_dataset.data", "rb"))
