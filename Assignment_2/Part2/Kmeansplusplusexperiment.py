from Part2.KMeansPlusPlus import KMeansPlusPlus
import pickle


dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))


model = KMeansPlusPlus(dataset1)
model.run()

dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))
