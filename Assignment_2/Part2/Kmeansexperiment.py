from Part2.KMeans import KMeans
import pickle



dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))

kmeans_model = KMeans(dataset1)

kmeans_model.run()

dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))

