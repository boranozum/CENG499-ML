from Part2.KMeansPlusPlus import KMeansPlusPlus
import pickle
import numpy as np
import matplotlib.pyplot as plt

dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))

dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))


k_values = [i for i in range(2, 11)]

lowest_scores1 = {i: [] for i in range(2, 11)}
lowest_scores2 = {i: [] for i in range(2, 11)}

for k in k_values:

    for i in range(10):

        min_loss1 = 999999999
        min_loss2 = 999999999

        for j in range(10):

            print('K=%d  |  i=%d  |  j=%d' % (k, i, j))

            kmeans_model1 = KMeansPlusPlus(dataset1, k)
            kmeans_model2 = KMeansPlusPlus(dataset2, k)

            cluster_centers1, clusters1, loss_value1 = kmeans_model1.run()
            cluster_centers2, clusters2, loss_value2 = kmeans_model2.run()

            if min_loss1 > loss_value1:
                min_loss1 = loss_value1

            if min_loss2 > loss_value2:
                min_loss2 = loss_value2

        lowest_scores1[k].append(min_loss1)
        lowest_scores2[k].append(min_loss2)

dataset1_means = []
dataset2_means = []

for i in range(2, 11):

    mean1 = np.mean(np.array(lowest_scores1[i]))
    mean2 = np.mean(np.array(lowest_scores2[i]))

    print(('Confidence interval of K=%d at dataset1: %.2f' + u'\u00B1' + ' %.2f') % (i, mean1, 1.96*np.std(np.array(lowest_scores1[i]))/(len(np.array(lowest_scores1[i]))**.5)))
    print(('Confidence interval of K=%d at dataset2: %.2f' + u'\u00B1' + ' %.2f') % (i, mean2, 1.96*np.std(np.array(lowest_scores2[i]))/(len(np.array(lowest_scores2[i]))**.5)))

    dataset1_means.append(mean1)
    dataset2_means.append(mean2)


plt.plot(k_values, dataset1_means)
plt.ylabel('Loss')
plt.xlabel('K')
plt.title('K vs. Loss for Kmeans++ Method on Dataset1')
plt.show()

plt.plot(k_values, dataset2_means)
plt.ylabel('Loss')
plt.xlabel('K')
plt.title('K vs. Loss for Kmeans++ Method on Dataset2')
plt.show()