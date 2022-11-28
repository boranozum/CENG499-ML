from Part2.KMeans import KMeans
import pickle
import matplotlib.pyplot as plt
import numpy as np


dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))

dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))

k_values = [i for i in range(2, 11)]       # 10 possible k values for kmeans

# Dictionaries that stores the averages of the minimum values calculated at each iteration for every k value
# with respect to the dataset
lowest_scores1 = {i: [] for i in range(2, 11)}
lowest_scores2 = {i: [] for i in range(2, 11)}

# Iterates through the K values
for k in k_values:

    # Each k value will be tested 10 times
    for i in range(10):

        # Initial arbitrary large values for finding minimum loss value
        min_loss1 = 999999999
        min_loss2 = 999999999

        # Finds the minimum of the 10 loss values
        for j in range(10):

            print('K=%d  |  i=%d  |  j=%d' % (k, i, j))

            # Creates two instances of kmeans model for each dataset with the K value
            kmeans_model1 = KMeans(dataset1, k)
            kmeans_model2 = KMeans(dataset2, k)

            # Clusters, cluster centers and loss values are computed for each model
            cluster_centers1, clusters1, loss_value1 = kmeans_model1.run()
            cluster_centers2, clusters2, loss_value2 = kmeans_model2.run()

            # Computes the minimum loss for each dataset
            if min_loss1 > loss_value1:
                min_loss1 = loss_value1

            if min_loss2 > loss_value2:
                min_loss2 = loss_value2

        lowest_scores1[k].append(min_loss1)
        lowest_scores2[k].append(min_loss2)

# Mean arrays that will be used to plot k vs. loss graphs
dataset1_means = []
dataset2_means = []

# Calculates the means and confidence intervals for each K value and prints them
for i in range(2, 11):

    mean1 = np.mean(np.array(lowest_scores1[i]))
    mean2 = np.mean(np.array(lowest_scores2[i]))

    print(('Confidence interval of K=%d at dataset1: %.2f' + u'\u00B1' + ' %.2f') % (i, mean1, 1.96*np.std(np.array(lowest_scores1[i]))/(len(np.array(lowest_scores1[i]))**.5)))
    print(('Confidence interval of K=%d at dataset2: %.2f' + u'\u00B1' + ' %.2f') % (i, mean2, 1.96*np.std(np.array(lowest_scores2[i]))/(len(np.array(lowest_scores2[i]))**.5)))

    dataset1_means.append(mean1)
    dataset2_means.append(mean2)

# Draws two k vs. loss graphs for the given datasets
plt.plot(k_values, dataset1_means)
plt.ylabel('Loss')
plt.xlabel('K')
plt.title('K vs. Loss for Kmeans Method on Dataset1')
plt.show()

plt.plot(k_values, dataset2_means)
plt.ylabel('Loss')
plt.xlabel('K')
plt.title('K vs. Loss for Kmeans Method on Dataset2')
plt.show()