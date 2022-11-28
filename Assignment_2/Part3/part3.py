import pickle
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



dataset = pickle.load(open("../data/part3_dataset.data", "rb"))

configs = [
    {
        'K': 2,
        'linkage': 'single',
        'distance': 'euclidean'
    },
    {
        'K': 3,
        'linkage': 'single',
        'distance': 'cosine'
    },
    {
        'K': 4,
        'linkage': 'complete',
        'distance': 'euclidean'
    },
    {
        'K': 5,
        'linkage': 'complete',
        'distance': 'cosine'
    }
]

for config in configs:

    clusters = AgglomerativeClustering(n_clusters=config['K'], linkage=config['linkage'],
                                       compute_distances=True, affinity=config['distance']).fit(dataset)

    silhouette_avg = silhouette_score(dataset, clusters.labels_)
    print(
        "For n_clusters =",
        config['K'],
        "The average silhouette_score is :",
        silhouette_avg,
    )

    fig, ax1 = plt.subplots(1,1)
    fig.set_size_inches(10, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(dataset) + (config['K'] + 1) * 10])

    sample_silhouette_values = silhouette_samples(dataset, clusters.labels_)

    y_lower = 10
    for i in range(config['K']):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[clusters.labels_ == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / config['K'])
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.show()



    # plt.title("HAC Dendogram")
    # # plot the top three levels of the dendrogram
    # plot_dendrogram(clusters, truncate_mode="level", p=3)
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    # plt.show()