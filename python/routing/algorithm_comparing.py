print(__doc__)

import time

import numpy as np
import matplotlib.pyplot as plt

import skfuzzy
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from hdbscan import HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
n_samples = 1500
# noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
#                                       noise=.05)
# noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
# blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
# no_structure = np.random.rand(n_samples, 2), None

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

clustering_names = [
    'MiniBatchKMeans', 'AffinityPropagation',
    'SpectralClustering', 'Ward', 'AgglomerativeClustering',
    'HDBSCAN', 'GMM']

plt.figure(figsize=(len(clustering_names) * 2 + 3, 9.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.1)

plot_num = 1

# datasets = [noisy_circles, noisy_moons, blobs, no_structure]
# for i_dataset, dataset in enumerate(datasets):
#     X, y = dataset
#     # normalize dataset for easier parameter selection
#     X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
X = np.random.rand(300,2)
bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

    # connectivity matrix for structured Ward
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
    # make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)

    # create clustering estimators
# ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
two_means = cluster.MiniBatchKMeans(n_clusters=5)
ward = cluster.AgglomerativeClustering(n_clusters=5, linkage='ward',
                                           connectivity=connectivity)
spectral = cluster.SpectralClustering(n_clusters=5,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")
hdbscan = HDBSCAN(min_cluster_size=7)
affinity_propagation = cluster.AffinityPropagation(damping=.9,
                                                       preference=-2.1)

average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock", n_clusters=5,
        connectivity=connectivity)
gmm = GaussianMixture(n_components=5, covariance_type='full').fit(X)
# birch = cluster.Birch(n_clusters=5)
clustering_algorithms = [
        two_means, affinity_propagation, spectral, ward, average_linkage,
        hdbscan, gmm]

for name, algorithm in zip(clustering_names, clustering_algorithms):
        # predict cluster memberships
    t0 = time.time()
    algorithm.fit(X)
    t1 = time.time()
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)

        # plot
    plt.subplot(330+plot_num)#len(clustering_algorithms)
    # if i_dataset == 0:
    plt.title(name, size=12)
    plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

    if hasattr(algorithm, 'cluster_centers_'):
        centers = algorithm.cluster_centers_
        center_colors = colors[:len(centers)]
        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(())
    plt.yticks(())
    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
    plot_num += 1

###############################################################################
#   FCM algorithm

# cntr, u =skfuzzy.cluster.cmeans(X, 5, 2, error=0.005, maxiter=1000, init=None)[0:2]
#
# plt.subplot(339)#len(clustering_algorithms)
# plt.title('FCM')
#
# for j in range(5):
#     plt.plot(X[:, 0],
#              X[:, 1], 'o', color=colors[u.argmax(axis=0) == j],
#              label='series ' + str(j))




# for cluster_id, cntr in enumerate(cntr):
#     cluster_membership = np.argmax(u, axis=0)
#     plt.scatter(X[:, 0], X[:, 1], color=colors[cluster_membership].tolist(), s=10)



plt.show()