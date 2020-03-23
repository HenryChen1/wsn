
print(__doc__)
import numpy as np
from sklearn.cluster import KMeans


# #################################################
# generate sample data
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=0)
X = np.random.rand(300,2)
# #######################################################
# Compute Affinity Propagation
k_means = KMeans(n_clusters=5).fit(X) # preference采用负的欧氏距离
# cluster_centers_indices = k_means.cluster_centers_indices_
cluster_centroids = k_means.cluster_centers_
labels = k_means.labels_  # 样本标签

n_clusters_ = len(cluster_centroids) # 类簇数

print('估计的类簇数: %d' % n_clusters_)
# print('Homogeneity: %0.3f' % metrics.homogeneity_score(labels_true, labels))
# print('Completeness: %0.3f' %metrics.completeness_score(labels_true, labels))
# print('V-measure: %0.3f' %metrics.v_measure_score(labels_true, labels))
# print('Adjusted Rand Index:%0.3f' %metrics.adjusted_rand_score(labels_true, labels))
# print('Adjusted Mutual Information:%0.3f'%metrics.adjusted_mutual_info_score(labels_true, labels))
# print('Silhouette Coefficient:%0.3f' %metrics.silhouette_score(X, labels, metric='sqeuclidean')) # sqeuclidean欧式距离平方

# ##########################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_),colors):
    class_members = labels == k
    # print('k:',k)
    # print('labels:',labels)
    # print('cls_member--------',class_members)

    # cluster_center = X[cluster_centers_indices[k]]
    # print('cluster_center:', cluster_center)
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_centroids[k,0], cluster_centroids[k,1], 'o',markerfacecolor=col,
             markeredgecolor='k', markersize=10)

    # 划线
    # for x in X[class_members]:
    #     plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters:%d' %n_clusters_)
plt.show()

# import numpy as np
# a = np.random.rand(10,1)
# b = []
# for i , value in enumerate(a):
#     b = a[i]
#     print(b)
