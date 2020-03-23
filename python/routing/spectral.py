from sklearn.cluster import SpectralClustering
import numpy as np
import matplotlib.pyplot as plt


colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

X = np.random.rand(100,2)

clustering = SpectralClustering(n_clusters=5,
                                assign_labels="discretize",
                                random_state=0).fit(X)

y_pred = clustering.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=50)

plt.show()