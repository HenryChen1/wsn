import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from sklearn.mixture import GaussianMixture

# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]
# X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.3, 0.4, 0.3],
#                   random_state = 0)

X = np.random.rand(300,2)
##设置gmm函数
gmm = GaussianMixture(n_components=5, covariance_type='full').fit(X)
##训练数据
y_pred = gmm.predict(X)

##绘图
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()