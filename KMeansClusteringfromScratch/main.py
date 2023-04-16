from kmeans import KMeansClustering
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123456789)

# X1 = np.random.uniform(low=1, high=5, size=(1000,2))
# X2 = np.random.uniform(low=10, high=15, size=(1000,2))
# X3 = np.random.uniform(low=20, high=25, size=(1000,2))
# X = np.concatenate((X1, X2, X3))
# np.random.shuffle(X)
X = np.random.uniform(low=-100, high=100, size=(1000,2))
print(f"Shape of data: {X.shape}")
k = 4
kmeans = KMeansClustering(X,3)
clusters = kmeans.kmeans(convergence=1e-10)
print("Clusters:")
print(clusters)
f1 = kmeans.plot_wcss()
f2 = kmeans.plot_cluster_data2d()
plt.show()