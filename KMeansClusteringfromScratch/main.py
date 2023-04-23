from kmeans import KMeansClustering
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123456789)

X1 = np.random.normal(loc=10, scale=5, size=(300,2))
# X22 = np.random.normal(loc=4, scale=2, size=(300,1))
# X1 = np.concatenate([X11,X22], axis=1)


X2 = np.random.normal(loc=40, scale=5, size=(300,2))

X3 = np.random.normal(loc=20, scale=5, size=(300,2))
# X32 = np.random.uniform(low=9, high=15, size=(300,1))
# X3 = np.concatenate([X31,X32], axis=1)

X = np.concatenate((X1, X2, X3))
np.random.shuffle(X)
# X = np.random.uniform(low=-500, high=100, size=(1000,2))
print(f"Shape of data: {X.shape}")
k = 3
kmeans = KMeansClustering(X,k)
clusters = kmeans.kmeans(convergence=1e-10, explain=True)
print("Clusters:")
print(clusters)

for idx, clus in enumerate(kmeans.all_custers):
    fig, axs = plt.subplots()
    for i in np.unique(clus):
        axs.scatter(X[np.where(clus==i),0], X[np.where(clus==i),1],label=i)
    plt.savefig(f"./KMeansClusteringfromScratch/explain/{idx}_fig.png")



f1 = kmeans.plot_wcss()
f2 = kmeans.plot_cluster_data2d()
plt.show()