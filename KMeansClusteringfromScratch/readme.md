# KMeans Clustering from Scratch

In this repository, Kmeans clustering algorithm is constructed from scratch. The approach used here is similar to the Kmeans clustering functions developed by `sklearn`. 

Kmeans clustering is performed using **Within Cluster Sum of Squares(WCSS)**. Initially, each data point is assigned to a cluster randomly. The centroid is computed for each cluster. The distance between all the data points and centroids are computed. Each data point is then reassigned to a cluster where the distance is minimum to a given centroid. After all the data points are reassigned, WCSS is computed. Lower the WCSS, better the algorithm has separated the data. 