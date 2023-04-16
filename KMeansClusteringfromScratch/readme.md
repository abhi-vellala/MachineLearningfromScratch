# KMeans Clustering from Scratch


In this repository, Kmeans clustering algorithm is constructed from scratch. The approach used here is similar to the Kmeans clustering functions developed by `sklearn`. 

Kmeans clustering is performed using **Within Cluster Sum of Squares(WCSS)**. Initially, each data point is assigned to a cluster randomly. The centroid is computed for each cluster. The distance between all the data points and centroids are computed. Each data point is then reassigned to a cluster where the distance is minimum to a given centroid. After all the data points are reassigned, WCSS is computed. Lower the WCSS, better the algorithm has separated the data. 

## Algorithm

1. $Initialize~ by~ assigning~ each~ data~ point~ to~ a~ cluster~ randomly~ between~ K = [0,...,k]$
2. $while: i \leq iterations:$
    1. $For: j~ in~ k:$
        1. $C_{j}=mean(X)$
        2. $dist_{j}=||X-C_{j}||_{2}$
    2. $K=argmin_{j}(dist_{j})$
    3. $WCSS_{i}=\sum_{j}||X-C_{j}||_{2}$
    4. $if~ abs(WCSS_{i}-WCSS_{i-1}) \leq convergence:$
        1. $break$

### Explanation

1. Initialize by assigning each data point to a cluster randomly.
2. While i $\leq$ total iterations:
    1. For each cluster:
        1. Calculate Centroid of the cluster
        2. Calculate distance between each data point to cluster centroid.
    2. Assign each data point to new clusters based on minimum distance.
    3. Calculate $WCSS_{i}$
    4. if $abs(WCSS_{i}-WCSS{i-1}) \leq convergence$:
        1. break the loop and return results.

## Usage

`kmeans.py` has the `KmeansClustering` class with 2d plot function. In `main.py` file, the class is imported. A 2d data is random generated using uniform distribution. You can generate n dimensional data or can use your custom data. Input parameters, data($X$) and number of clusters you want($k$) to the object and call function `kmeans()` to run Kmeans algorithm on the data. 

````
X = np.random.uniform(low=-100, high=100, size=(1000,2))
print(f"Shape of data: {X.shape}")
k = 4
kmeans = KMeansClustering(X,3)
clusters = kmeans.kmeans(convergence=1e-10)
````

You can call `plot_wcss()` to see how wcss score is changing and algorithm has achieved local minima. You can call `plot_cluster_data2d()` to plot 2d data colored according to clusters. 

````
p1 = kmeans.plot_wcss()
p2 = kmeans.plot_cluster_data2d()
plt.show()
````

## Results

Here are some results:





