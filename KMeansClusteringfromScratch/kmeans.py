import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123456789)


class KMeansClustering:
    """
    KMeans Clustering algorithm takes X(data) and k(number of clusters) as inputs.
    """
    def __init__(self, X, k):
        self.X = X
        self.k = k

    def euclidean_distance(self, data,centroid):
        """
        Calculates euclidean distance bewteen each data point and a centroid.
        input:
        data: numpy.array
              Data for which distance to be calculated
        centroid: numpy.array
              A vector likw numpy array that represents centroid of a cluster
        
        Returns: numpy.array
        L2 norm distance (Elcuidean distance)
        """
        distances = []
        for i in range(data.shape[0]):
            distances.append(np.sqrt(np.sum(np.square((data[i,:]-centroid))))) # calculate L2 norm (euclidean distance)
        return np.array([distances]) # returns list distances between each data point to the given centroid

    def get_centroid(self, data):
        """
        To calculate the centroid of a cluster given data
        
        Inputs:
        data: numpy.array
            Data for whcih centroid has to be calculated
        Returns:
        centroid: float
        """
        return np.mean(data, axis=1) # mean point of the given data which represents the centroid of the given data

    def calculate_wcss(self, X, clusters):
        """
        The evaluation metric of kmeans clustering - Within Cluster Sum of Squares (WCSS). 
        Used to understand if the kmeans algorithm has achieved local minima. 
        Lower the wcss, better the algorithm has achieved spearability in data
        input:
            X: np.array
            Data given to Kmeans clustering algorithm
            clusters: list
            List of cluster ids to which each data point belong to. 
        return:
            wcss: float
            The calculated within cluster sum of suqares of all the clusters.
        """
        wcss = 0
        for cluster in clusters:
            data = X[np.where(clusters == cluster),:] # separating data per each cluster
            centroid = self.get_centroid(data) # calculate centroid
            wcss += np.sum(np.square(data - centroid)) # calculate cluster sum and add it to total sum of WCSS
        return wcss

    def kmeans(self, convergence=0.0001, no_iterations=100, explain=False):
        """
        Performs Kmeans clustering algorithm. 
        input: 
            convergence: float
            The convergence point of within cluster sum of squares. 
            no_iteration: int
            Maximum number of iteration the algorithm can perform
        returns:
            clusters: list
            List of cluster value to which the given data belong to
        """
        X = self.X
        k = self.k
        clusters = np.random.choice(k, len(X)) # initializing the clusters randomly to each data point of given data
        centroids = []
        distances = np.empty(shape=(len(X),k)) # initializing a distance array to save distance between centroids to each data. Shape: (nrows(X) x k)
        iter = 0
        self.wcss = []
        self.all_custers = []
        # while current iteration is less than the number of iteration given in the input
        while iter < no_iterations:
            # for each cluster calculate centroid and euclidean distance between each data point to the cluster centroid    
            for clus in np.unique(clusters):
                data = X[np.where(clusters == clus),:] # separate data assigned to each cluster
                centroid = self.get_centroid(data) # calculate centroid
                # print(centroid)
                centroids.append(centroid)
                # calculate euclidean distance between centroid to each data point and add it to the respective distance array
                distances[:,clus] = self.euclidean_distance(X, centroid) 
            # If the distance between centroid and a data point is minimum, the data point is assigned to a cluster with the centroid.
            clusters = np.argmin(distances,axis=1) # Taking the minimum distances and assigning the data points to clusters. This forms new clusters
            curr_wcss = self.calculate_wcss(X, clusters) # calculate wcss
            self.wcss.append(curr_wcss)

            print(f"{iter}: {curr_wcss}")
            if explain:
                self.all_custers.append(clusters)
            # if the difference between wcss value of previous iteration and current iteration is less than convergence given in input,
            # we achieved convergence point and local minima. The while loop breaks and continues to return the output.
            if iter >= 1 and abs(self.wcss[iter] - self.wcss[iter-1]) < convergence:
                self.final_convergence = abs(self.wcss[iter] - self.wcss[iter-1])
                self.final_iternation = iter
                self.clusters = clusters # final cluster ids assigned to each data point
                break
            # error catch: if the convergence is not achieved after the number of iterations given in input, should return error
            # to increase number of iterations
            if iter == no_iterations:
                raise Exception("Kmeans did not reach convergence. Increase the number of iterations")
            iter += 1
            
        return self.clusters
    
    def plot_wcss(self):
        """
        Plot function of within cluster sum of squares vs Iterations
        """
        fig, axs = plt.subplots()
        axs.plot(range(len(self.wcss)), self.wcss)
        axs.set_xlabel("Iterations")
        axs.set_ylabel("Within Custer Sum of Squares")
        axs.set_title("Convergence of Within Cluster Sum of Squares")
        return axs
        # plt.show()
        
    def plot_cluster_data2d(self):
        """
        Generates 2d plot of the data with respect to the cluster it belongs to
        """
        fig, axs = plt.subplots()
        for i in np.unique(self.clusters):
            axs.scatter(self.X[np.where(self.clusters==i),0], self.X[np.where(self.clusters==i),1],label=i)
        axs.set_title("Data divied into clusters")    
        axs.legend()
        return axs
        # plt.show()
        
    
