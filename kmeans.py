import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets.samples_generator import make_blobs


def initialize_centroids(X,k):
    """
    randomly picks k unique points from X as the initial centroids
    
    inputs--
    X: input matrix--> np.ndarray
    k: number of centroids
    
    outputs--
    array of centroid vectors
    """ 
    return X[np.random.choice(range(X.shape[0]), k, replace=False)]



def distance_bw_centroids(old:np.ndarray,new:np.ndarray):
    """returns euclidian distance between old and new centroids"""
    return np.linalg.norm(old.ravel()-new.ravel())  


def assign_buckets(X:np.ndarray, centroids:np.ndarray):
    """
    returns a np.array of bucket indices
    for a given set of points and centroids
    """
    dist_matrix = np.stack([np.linalg.norm(X-c, axis=1) for c in centroids])
    return np.argmin(dist_matrix, axis=0)


def kmeans(X:np.ndarray, k:int, centroids=None, tolerance=1e-2):
    """
    Inputs :
        X:Dataset array
        k:number of clusters
        centroids: None for random initialization, 'kmeans++' for smart initialization
        tolerance: max tolerable distance between new and old centroids
    Outputs :
        array of centroids (k,X.shape[1])
        array of cluster indices corresponding to each data-point (X.shape[0])
    """
    if centroids=='kmeans++':
        centroids = initialize_centroids_plus(X,k)
    else:
        centroids = initialize_centroids(X,k)
    
    # for values of k>20, it takes a very long time to converge
    # so we are putting a limit on iterations such that if k increases, number of iterations decrease
    # also we can have more iterations for smaller values of k, which will yield a better set of centroids
    iter_limit = int(800/k)
    iters = 0
    d = 5    
    while (d > tolerance) and (iters< iter_limit):
        bucket_idx = assign_buckets(X, centroids)
        new_centroids = [np.mean(X[bucket_idx == i], keepdims = True, axis = 0 ) for i in range(k)]
        new_centroids = np.array(new_centroids).ravel().reshape(k,X.shape[1])
        old_centroids = centroids
        centroids = new_centroids
        d = distance_bw_centroids(old_centroids, new_centroids)
        iters+=1
    return centroids,bucket_idx


# ### TOY Dataset with Random Initialization

#creating dataset
centers = [(-5, -5), (5, 5), (-5,5),(5,-5), (0,0)]
cluster_std = [0.8, 1, 1.5,1, 1]

X, y = make_blobs(n_samples=400, cluster_std=cluster_std, centers=centers, n_features=4, random_state=1)

#initial centroids
centroids = initialize_centroids(X,5)

plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", s=10, label="Cluster1")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="yellow", s=10, label="Cluster2")
plt.scatter(X[y == 2, 0], X[y == 2, 1], color="green", s=10, label="Cluster3")
plt.scatter(X[y == 3, 0], X[y == 3, 1], color="orange", s=10, label="Cluster4")
plt.scatter(X[y == 4, 0], X[y == 4, 1], color="c", s=10, label="Cluster5")
plt.plot(centroids[:,0],centroids[:,1],"bs")
plt.plot()


# ### After applying kmeans with Random Initialization

center_kmeans, buckets_kmeans  = kmeans(X,k=5)

plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", s=10, label="Cluster1")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="yellow", s=10, label="Cluster2")
plt.scatter(X[y == 2, 0], X[y == 2, 1], color="green", s=10, label="Cluster3")
plt.scatter(X[y == 3, 0], X[y == 3, 1], color="orange", s=10, label="Cluster4")
plt.scatter(X[y == 4, 0], X[y == 4, 1], color="c", s=10, label="Cluster5")
plt.plot(center_kmeans[:,0],center_kmeans[:,1],"bs")


# ## KMEANS++

def initialize_centroids_plus(X:np.ndarray,k:int):
    """
    implementing kmeans++ initialization
    input : X,k
    output : returns k smartly selected centroids
    """
    centroids = initialize_centroids(X,1)
    for _ in range(k-1):
        dist_matrix = np.stack([np.linalg.norm(X-c, axis=1) for c in centroids]) #centroids*X.shape[0]
        min_point_to_centroids = np.min(dist_matrix, axis=0)
        new_centroid_idx = np.argmax(min_point_to_centroids)
        centroids = np.vstack((centroids,X[new_centroid_idx]))
    return centroids


# ### TOY dataset the kmeans++ initialization

centroids = initialize_centroids_plus(X,5)

plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", s=10, label="Cluster1")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="yellow", s=10, label="Cluster2")
plt.scatter(X[y == 2, 0], X[y == 2, 1], color="green", s=10, label="Cluster3")
plt.scatter(X[y == 3, 0], X[y == 3, 1], color="orange", s=10, label="Cluster4")
plt.scatter(X[y == 4, 0], X[y == 4, 1], color="c", s=10, label="Cluster5")
plt.plot(centroids[:,0],centroids[:,1],"bs")
plt.plot()

Comparing kmeans++ initialization to random initialization, we clearly see that inital locations of the centroids are much closer to their soon to be optimal positions
# ## MODEL EVALUATION

# ### Elbow Plot

def sse(X, k):
    center_kmeans, buckets_kmeans  = kmeans(X,k,centroids='kmeans++')
    sum_err = 0
    for i in range(k):
        sum_err += np.linalg.norm(X[buckets_kmeans==i]-center_kmeans[i])
    return sum_err    

x_axis, y_axis = [],[]
for k in range(1,20):
    x_axis.append(k)
    y_axis.append(sse(X,k))

plt.plot(x_axis, y_axis)
plt.xlabel("k")
plt.ylabel("sse")
plt.title("Elbow Plot")
    


# ## IMAGE COMPRESSION

def compressor(pic:np.ndarray,k:int):
    """
    inputs : 
        pic : original image matrix of the form (h,w,c)
        k   : number of clusters
    output : compressed image
    """
    if len(pic.shape)==2: #greyscale images
        h,w = pic.shape
        c =1
    else:
        h,w,c = pic.shape
    pic = pic.reshape(h*w,c)
    centroids, buckets_idx = kmeans(pic,k=k, centroids='kmeans++')
    new_img_arr = centroids[buckets_idx].astype('uint8')
    if c==1:
        new_img_matrix = new_img_arr.reshape(h,w)
    else:
        new_img_matrix = new_img_arr.reshape(h,w,c)
    return Image.fromarray(new_img_matrix)


#image import
im = Image.open('messi.jpg')
pic_array = np.asarray(im)
h,w,c = pic_array.shape
uniq_colors = len(np.unique(pic_array.reshape(h*w,c)))
print(f"number of unique colours in this picture is {uniq_colors}")
im

#compressed image
compressed_pic = compressor(pic_array,20)
pic_array = np.asarray(compressed_pic)
h,w,c = pic_array.shape
uniq_colors = len(np.unique(pic_array.reshape(h*w,c)))
print(f"number of unique colours in this picture is {uniq_colors}")
compressed_pic


#grey scale images
im = Image.open('north-africa-1940s-grey.png')
pic_array = np.asarray(im)
im

compressed_pic = compressor(pic_array,4)
compressed_pic

