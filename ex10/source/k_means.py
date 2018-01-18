import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans

def k_means_image(fname, n_clusters):
    '''Function which clusters the colors of an image.'''
    
    img = plt.imread(fname, n_clusters)
    img_flat = np.reshape(img, (-1, 3))
    # create a grid which indexes pixels
    img_grid = np.arange(img.shape[0] * img.shape[1]).reshape((img.shape[0],-1))
    
    # perform clustering in color space
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img_flat)
    
    # assign each pixels to it's cluster
    img_centroids = np.reshape(kmeans.labels_, (img.shape[0], img.shape[1]))
    
    return kmeans, img_grid, img_centroids
    
    
if __name__ == '__main__':
    k, grid, labels = k_means_image('grass2.jpg', 2)

    plt.figure()
    plt.imshow(labels)
    plt.title('2 clusters')
    plt.savefig('Figure_4.png')
    
