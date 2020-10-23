import numpy as np
from PIL import Image
import copy

def update_pixel_cluster(K,image,centroids):
    pixel_cluster = []
    cost = 0
    for pixel in image:
        dist_values = []
        for i in range(K):
            euc_dist = np.sqrt(np.power(pixel[0]-centroids[i][0],2) + np.power(pixel[1]-centroids[i][1],2) + np.power(pixel[2]-centroids[i][2],2))
            dist_values.append(euc_dist)
            cost += euc_dist ** 2
        cluster_assignment = np.argmin(dist_values)
        pixel_cluster.append(cluster_assignment)
    print("Cost:" ,cost)
    return pixel_cluster


def update_centroid(centroids, pixel_cluster, image):
    for i in range(K):
        sum_count = 0
        index = 0
        R = 0
        G = 0
        B = 0
        for cluster in pixel_cluster:
            if cluster == i:
                sum_count+=1
                R += image[index][0]
                G += image[index][1]
                B += image[index][2]
                # print(R,G,B)
            index += 1
        if sum_count != 0:
            new_centroid = [R/sum_count, G/sum_count, B/sum_count]
        else:
            new_centroid = centroids[i]
        centroids[i] = new_centroid
    return centroids

if __name__ =='__main__':
    K = 8 # number of clusters
    centroids = [[255, 255, 255],
    [255, 0, 0],
    [128, 0, 0],
    [0, 255, 0],
    [0, 128, 0],
    [0, 0, 255],
    [0, 0, 128],
    [0, 0, 0]] #array of initial centroid values for K clusters
    image = np.genfromtxt('kmeans-image.txt', delimiter = ' ') # data to be clustered, shape: (210012, 3), (w, h = 516, 407)
    # print(image.shape)

    updating = True
    counter = 0
    buffer = 0
    old_centroids = np.zeros(shape=(8,3))
    checker = 0

    print("One moment, this will take some time...")

    while updating:
        counter += 1
        pixel_cluster = update_pixel_cluster(K,image,centroids)
        centroids = update_centroid(centroids, pixel_cluster, image)
        print("Iteration ", counter,": ", centroids)

        #criteria to stop updating
        for i in range(len(centroids)):
            for j in range(3):
                if old_centroids[i][j] == centroids[i][j]:
                    checker += 1
        if checker == 24:
            buffer += 1
            if buffer > 3:
                updating = False
        checker = 0
        old_centroids = copy.deepcopy(centroids)

    #converting all pixels in image to k-means centroid values
    image_to_centroid = []
    for pixel in pixel_cluster:
        image_to_centroid.append(centroids[pixel])
    image_to_centroid = np.array(image_to_centroid)
    image_to_centroid = image_to_centroid.astype(int).reshape(516,407,3).astype(np.uint8)
    print(image_to_centroid.shape)

    img = Image.fromarray(image_to_centroid, 'RGB')
    img.save('k_means.png')
    img.show()