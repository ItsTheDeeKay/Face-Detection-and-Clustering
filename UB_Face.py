'''
Notes:
1. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
2. If you want to show an image for debugging, please use show_image() function in helper.py.
'''
# Author: DeeKay Goswami

import os
import cv2
import sys
import math
import numpy as np
import face_recognition
from utils import show_image
from typing import Dict, List

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: np.ndarray) -> List[List[float]]:
    """
    Args:
        img : input image is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    detection_results: List[List[float]] = [] # Please make sure your output follows this data format.
    global face_Algorithm
    global scale_Factor
    global neighbors
    global faces
    
    # This will assign parameters to detect faces...
    scale_Factor = 1.089
    neighbors = 3

    # This will load haarcascade library for face detection...
    face_Algorithm_Path = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
    face_Algorithm = cv2.CascadeClassifier(face_Algorithm_Path)

    # This detects faces in the image and returns float data of rectangle box...
    faces = face_Algorithm.detectMultiScale(img, scale_Factor, neighbors)
    sorted_Faces = list(faces)
    floated_List = np.array(sorted_Faces, dtype=float)
    detection_results = floated_List.tolist()
    # print(detection_results)

    return detection_results


def cluster_faces(imgs: Dict[str, np.ndarray], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    cluster_results: List[List[str]] = [[]] * K # Please make sure your output follows this data format.
    
    # Add your code here. Do not modify the return and input arguments.
    
    global face_Algorithm
    global all_Encodings
    global scale_Factor
    global images_Keys
    global final_List
    global neighbors
    global images
    global face
    global i_K

    iterations = 31
    images = imgs
    i_K = int(K)
    final_List = []
    all_Encodings = []
    
    # This will compute dimensions of the incoming dictionary to later 
    # understand the flow of the dictionary...
    dim1 = len(imgs)
    dim2 = 0
    for d in imgs:
        dim2 = max(dim2, len(d))
    print("Dimension of the dictionary:- " , dim1,",", dim2)

    # This will assign parameters to detect faces...
    scale_Factor = 1.089
    neighbors = 3
    face_Algorithm = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

    # This will sequentially extract all the keys present in the dictionary...
    images_Keys = list(images.keys())

    # This will iterate all the keys in the dictionary to extract value
    # associated with it...
    for key in images:
        iterate_Image = images[key]
        # print(key)
        face = face_Algorithm.detectMultiScale(iterate_Image, scale_Factor, neighbors)
        # print(iterate_Image)

    # This loop is used to look up rectangular data in face area and assigns
    # for face encoding...
        for (x, y, w, h) in face:
            boxes = [(y,(x+w),(y+h),x)]

            # This will return 128 dimension array for each 
            # face present in the image ‘iterate_Image’...
            encoding = face_recognition.face_encodings(iterate_Image, boxes)
            all_Encodings.append(encoding)

    # This will convert all the encodings data of the images to array...
    all_Encodings = np.array(all_Encodings)

    # This will pass all encodings data into centroids function and 
    # returns centroids data of all the images...
    centroids = centroids_Algorithm(all_Encodings, i_K)

    # This will return cluster images sequence by Kmeans++ algorithm...
    clustered = k_Algorithm(all_Encodings, i_K, centroids, iterations)
    # print(clustered)
    cluster_results = assign_Cluster(clustered)
    print(cluster_results)
    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# Your functions. (if needed)

# This function is used to append cluster images to its 
# appropriate image keys...
def assign_Cluster(AC):
    for i in range(i_K):
        assign_Key = []
        for j, val in enumerate(AC):
            if val == i:
                assign_Key.append(images_Keys[j])
        final_List.append(assign_Key)
    # print(final)
    return final_List  

# Euclidean distance b/w 2 points
def Euclidean_distance(a, b):
    return math.sqrt(np.sum((a - b)**2))

# Getting centroids for K-Means++ algorithm
def centroids_Algorithm(data, k):
    centroids = []

    # This will randomly choose first centroid...
    centroids.append(data[np.random.randint(data.shape[0]), :])

    # This will Loop through for given iterations...
    for ind in range(k - 1):
        distance = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize

            # This will loop through all the centroids data...
            for j in range(len(centroids)):
                # Calculate euclidean distance b/w centroid & point pair
                temp_dist = Euclidean_distance(point, centroids[j])
                d = min(d, temp_dist)

            # This will append the best distance...
            distance.append(d)

        # This will convert to a 'numpy' array...
        distance = np.array(distance)

        # This will choose the farthest point as the next centroid...
        next_centroid = data[np.argmax(distance), :]

        centroids.append(next_centroid)

    return centroids

# This function is used to calculate distance between points & centroids...
def Centroids_distance(x, y, eu):
    gap = []

    # This will loop through all centroids & points and calculate distance b/w a given pair...
    for i in range(len(x)):
        for j in range(len(y)):
            d = x[i][0]-y[j][0]
            d = np.sum(np.power(d, 2))
            gap.append(d)
    gap = np.array(gap)
    gap = np.reshape(gap, (len(x), len(y)))

    # This will return distance matrix (between points & centroids)...
    return gap


# Main K-Means++ Function
def k_Algorithm(x, k, cent, iter):

    centroids = cent

    # This will compute matrix of distance between centroids & points...
    dist_matrix = Centroids_distance(x, centroids, 'euclidean')

    # This will get the nearest centroid (class) for the image...
    image_class = np.array([np.argmin(d) for d in dist_matrix])

    # This will loop through the number of iterations passed while calling this function...
    for i in range(iter):
        centroids = []
        for j in range(k):

            # This will compute all encodings for particular class, 
            # and add & find the mean to get a new centroid...
            new_cent = x[image_class == j]
            ms = 0
            for l in range(len(new_cent)):
                ms += new_cent[l]
            new_cent = np.divide(ms, len(new_cent))
            centroids.append(new_cent)
        dist_matrix = Centroids_distance(x, centroids, 'euclidean')
        image_class = np.array([np.argmin(d) for d in dist_matrix])
    # print(image_class)
    
    return image_class
