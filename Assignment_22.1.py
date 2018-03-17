#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 22:07:18 2018

@author: macuser
"""

from skimage import io
from sklearn.cluster import KMeans
import numpy as np
import scipy.misc

face = scipy.misc.face()
print(face.shape)   
print(face.max())
io.imshow(face)
io.show()

rows = face.shape[0]
cols = face.shape[1]
face = face.reshape(face.shape[0]*face.shape[1],3)
kmeans = KMeans(n_clusters=5, n_init=10, max_iter=200)
kmeans.fit(face)

clusters = np.asarray(kmeans.cluster_centers_,dtype=np.uint8) 
labels = np.asarray(kmeans.labels_,dtype=np.uint8 )  
labels = labels.reshape(rows,cols); 

np.save('codebook_face.npy',clusters,io.imsave('compressed_face.png',labels))

centers = np.load('codebook_face.npy')
c_image=io.imread('compressed_face.png')

image = np.zeros((c_image.shape[0],c_image.shape[1],3),dtype=np.uint8 )
for i in range(c_image.shape[0]):
    for j in range(c_image.shape[1]):
            image[i,j,:] = centers[c_image[i,j],:]
            
io.imsave('reconstructed_face.png',image)
io.imshow(image)
io.show()