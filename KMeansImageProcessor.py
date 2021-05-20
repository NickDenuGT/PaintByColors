import cv2
import numpy as np
import matplotlib.pyplot as plt
# For Google Colab we use the cv2_imshow() function
from google.colab.patches import cv2_imshow
import random
import math
import copy

from scipy import ndimage

def Dist(i, j, clus_pt):
    return math.sqrt(pow(i - clus_pt[0], 2) + pow(j - clus_pt[1], 2))

def ColorDiff(color1, color2):
    return math.sqrt(pow(int(color1[0]) - int(color2[0]), 2) +
     pow(int(color1[1]) - int(color2[1]), 2) +
     pow(int(color1[2]) - int(color2[2]), 2))

def FindClusterToJoin(i, j, img, dist, clus_pts):
    indices = []
    counter = 0
    for clus_pt in clus_pts:
        if Dist(i, j, clus_pt) < dist and ColorDiff(img[i - 1, j - 1], img[clus_pt[0] - 1, clus_pt[1] - 1]):
            indices.append(counter)
            counter = counter + 1
    Height, Width, c = img.shape
    MinDiff = Height * Width
    Ret_Cluster = -1
    for index in indices:
        Diff = ColorDiff(img[i - 1, j - 1], img[clus_pts[index - 1][0] - 1, clus_pts[index - 1][1] - 1])
        if MinDiff > Diff:
            MinDiff = Diff
            Ret_Cluster = index

    return Ret_Cluster

NumClusters = 10
FileName = "ManinderParty.jpg"

# Loading our image with a cv2.imread() function
img=cv2.imread(FileName,cv2.IMREAD_COLOR)
img_original=cv2.imread(FileName,cv2.IMREAD_COLOR)
Height, Width, c = img.shape

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cluster_smooth_img = ndimage.gaussian_filter(img, sigma=(3, 3, 0))

Pixel_Values = cluster_smooth_img.reshape(-1,3)

Pixel_Values = np.float32(Pixel_Values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.1)

_, labels, (centers) = cv2.kmeans(Pixel_Values, NumClusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)

labels = labels.flatten()

min_width = 3

csc_img = centers[labels.flatten()]

csc_img = csc_img.reshape(img.shape)

def DrawBorders(img):

    ret_img = copy.deepcopy(img)
    for x in range(img.shape[0] - 1):
        for y in range(img.shape[1] - 1):
            if not img[x][y] == img[x + 1][y] or not img[x][y] == img[x][y + 1]:
                ret_img[x][y] = (0,0,0)
            else:
                ret_img[x][y] = (155,155,155)
    return ret_img

border_img = DrawBorders(csc_img)

plt.imshow(border_img)
plt.show()
