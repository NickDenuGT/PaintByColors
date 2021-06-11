import cv2
import numpy as np
import matplotlib.pyplot as plt
# For Google Colab we use the cv2_imshow() function
from google.colab.patches import cv2_imshow
import random
import math
import copy
import sys

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
LABELINGCHUNKSIZE = 20

BLACK = (0,0,0)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.3
font_color = BLACK
font_thickness = 1

FileName = sys.argv[1]

# Loading our image with a cv2.imread() function
img=cv2.imread(FileName,cv2.IMREAD_COLOR)
img_original=cv2.imread(FileName,cv2.IMREAD_COLOR)
Height, Width, c = img.shape

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cluster_smooth_img = ndimage.gaussian_filter(img, sigma=(5, 5, 0))

Pixel_Values = cluster_smooth_img.reshape(-1,3)

Pixel_Values = np.float32(Pixel_Values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.1)

_, labels, (centers) = cv2.kmeans(Pixel_Values, NumClusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)

labels = labels.flatten()

min_width = 3

csc_img = centers[labels.flatten()]

csc_img = csc_img.reshape(img.shape)

def GetLabel(img, x, y):
    return labels[x * img.shape[0] + y]

# x and y cannot be on the edge of the image
def IsBorder(img, x, y):
    if x == 0 or y == 0 or x == img.shape[0] or y == img.shape[1]:
        return False
        # You're on the edge...of glory
    #elif ((not img[x][y][0] == img[x + 1][y][0]) or (not img[x][y][0] == img[x][y - 1][0])) and ((not img[x][y][1] == img[x + 1][y][1]) or (not img[x][y][1] == img[x][y - 1][1])) and ((not img[x][y][2] == img[x + 1][y][2]) or (not img[x][y][2] == img[x][y - 1][2])):
    elif (not GetLabel(img, x, y) == GetLabel(img, x + 1, y)) or (not GetLabel(img, x, y) == GetLabel(img, x, y - 1)):
        return True
    else:
        return False

def DrawBorders(img):

    ret_img = copy.deepcopy(img)
    for x in range(img.shape[0] - 1):
        for y in range(img.shape[1] - 1):
            if IsBorder(img, x, y) == True:
                ret_img[x][y] = (0,0,0)
            else:
                ret_img[x][y] = (255,255,255)
    return ret_img

border_img = DrawBorders(csc_img)

def GroupSharesLabel(img, x, y):
    val = GetLabel(img, x, y)
    for i in range(LABELINGCHUNKSIZE):
        for j in range(LABELINGCHUNKSIZE):
            if not GetLabel(img, x + i, y + j ) == val:
                return False
    return True

def PrintLabel(img, x, y):
    text = str(GetLabel(img, x, y))
    return cv2.putText(img, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

def LabelDrawing(img):
    ret_img = copy.deepcopy(img)
    for x in range(int(round(img.shape[0] / LABELINGCHUNKSIZE)) - 1):
        for y in range(int(round(img.shape[1] / LABELINGCHUNKSIZE)) - 1):
            if GroupSharesLabel(img, x * LABELINGCHUNKSIZE, y * LABELINGCHUNKSIZE) == True:
                ret_img = PrintLabel(ret_img, x * LABELINGCHUNKSIZE + int(round(LABELINGCHUNKSIZE / 2)), y * LABELINGCHUNKSIZE + LABELINGCHUNKSIZE - 1)
    return ret_img

label_drawing = LabelDrawing(border_img)

plt.imshow(border_img)
plt.show()
