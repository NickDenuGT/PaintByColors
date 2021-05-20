import cv2
import numpy as np
import matplotlib.pyplot as plt
# For Google Colab we use the cv2_imshow() function
#from google.colab.patches import cv2_imshow
import random
import math
import copy
import numpy

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
        if Dist(i, j, clus_pt) < dist:
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



NumClusters = 25
FileName = "Tree.jpg"

# Loading our image with a cv2.imread() function
img=cv2.imread(FileName,cv2.IMREAD_COLOR)
img_original=cv2.imread(FileName,cv2.IMREAD_COLOR)
Height, Width, c = img.shape
print(Height)
print(Width)
# img=cv2.imread("Cybertruck.jpg",1)

ClusterLists = []
ClusterPointsCoords = []
#Select Random Points across the images
for x in range(NumClusters):
    ClusterHead = []
    x_coord = random.randint(0, Height)
    y_coord = random.randint(0, Width)
    ClusterHead.append((x_coord, y_coord))
    ClusterPointsCoords.append((x_coord, y_coord))
    ClusterLists.append(ClusterHead)

print(ClusterPointsCoords)

ClusterMap = []
#For each point, look for closest cluster Points
for i in range(Width):
    for j in range(Height):
        ClusterIndex = FindClusterToJoin(i - 1, j - 1, img, (Height * Width) * 10 / NumClusters, ClusterPointsCoords)
        ClusterLists[ClusterIndex].append((j, i))
        ClusterMap.append(ClusterIndex)



#for each cluster, find the average COLOR
ClusterColors = []
for ClusterList in ClusterLists:
    total_b = 0
    total_g = 0
    total_r = 0
    count = 0
    for coord in ClusterList:
        (b, g, r) = img[coord[0] - 1, coord[1] - 1]
        total_b += b
        total_g += g
        total_r += r
        count += 1
    ClusterColors.append(((total_b) / float(count), ((total_g) / float(count)), ((total_r) / float(count))))

#Sort clusters by color
for i in range(Width):
    for j in range(Height):
        ClusterNumber = ClusterMap[(i - 1) * Width + j - 1]
        img[i, j] = [(ClusterColors[ClusterNumber][0]), (ClusterColors[ClusterNumber][1]), (ClusterColors[ClusterNumber][2])]



cv2.imshow("img", img)
cv2.imshow("img_original", img_original)
cv2.waitKey(0)
cv2.destroyAllWindows()



#img[0:150, 0:300] = [0,255,0]

# Loading our image with a cv2.imread() function
#gray=cv2.imread("Cathy.jpg",cv2.IMREAD_GRAYSCALE)
# gray=cv2.imread("Cybertruck.jpg",0)

# For Google Colab we use the cv2_imshow() function
# but we can use cv2.imshow() if we are programming on our computer

'''
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imshow("gray",gray)
# We can show the image using the matplotlib library.
# OpenCV loads the color images in reverse order:
# so it reads (R,G,B) like (B,G,R)
# So, we need to flip color order back to (R,G,B)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# We can use comand plt.axis("off") to delete axis
# from our image
plt.title('Original')

# If we want to get the dimensions of the image we use img.shape
# It will tell us the number of rows, columns, and channels
dimensions = img.shape
print(dimensions)

# If an image is a grayscale, img.shape returns
#the number of rows and columns
dimensions = gray.shape
print(dimensions)

# We can obtain a total number of elements by using img.size
total_number_of_elements= img.size
print(total_number_of_elements)

# To get the value of the pixel (x=50, y=50), we would use the following code
(b, g, r) = img[50, 50]
print("Pixel at (50, 50) - Red: {}, Green: {}, Blue: {}".format(r,g,b))

# We changed the pixel color to red
img[50, 50] = (0, 0, 255)

# Displaying updated image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Updated')

'''
