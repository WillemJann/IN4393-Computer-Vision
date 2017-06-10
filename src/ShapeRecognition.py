import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.io import imread
from skimage.color import rgb2hsv

# Load picture and detect edges
image = imread('streetview.jpg')#, as_grey=True)
image = rgb2hsv(image)
binary = np.logical_or(image[:,:,0] < 0.36, \
			np.logical_or(image[:,:,0] > 0.88, \
				np.logical_and(image[:,:,0] > 0.61,image[:,:,0] < 0.72))) 

edges = canny(binary, sigma=2.3)#, sigma=3, low_threshold=50, high_threshold=150)



fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 4), sharey=True,
                                subplot_kw={'adjustable':'box-forced'})

ax1.set_title('edges')
ax1.imshow(edges, cmap=plt.cm.gray)

ax2.set_title('HSV threshold')
ax2.imshow(binary, cmap=plt.cm.gray)
plt.show()
'''
# Detect two radii
hough_radii = np.arange(20,70)
hough_res = hough_circle(edges, hough_radii)

# Select the most prominent 5 circles
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=5)

# Draw them
image = color.gray2rgb(image)
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius)
    image[circy, circx] = (220, 20, 20)

fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 4), sharey=True,
                                subplot_kw={'adjustable':'box-forced'})

ax1.set_title('Image + circles')
ax1.imshow(image, cmap=plt.cm.gray)

ax2.set_title('Edges (white)')
ax2.imshow(edges, cmap=plt.cm.gray)
plt.show()
'''
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

path = 'streetview.png'
print("Loading Image: %s" % path)
# Load an color image in grayscale
img = cv2.imread(path,0)
img = cv2.medianBlur(img,3)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)


print("Run HoughCircles function")
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,dp=1,minDist=10,
                            param1=50,param2=40,minRadius=10,maxRadius=50)

print("Draw circles on top of processed image")
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

print("Show processed image with circles")
plt.imshow(cimg, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
'''