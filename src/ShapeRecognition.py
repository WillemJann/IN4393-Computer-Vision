import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.io import imread
from skimage.color import rgb2hsv
from skimage.morphology import opening

# Load picture and detect edges
or_image = imread('streetview.jpg')#, as_grey=True)
image = rgb2hsv(or_image)

# Red filter constraints
h_red = np.logical_or(image[:,:,0] >= float(240)/360, \
                      image[:,:,0] <= float(10)/360)
s_red = image[:,:,1] >= float(40)/360
v_red = image[:,:,2] >= float(30)/360
binary_red = np.logical_and(h_red, np.logical_and(s_red, v_red))

# Blue filter constraints
h_blue = np.logical_and(image[:,:,0] > float(210)/360, \
                        image[:,:,0] <= float(230)/360)
s_blue = image[:,:,1] >= float(127.5)/360
v_blue = image[:,:,2] >= float(20)/360
binary_blue = np.logical_and(h_blue, np.logical_and(s_blue, v_blue))

# White / Achromatic pixels
# paper: https://thesai.org/Downloads/Volume7No1/Paper_93-Traffic_Sign_Detection_and_Recognition.pdf
# paper: https://www.researchgate.net/publication/296196586_Ayoub_Ellahyani_Mohamed_El_AnsariIlyas_El_Jaafari_Traffic_sign_detection_and_recognition_based_on_random_forests_Applied_Soft_Computing
# TODO: Discard ROIs based on size and aspect ratio constraints for traffic signs containing white areas
D = 20
a_image = or_image.astype(int)
achromatic = (np.absolute(a_image[:,:,0] - a_image[:,:,1]) + \
              np.absolute(a_image[:,:,1] - a_image[:,:,2]) + \
              np.absolute(a_image[:,:,2] - a_image[:,:,0]) ) / (2.5*D)
achromatic = achromatic < 1.0


binary = np.logical_or(binary_blue, binary_red)
binary = opening(binary)

''''
# paper: https://hal.archives-ouvertes.fr/hal-00658086/document
# Possible inspiration: https://github.com/glemaitre/traffic-sign-detection/blob/master/src/img_processing/colorConversion.cpp
# Log Chromatic 
R_c = image[:,:,0] / (image[:,:,0]+image[:,:,1]+image[:,:,2])
G_c = image[:,:,1] / (image[:,:,0]+image[:,:,1]+image[:,:,2])
B_c = image[:,:,2] / (image[:,:,0]+image[:,:,1]+image[:,:,2])
lccs_RG = np.log(R_c / G_c)
lccs_BG = np.log(B_c / G_c)

red_thresholds = {'RG':(0.5, 2.1), 'BG': (-0.9, 0.8)}
RG_thres = np.logical_and(red_thresholds['RG'][0] <= lccs_RG, lccs_RG <= red_thresholds['RG'][1])
BG_thres = np.logical_and(red_thresholds['BG'][0] <= lccs_BG, lccs_BG <= red_thresholds['BG'][1])

binary = np.logical_and(RG_thres, BG_thres)
'''

#edges = canny(binary, sigma=2.3)#, sigma=3, low_threshold=50, high_threshold=150)



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(10, 4), sharey=True,
                                subplot_kw={'adjustable':'box-forced'})

ax1.set_title('red')
ax1.imshow(binary_red, cmap=plt.cm.gray)

ax2.set_title('blue')
ax2.imshow(binary_blue, cmap=plt.cm.gray)

ax3.set_title('White areas')
ax3.imshow(achromatic, cmap=plt.cm.gray)

ax4.set_title('Combined red and blue')
ax4.imshow(binary, cmap=plt.cm.gray)

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