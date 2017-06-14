import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.io import imread
from skimage.color import rgb2hsv
from skimage.morphology import binary_dilation, square, binary_closing, binary_opening

import matplotlib.patches as mpatches
from skimage.measure import label, regionprops
from skimage.color import label2rgb

# Load picture and detect edges
or_image = imread('streetview.jpg')#, as_grey=True)
image = rgb2hsv(or_image)

# Red filter constraints
h_red = np.logical_or(image[:,:,0] >= float(240)/360, \
                      image[:,:,0] <= float(10)/360)
s_red = image[:,:,1] >= float(40)/360
v_red = image[:,:,2] >= float(30)/360
binary_red = np.logical_and(h_red, np.logical_and(s_red, v_red))
binary_red = binary_opening(binary_red)


# Blue filter constraints
h_blue = np.logical_and(image[:,:,0] > float(210)/360, \
                        image[:,:,0] <= float(230)/360)
s_blue = image[:,:,1] >= float(127.5)/360
v_blue = image[:,:,2] >= float(20)/360
binary_blue = np.logical_and(h_blue, np.logical_and(s_blue, v_blue))
binary_blue = binary_opening(binary_blue)


# White / Achromatic pixels
# paper: https://thesai.org/Downloads/Volume7No1/Paper_93-Traffic_Sign_Detection_and_Recognition.pdf
# paper: https://www.researchgate.net/publication/296196586_Ayoub_Ellahyani_Mohamed_El_AnsariIlyas_El_Jaafari_Traffic_sign_detection_and_recognition_based_on_random_forests_Applied_Soft_Computing
# TODO: Discard ROIs based on size and aspect ratio constraints for traffic signs containing white areas
D = 17
a_image = or_image.astype(int)
achromatic = (np.absolute(a_image[:,:,0] - a_image[:,:,1]) + \
              np.absolute(a_image[:,:,1] - a_image[:,:,2]) + \
              np.absolute(a_image[:,:,2] - a_image[:,:,0]) ) / (3*D)
achromatic = achromatic < 1.0


binary = np.logical_or(binary_blue, binary_red)
binary = binary_closing(binary)
binary = binary_dilation(binary, square(5))

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
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(10, 4), sharey=True,
                                subplot_kw={'adjustable':'box-forced'})

#edges = canny(binary, sigma=2.3)#, sigma=3, low_threshold=50, high_threshold=150)

# label image regions
labeled = label(binary)
image_label_overlay = label2rgb(labeled, image=or_image)
patches = []
for region in regionprops(labeled):
    # take regions with large enough areas
    bbox = region.bbox
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width >= 20 and height >= 20:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax4.add_patch(rect)
        patches.append(or_image[minr:maxr, minc:maxc, :])


ax1.set_title('red')
ax1.imshow(binary_red, cmap=plt.cm.gray)

ax2.set_title('blue')
ax2.imshow(binary_blue, cmap=plt.cm.gray)

ax3.set_title('Combined red and blue')
ax3.imshow(binary, cmap=plt.cm.gray)

ax4.set_title('labels')
ax4.imshow(image_label_overlay)
plt.show()

fig2, axis = plt.subplots(ncols=len(patches), nrows=1, figsize=(10, 4),
                                subplot_kw={'adjustable':'box-forced'})

axs = axis.ravel()
for i in range(len(patches)):
	axs[i].set_title('patch: %d' % i)
	axs[i].imshow(patches[i])
plt.show()