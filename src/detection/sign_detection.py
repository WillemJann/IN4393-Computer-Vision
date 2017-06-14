import io
import skimage.io

import numpy as np
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.io import imread
from skimage.color import rgb2hsv
from skimage.morphology import binary_dilation, square, binary_closing, binary_opening
from skimage.draw import line
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches

def detect_signs(image):
	# TODO: implement sign detection

	binary_segmentation = binary_filtering(image)
	ROIs = extract_ROIs(binary_segmentation)
	# TO DO, recognize the actual shape of the sign.
	annotated = visualize_ROIs(ROIs, image)

	# Return results
	#im = image
	#im[:,:,0] = binary_segmentation * image[:,:,0]
	#im[:,:,1] = binary_segmentation * image[:,:,1]
	#im[:,:,2] = binary_segmentation * image[:,:,2]
	output = io.BytesIO()
	skimage.io.imsave(output, annotated, plugin='pil', format_str='jpeg')
	output.seek(0)

	recognized_signs = {}

	return output, recognized_signs


def binary_filtering(image):
	hsv = rgb2hsv(image)

	# Red filter constraints
	h_red = np.logical_or(hsv[:,:,0] >= float(240)/360, \
						  hsv[:,:,0] <= float(10)/360)
	s_red = hsv[:,:,1] >= float(40)/360
	v_red = hsv[:,:,2] >= float(30)/360
	binary_red = np.logical_and(h_red, np.logical_and(s_red, v_red))
	binary_red = binary_opening(binary_red)

	# Blue filter constraints
	h_blue = np.logical_and(hsv[:,:,0] > float(210)/360, \
							hsv[:,:,0] <= float(230)/360)
	s_blue = hsv[:,:,1] >= float(127.5)/360
	v_blue = hsv[:,:,2] >= float(20)/360
	binary_blue = np.logical_and(h_blue, np.logical_and(s_blue, v_blue))
	binary_blue = binary_opening(binary_blue)

	# White / Achromatic pixels
	# paper: https://thesai.org/Downloads/Volume7No1/Paper_93-Traffic_Sign_Detection_and_Recognition.pdf
	# paper: https://www.researchgate.net/publication/296196586_Ayoub_Ellahyani_Mohamed_El_AnsariIlyas_El_Jaafari_Traffic_sign_detection_and_recognition_based_on_random_forests_Applied_Soft_Computing
	# TODO: Discard ROIs based on size and aspect ratio constraints for traffic signs containing white areas
	D = 17
	a_image = image.astype(int)
	achromatic = (np.absolute(a_image[:,:,0] - a_image[:,:,1]) + \
				  np.absolute(a_image[:,:,1] - a_image[:,:,2]) + \
				  np.absolute(a_image[:,:,2] - a_image[:,:,0]) ) / (3*D)
	achromatic = achromatic < 1.0


	binary = np.logical_or(binary_blue, binary_red)
	binary = binary_closing(binary)
	binary = binary_dilation(binary, square(5))

	return binary

def extract_ROIs(binary_image):
	# label image regions
	labeled = label(binary_image)
	#image_label_overlay = label2rgb(labeled, image=or_image)
	patches = []
	for region in regionprops(labeled):
		# take regions with large enough areas
		bbox = region.bbox
		width = bbox[2] - bbox[0]
		height = bbox[3] - bbox[1]
		if width >= 20 and height >= 20:
			# draw rectangle around segmented coins
			minr, minc, maxr, maxc = bbox
			#rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
			#                          fill=False, edgecolor='red', linewidth=2)

			#ax4.add_patch(rect)
			patches.append( (minr,maxr-1,minc,maxc-1) )
	return patches

def visualize_ROIs(patches, image):
	for patch in patches:
		minr, maxr, minc, maxc = patch
		draw_box(image, minr,maxr,minc,maxc)

def draw_box(image, minr, maxr, minc, maxc):
	print(image.shape)
	print('rows: %d to %d' % (minr, maxr))
	print('cols: %d to %d' % (minc, maxc))

	top_line = line(minr,maxr, minc, minc)
	image[top_line[0],top_line[1], :] = 0
	image[top_line[0],top_line[1], :] = 1

	right_line = line(maxr, maxr, minc, maxc)
	image[right_line[0],right_line[1],:] = 0
	image[right_line[0],right_line[1],2] = 1

	bottom_line = line(minr,maxr,maxc,maxc)
	image[bottom_line[0],bottom_line[1],:] = 0
	image[bottom_line[0],bottom_line[1],2] = 1

	left_line= line(minr,minr, minc,maxc)
	image[left_line[0],left_line[1],:] = 0
	image[left_line[0],left_line[1],2] = 1
