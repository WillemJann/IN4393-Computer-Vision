import numpy as np

from skimage.color import rgb2hsv
from skimage.morphology import binary_opening, binary_closing, binary_dilation, square
from skimage.measure import label, regionprops

def binary_filtering(image):
    hsv = rgb2hsv(image)

    # Red filter constraints
    h_red = np.logical_or(hsv[:, :, 0] >= float(320) / 360, hsv[:, :, 0] <= float(10) / 360)
    s_red = hsv[:, :, 1] >= float(120) / 360
    v_red = hsv[:, :, 2] >= float(30) / 360
    binary_red = np.logical_and(h_red, np.logical_and(s_red, v_red))
    binary_red = binary_opening(binary_red)
    red_labels = label(binary_red)

    for region in regionprops(red_labels):
        # take regions with large enough areas
        bbox = region.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width > 0.3 * image.shape[0] or height > 0.3 * image.shape[1]:
            region_pixels = red_labels == region.label
            binary_red[region_pixels] = 0
    binary_red = binary_dilation(binary_red, square(5))

    # Blue filter constraints
    h_blue = np.logical_and(hsv[:, :, 0] > float(210) / 360, hsv[:, :, 0] <= float(230) / 360)
    s_blue = hsv[:, :, 1] >= float(290) / 360
    v_blue = hsv[:, :, 2] >= float(20) / 360
    binary_blue = np.logical_and(h_blue, np.logical_and(s_blue, v_blue))
    binary_blue = binary_opening(binary_blue)
    blue_labels = label(binary_blue)

    for region in regionprops(blue_labels):
        # take regions with large enough areas
        bbox = region.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width > 0.3 * image.shape[0] or height > 0.3 * image.shape[1]:
            region_pixels = red_labels == region.label
            binary_blue[region_pixels] = 0
    binary_blue = binary_dilation(binary_blue, square(5))

    # White / Achromatic pixels
    # paper: https://thesai.org/Downloads/Volume7No1/Paper_93-Traffic_Sign_Detection_and_Recognition.pdf
    # paper: https://www.researchgate.net/publication/296196586_Ayoub_Ellahyani_Mohamed_El_AnsariIlyas_El_Jaafari_Traffic_sign_detection_and_recognition_based_on_random_forests_Applied_Soft_Computing
    # TODO: Discard ROIs based on size and aspect ratio constraints for traffic signs containing white areas
    '''
    D = 17
    a_image = image.astype(int)
    achromatic = (np.absolute(a_image[:, :, 0] - a_image[:, :, 1]) + \
                  np.absolute(a_image[:, :, 1] - a_image[:, :, 2]) + \
                  np.absolute(a_image[:, :, 2] - a_image[:, :, 0])) / (3 * D)
    achromatic = achromatic < 1.0
    '''

    binary = np.logical_or(binary_blue, binary_red)
    binary = binary_closing(binary)
    binary = binary_dilation(binary, square(5))

    return binary

def extract_ROIs(binary_image):
    # label image regions
    labeled = label(binary_image)
    patches_coords = []
    
    for region in regionprops(labeled):
        # take regions with large enough areas
        bbox = region.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if width >= 20 and height >= 20:
            min_row, min_col, max_row, max_col = bbox
            patches_coords.append( (int(min_row), int(max_row), int(min_col), int(max_col)) )
    
    return patches_coords
