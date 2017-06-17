import io
import skimage.io

import numpy as np
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.io import imread
from skimage.color import rgb2hsv, rgb2gray
from skimage.morphology import binary_dilation, square, binary_closing, binary_opening
from skimage.draw import line
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.feature import peak_local_max, canny
from skimage.transform import hough_ellipse, resize
from skimage.draw import ellipse_perimeter
from skimage.util import crop

def detect_signs(image):
    # TODO: implement sign detection

    binary_segmentation = binary_filtering(image)
    ROIs, patch_coords = extract_ROIs(binary_segmentation)
    # TO DO, recognize the actual shape of the sign.
    # annotated = visualize_ROIs(ROIs, image)

    # Return results
    # im = image
    # im[:,:,0] = binary_segmentation * image[:,:,0]
    # im[:,:,1] = binary_segmentation * image[:,:,1]
    # im[:,:,2] = binary_segmentation * image[:,:,2]

    output = io.BytesIO()
    visualize_ROIs(ROIs, image, output)
    #find_circles(image, binary_segmentation)
    #skimage.io.imsave(output, image, plugin='pil', format_str='jpeg')
    output.seek(0)

    recognized_signs = {}

    return output, recognized_signs


def binary_filtering(image):
    hsv = rgb2hsv(image)

    # Red filter constraints
    h_red = np.logical_or(hsv[:, :, 0] >= float(240) / 360, \
                          hsv[:, :, 0] <= float(10) / 360)
    #s_red = hsv[:, :, 1] >= float(40) / 360
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
    h_blue = np.logical_and(hsv[:, :, 0] > float(210) / 360, \
                            hsv[:, :, 0] <= float(230) / 360)
    #s_blue = hsv[:, :, 1] >= float(127.5) / 360
    s_blue = hsv[:, :, 1] >= float(200) / 360
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
    D = 17
    a_image = image.astype(int)
    achromatic = (np.absolute(a_image[:, :, 0] - a_image[:, :, 1]) + \
                  np.absolute(a_image[:, :, 1] - a_image[:, :, 2]) + \
                  np.absolute(a_image[:, :, 2] - a_image[:, :, 0])) / (3 * D)
    achromatic = achromatic < 1.0

    binary = np.logical_or(binary_blue, binary_red)
    binary = binary_closing(binary)
    binary = binary_dilation(binary, square(5))

    return binary


def extract_ROIs(binary_image):
    # label image regions
    labeled = label(binary_image)
    # image_label_overlay = label2rgb(labeled, image=or_image)
    patches = []
    patch_coords = []
    #mask = np.zeros(binary_image.shape, dtype=np.uint8)
    for region in regionprops(labeled):
        # take regions with large enough areas
        bbox = region.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width >= 15 and height >= 15:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='green', linewidth=3)

            # ax4.add_patch(rect)
            patches.append(rect)
            patch_coords.append( ((int(minr), int(maxr)), (int(minc), int(maxc))) )
            #mask[minr:maxr, minc:maxc] = 1
    return patches, patch_coords#, mask


def visualize_ROIs(patches, image, output):
    # get the dimensions
    ypixels, xpixels, bands = image.shape

    # get the size in inches
    dpi = 72.
    xinch = xpixels / dpi
    yinch = ypixels / dpi

    # plot and save in the same size as the original
    fig = plt.figure(frameon=False)
    fig.set_size_inches(xinch, yinch)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect='normal')
    for region in patches:
        ax.add_patch(region)

    plt.savefig(output, format='jpeg', dpi=dpi)

def find_circles(image, mask=None):
    print "Finding circles."
    #image_gray = image
    #image_gray[:,:,0] = mask * image[:,:,0]
    #image_gray[:, :, 1] = mask * image[:, :, 1]
    #image_gray[:, :, 2] = mask * image[:, :, 2]
    image_gray = rgb2gray(image)

    print "canny"
    edges = canny(image_gray, sigma=2.0,
                  low_threshold=0.55, high_threshold=0.8)

    fig = plt.figure()
    #plt.imshow(edges, cmap=plt.cm.gray)
    #plt.show()

    print "Hough Transform"
    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    result = hough_ellipse(edges, accuracy=20, threshold=250,
                           min_size=10, max_size=70)
    result.sort(order='accumulator')

    if result:
        print "Estimate parameters for ellipse"
        # Estimated parameters for the ellipse
        best = list(result[-1])
        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        orientation = best[5]

        print "Draw ellipse"
        # Draw the ellipse on the original image
        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        image[cy, cx] = (0, 0, 255)


def test(image):
    binary_segmentation = binary_filtering(image)

    patches, coords = extract_ROIs(binary_segmentation)
    '''
    plt.figure()
    image[:,:,0] = mask * image[:,:,0]
    image[:, :, 1] = mask * image[:, :, 1]
    image[:, :, 2] = mask * image[:, :, 2]
    plt.imshow(image)


    bw_image = rgb2gray(image)
    edges = canny(bw_image, sigma=2.3)
    plt.figure()
    plt.imshow(edges, cmap=plt.cm.gray)
    '''
    for rows, columns in coords:
        patch = image[rows[0]:rows[1], columns[0]:columns[1], :]

        # patch = resize(patch, (75,75))
        # plt.figure()
        bw_image = rgb2gray(patch)
        edges = canny(bw_image, sigma=2.3)


        # Detect two radii
        hough_radii = np.arange(20, 75, 3)
        hough_res = hough_circle(edges, hough_radii)

        # Select the most prominent 5 circles
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, threshold=0.33, normalize=True)

        print(accums)
        # Draw them
        # image = color.gray2rgb(bw_image)
        plot = False
        for center_y, center_x, radius in zip(cy, cx, radii):
            if radius:
                plot = True
                circy, circx = circle_perimeter(center_y, center_x, radius, shape=bw_image.shape)
                patch[circy, circx, :] = (0, 250, 0)

        if plot:
            plt.figure()
            plt.imshow(patch, cmap=plt.cm.gray)
    plt.show()
    '''
    #find_circles(image, binary_segmentation)
    #fig = plt.figure()
    #plt.imshow(output)
    #plt.show()
    '''

# Run application
if __name__ == "__main__":
    from skimage.transform import hough_circle, hough_circle_peaks
    from skimage.feature import canny
    from skimage.draw import circle_perimeter

    from PIL import Image
    import glob

    image_list = []
    for filename in glob.glob('../test/*.jpg'):  # assuming gif
        image = imread(filename)
        plt.figure()
        plt.imshow(image)
        plt.show()
        test(image)
