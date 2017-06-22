import collections
import glob
from operator import itemgetter
import os
import pickle

from numpy.random.mtrand import normal
from scipy import ndimage as ndi
from scipy.spatial.distance import euclidean
from skimage import color, draw, feature, filters, io, transform, morphology
from skimage import data
from skimage import data, color
from skimage.color import rgb2hsv, rgb2gray
from skimage.draw import circle_perimeter
from skimage.draw import line
from skimage.feature._canny import canny
from skimage.feature.peak import peak_local_max
from skimage.filters import rank
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_erosion, square, binary_closing, binary_opening
from skimage.morphology import skeletonize
from skimage.morphology import watershed, disk
from skimage.morphology.misc import remove_small_objects
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.util import img_as_ubyte
from sklearn import svm

import matplotlib.pyplot as plt
import numpy as np


TRAINING_FOLDER = '../../data/circles_normalized/'
CLASSIFIER_FILE = '../../data/classifiers/svm_circles_normalized.pickle'

HOG_ORIENTATIONS = 8
HOG_CELL_SIZE = (3,3)
HOG_BLOCK_SIZE = (1,1)
HOG_BLOCK_NORM = 'L2-Hys'

# Returns dictonary with all labeled images from the training folder
def load_training_data():
    files = glob.glob(TRAINING_FOLDER + '*.png')
    data = collections.OrderedDict()
    
    for file in files:
        filename = os.path.basename(file)
        label = os.path.splitext(filename)[0]
        
        data[label] = io.imread(file)
    
    return data

# Returns dictonary with feature vector for each training image
def extract_hog_features(training_images):
    training_set = collections.OrderedDict()
    
    for label, image in training_images.iteritems():
        # Fill alpha channel with white and convert image to grayscale
        image = color.rgba2rgb(image, background=(0,0,0))
        image = color.rgb2gray(image)
        
        # Extract HOG features
        features = feature.hog(image, HOG_ORIENTATIONS, HOG_CELL_SIZE, HOG_BLOCK_SIZE, HOG_BLOCK_NORM)
        
        # Show results
        #_, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        #ax1.imshow(image, cmap='gray')
        #ax2.imshow(hog_image, cmap='gray')
        #plt.show()
        
        # Add features to training set
        training_set[label] = features
    
    return training_set

# Returns a trained SVM classifier based on training set
def train_classifier(training_set):
    training_samples = training_set.values();
    labels = training_set.keys();
    
    classifier = svm.SVC()
    classifier.fit(training_samples, labels)
    
    return classifier

def test(image):
    #image_resized = transform.resize(image, (75, 75), mode='reflect')
    
    # Convert to HSV
    image_hsv = color.rgb2hsv(image)
    image_hue = image_hsv[:,:,0]
    image_hue[image_hue > 0.8] = 0

    '''
    # Apply edge detection
    edges = canny(image_hue, sigma=2.3)
    
    # Display results
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    ax = axes.ravel()
    
    ax[0].imshow(image)
    ax[0].set_title("Hue channel")
    
    ax[1].imshow(image_hue, cmap='gray')
    ax[1].set_title("Hue channel")
    
    ax[2].imshow(edges, cmap='gray')
    ax[2].set_title("Edges")
    
    fig.tight_layout()
    plt.show()
    '''
    
    # Apply morphological watershed
    denoised = rank.median(image_hue, disk(5))
    gradient = rank.gradient(denoised, disk(2))
    markers = gradient < 15
    markers = ndi.label(markers)[0]
    labels = watershed(gradient, markers)
    
    # display results
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9, 6), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    ax = axes.ravel()
    
    ax[0].imshow(image, cmap='gray', interpolation='nearest')
    ax[0].set_title("Original")
    
    ax[1].imshow(image_hue, cmap='gray', interpolation='nearest')
    ax[1].set_title("Hue channel")
    
    ax[2].imshow(denoised, cmap='gray', interpolation='nearest')
    ax[2].set_title("Denoised")
    
    ax[3].imshow(gradient, cmap='spectral', interpolation='nearest')
    ax[3].set_title("Local Gradient")
    
    ax[4].imshow(markers, cmap='spectral', interpolation='nearest')
    ax[4].set_title("Markers")
    
    ax[5].imshow(image, cmap='gray', interpolation='nearest')
    ax[5].imshow(labels, cmap='spectral', interpolation='nearest', alpha=.7)
    ax[5].set_title("Segmented")
    
    for a in ax:
        a.axis('off')
    
    fig.tight_layout()
    plt.show()
    '''
    # Apply watershed segmentation
    image_segmented = binary_filtering(image)
    
    distance = ndi.distance_transform_edt(image_segmented)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=image_segmented)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=image_segmented)
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    ax = axes.ravel()
    
    ax[0].imshow(image)
    ax[0].set_title('Original image')
    ax[1].imshow(image_segmented, cmap='gray', interpolation='nearest')
    ax[1].set_title('Overlapping objects')
    ax[2].imshow(-distance, cmap='gray', interpolation='nearest')
    ax[2].set_title('Distances')
    ax[3].imshow(labels, cmap='spectral', interpolation='nearest')
    ax[3].set_title('Separated objects')
    
    for a in ax:
        a.set_axis_off()
    
    fig.tight_layout()
    plt.show()
    '''

def detect_circles(image):    
    min_image_size = min(image.shape[0], image.shape[1])
    max_image_size = max(image.shape[0], image.shape[1])
    
    min_radius = max(10, int(min_image_size / 4))
    max_radius = int(max_image_size / 2) 
    
    #print 'Find circles with radius: (%d, %d)' % (min_radius, max_radius)
    
    hough_radii = np.arange(min_radius, max_radius)
    hough_res = transform.hough_circle(image, hough_radii)
    
    min_distance = int(max_image_size / 3)
    threshold = 0.55
    num_peaks = np.inf
    total_num_peaks = 5
    normalize = True
    
    _, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii, min_distance, min_distance, threshold, num_peaks, total_num_peaks, normalize)
    detected_circles = zip(cy, cx, radii)
    
    # cluster circles
    clusters = []
    cluster_distance = 10
    
    for circle in detected_circles:
        distance = None
        
        # Add current circle to an existing cluster if it is nearby this cluster
        for cluster in clusters:
            cluster_mean = np.mean(np.array(cluster)[:,:2], axis=0)
            distance = euclidean(circle[:2], cluster_mean)
            
            if (distance <= cluster_distance):
                cluster.append(circle)
                break
        
        # Create new cluster if circle is not close to an existing cluster
        if distance is None or distance > cluster_distance:
            clusters.append([circle])
        
    # find circles
    circles = []
    
    for cluster in clusters:
        largest_circle = max(cluster, key=itemgetter(2))
        circles.append(largest_circle)
    
    # generate circle image
    circle_image = color.gray2rgb(image.astype(np.uint8)*255)
    for center_y, center_x, radius in circles:
        #print '=> circle: (%s, %s), radius: %s, intensity: %s' % (center_y, center_x, radius, intensity)
        circy, circx = draw.circle_perimeter(center_y, center_x, radius, shape=image.shape)
        circle_image[circy, circx] = (220, 20, 20)
    
    # return only first circle
    circle_mask = np.ones(image.shape, dtype=bool)
    
    if circles:
        circy, circx = draw.circle(circles[0][0], circles[0][1], circles[0][2], shape=image.shape)
        circle_mask[circy, circx] = False
    
    # Return results
    return circle_mask, circle_image

def test_new(image):
    image_size = max(image.shape[0], image.shape[1])
    filter_size = int(image_size / 15)
    
    #print 'Image size: %d' % image_size
    
    # Convert to HSV
    image_hsv = color.rgb2hsv(image)
    
    # Get HSV channels
    H = image_hsv[:,:,0]
    S = image_hsv[:,:,1]
    V = image_hsv[:,:,2]
    
    # Fix H channel
    H[H > 0.8] = 0

    # Red filter constraints
    binary_red = np.logical_and(H <= 0.05, S >= 0.3)
    red_segments = remove_small_objects(binary_red, 64)
    red_segments = binary_closing(red_segments, disk(filter_size))
    red_labels = label(red_segments)

    # Blue filter constraints
    binary_blue = np.logical_and(np.logical_and(H >= 0.55, H <= 0.65), S >= 0.4)
    blue_segments = remove_small_objects(binary_blue, 64)
    blue_segments = binary_closing(blue_segments, disk(filter_size))
    blue_labels = label(blue_segments)
    
    # Detect circles
    red_skeleton = np.logical_xor(binary_dilation(red_segments), binary_erosion(red_segments))
    blue_skeleton = np.logical_xor(binary_dilation(blue_segments), binary_erosion(blue_segments))
    
    red_circle, red_circles_image = detect_circles(red_skeleton)
    blue_circle, blue_circles_image = detect_circles(blue_skeleton)
    
    # Mask road sign
    red_sign = image.copy()
    red_sign[red_circle] = (255, 255, 255)
    blue_sign = image.copy()
    blue_sign[blue_circle] = (255, 255, 255)
    
    # Display results
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 9), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    ax = axes.ravel()
    
    ax[0].imshow(image)
    ax[0].set_title('Original image')
    
    ax[1].imshow(H, cmap='gray')
    ax[1].set_title("H channel")
    
    ax[2].imshow(S, cmap='gray')
    ax[2].set_title("S channel")
    
    ax[3].imshow(V, cmap='gray')
    ax[3].set_title("V channel")
    
    ax[4].imshow(binary_red, cmap='gray')
    ax[4].set_title("Binary red")
    
    ax[5].imshow(red_labels, cmap='nipy_spectral')
    ax[5].set_title("Red labels")
    
    ax[6].imshow(red_circles_image)
    ax[6].set_title("Red circles")
    
    ax[7].imshow(red_sign)
    ax[7].set_title("Red sign")
    
    ax[8].imshow(binary_blue, cmap='gray')
    ax[8].set_title("Binary blue")
    
    ax[9].imshow(blue_labels, cmap='nipy_spectral')
    ax[9].set_title("Blue labels")
    
    ax[10].imshow(blue_circles_image)
    ax[10].set_title("Blue circles")
    
    ax[11].imshow(blue_sign)
    ax[11].set_title("Blue sign")
    
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Train classifier and save to disk
    #training_set = load_training_data()
    #training_set = extract_hog_features(training_set)
    #classifier = train_classifier(training_set)
    #pickle.dump(classifier, open(CLASSIFIER_FILE, 'wb'), pickle.HIGHEST_PROTOCOL)
    
    # Load images
    for filename in glob.glob('../../data/streetview_images_segmented/positive/*.jpg'):
        image = io.imread(filename)
        test_new(image)
