import operator
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import euclidean
from skimage import draw, feature, transform
from skimage.color import gray2rgb, rgb2hsv, rgb2gray
from skimage.io import imread
from skimage.measure import label
from skimage.morphology import binary_dilation, binary_erosion, binary_closing
from skimage.morphology import disk
from skimage.morphology.misc import remove_small_objects
from skimage.transform import resize

from find_patches import binary_filtering, extract_ROIs
from train_classifier_clean import get_classifier

HOG_ORIENTATIONS = 9
HOG_CELL_SIZE = (8,8)
HOG_BLOCK_SIZE = (3,3)
HOG_BLOCK_NORM = 'L2-Hys'

def detect_circles(image):    
    min_image_size = min(image.shape[0], image.shape[1])
    max_image_size = max(image.shape[0], image.shape[1])
    
    min_radius = max(10, int(min_image_size / 4))
    max_radius = int(max_image_size / 2) 
    
    #print 'Find circles with radius: (%d, %d)' % (min_radius, max_radius)
    
    hough_radii = np.arange(min_radius, max_radius)
    hough_res = transform.hough_circle(image, hough_radii)
    
    min_distance = int(max_image_size / 3)
    threshold = 0.6
    num_peaks = np.inf
    total_num_peaks = 5
    normalize = True
    
    accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii, min_distance, min_distance, threshold, num_peaks, total_num_peaks, normalize)
    detected_circles = zip(cy, cx, radii, accums)
    
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
    clustered_circles = []
    
    for cluster in clusters:
        largest_circle = max(cluster, key=operator.itemgetter(2))
        clustered_circles.append(largest_circle)
    
    # generate circle image
    detected_circle_image = gray2rgb(image.astype(np.uint8)*255)
    for center_y, center_x, radius, intensity in detected_circles:
        #print '=> circle: (%s, %s), radius: %s, intensity: %s' % (center_y, center_x, radius, intensity)
        circy, circx = draw.circle_perimeter(center_y, center_x, radius, shape=image.shape)
        detected_circle_image[circy, circx] = (220, 20, 20)
    
    # generate circle image
    clustered_circle_image = gray2rgb(image.astype(np.uint8)*255)
    for center_y, center_x, radius, intensity in clustered_circles:
        #print '=> circle: (%s, %s), radius: %s, intensity: %s' % (center_y, center_x, radius, intensity)
        circy, circx = draw.circle_perimeter(center_y, center_x, radius, shape=image.shape)
        clustered_circle_image[circy, circx] = (220, 20, 20)
    
    # Return results
    return clustered_circles, clustered_circle_image, detected_circle_image

def crop_circle(image, circle):
    cropped_image = rgb2gray(image)

    center_y, center_x, radius, _ = circle
    x1 = center_x - radius
    x2 = center_x + radius
    y1 = center_y - radius
    y2 = center_y + radius
    
    circy, circx = draw.circle(center_y, center_x, radius, shape=image.shape)
    mask = np.ones(cropped_image.shape, dtype=bool)
    mask[circy, circx] = False
    
    cropped_image[mask] = 0
    cropped_image = cropped_image[y1:y2, x1:x2]
    
    return cropped_image

def get_hog_features(image):
    image = resize(image, (75,75))
    return feature.hog(image, HOG_ORIENTATIONS, HOG_CELL_SIZE, HOG_BLOCK_SIZE, HOG_BLOCK_NORM, visualise=True, transform_sqrt=True)

def test(image, filename, classifier, display='both'):
    image_size = max(image.shape[0], image.shape[1])
    filter_size = int(image_size / 15)
        
    # Convert to HSV
    image_hsv = rgb2hsv(image)
    
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
    
    red_circles, red_circles_image, red_all_circles_image  = detect_circles(red_skeleton)
    blue_circles, blue_circles_image, blue_all_circles_image  = detect_circles(blue_skeleton)
    
    # Crop circles and extract HOG features
    if len(red_circles) >= 1:
        red_cropped_1 = crop_circle(image, red_circles[0])
        red_features_1, red_hog_image_1 = get_hog_features(red_cropped_1)
    if len(red_circles) >= 2:
        red_cropped_2 = crop_circle(image, red_circles[1])
        red_features_2, red_hog_image_2 = get_hog_features(red_cropped_2)
        
    if len(blue_circles) >= 1:
        blue_cropped_1 = crop_circle(image, blue_circles[0])
        blue_features_1, blue_hog_image_1 = get_hog_features(blue_cropped_1)
    if len(blue_circles) >= 2:
        blue_cropped_2 = crop_circle(image, blue_circles[1])
        blue_features_2, blue_hog_image_2 = get_hog_features(blue_cropped_2)
        
    # Perform classification
    if len(red_circles) >= 1 and red_features_1 is not None:
        print 'Result: %s' % classifier.predict(red_features_1.reshape(1, -1))
    if len(red_circles) >= 2 and red_features_2 is not None:
        print 'Result: %s' % classifier.predict(red_features_2.reshape(1, -1))
        
    if len(blue_circles) >= 1 and blue_features_1 is not None:
        print 'Result: %s' % classifier.predict(blue_features_1.reshape(1, -1))
    if len(blue_circles) >= 2 and blue_features_2 is not None:
        print 'Result: %s' % classifier.predict(blue_features_2.reshape(1, -1))
    
    # Display results
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 9), sharex=False, sharey=False, subplot_kw={'adjustable':'box-forced'})
    ax = axes.ravel()
    
    ax[0].imshow(image)
    ax[0].set_title('Original patch')
    
    ax[1].imshow(H, cmap='gray')
    ax[1].set_title("H channel")
    
    ax[2].imshow(S, cmap='gray')
    ax[2].set_title("S channel")
    
    ax[3].imshow(V, cmap='gray')
    ax[3].set_title("V channel")
    
    if display == 'both':
        ax[4].imshow(binary_red, cmap='gray')
        ax[4].set_title("Binary red")
        
        ax[5].imshow(red_labels, cmap='nipy_spectral')
        ax[5].set_title("Red labels")
        
        ax[6].imshow(red_circles_image)
        ax[6].set_title("Red circles")
        
        if len(red_circles) >= 1:
            ax[7].imshow(red_hog_image_1, cmap='gray')
            ax[7].set_title("Red sign #1 HOG")
        
        ax[8].imshow(binary_blue, cmap='gray')
        ax[8].set_title("Binary blue")
        
        ax[9].imshow(blue_labels, cmap='nipy_spectral')
        ax[9].set_title("Blue labels")
        
        ax[10].imshow(blue_circles_image)
        ax[10].set_title("Blue circles")
        
        if len(blue_circles) >= 1:
            ax[11].imshow(blue_cropped_1, cmap='gray')
            ax[11].set_title("Blue sign #1")
        
    
    elif display == 'red':
        ax[4].imshow(binary_red, cmap='gray')
        ax[4].set_title("Binary red")
        
        ax[5].imshow(red_labels, cmap='nipy_spectral')
        ax[5].set_title("Red labels")
        
        ax[6].imshow(red_all_circles_image)
        ax[6].set_title("Red circles")
        
        ax[7].imshow(red_circles_image)
        ax[7].set_title("Red circles clustered")
        
        if len(red_circles) >= 1:
            ax[8].imshow(red_cropped_1, cmap='gray')
            ax[8].set_title("Red sign #1")
            
            ax[9].imshow(red_hog_image_1, cmap='gray')
            ax[9].set_title("Red sign #1 HOG")
        
        if len(red_circles) >= 2:
            ax[10].imshow(red_cropped_2, cmap='gray')
            ax[10].set_title("Red sign #2")
            
            ax[11].imshow(red_hog_image_2, cmap='gray')
            ax[11].set_title("Red sign #2 HOG")
    
    elif display == 'blue':
        ax[4].imshow(binary_blue, cmap='gray')
        ax[4].set_title("Binary blue")
        
        ax[5].imshow(blue_labels, cmap='nipy_spectral')
        ax[5].set_title("Blue labels")
        
        ax[6].imshow(blue_all_circles_image)
        ax[6].set_title("Blue circles")
        
        ax[7].imshow(blue_circles_image)
        ax[7].set_title("Blue circles clustered")
        
        if len(blue_circles) >= 1:
            ax[8].imshow(blue_cropped_1, cmap='gray')
            ax[8].set_title("Blue sign #1")
            
            ax[9].imshow(blue_hog_image_1, cmap='gray')
            ax[9].set_title("Blue sign #1 HOG")
        
        if len(blue_circles) >= 2:
            ax[10].imshow(blue_cropped_2, cmap='gray')
            ax[10].set_title("Blue sign #2")
            
            ax[11].imshow(blue_hog_image_2, cmap='gray')
            ax[11].set_title("Blue sign #2 HOG")
    
    
    # Disable axes
    for i in range(0, 12):
        ax[i].axis('off')
        
    # Show results
    fig.tight_layout()
    plt.gcf().canvas.set_window_title(os.path.basename(filename))
    plt.show()


if __name__ == "__main__":
    # Load classifier
    classifier = get_classifier()
    
    # Evaluate individual patches
    for filename in glob.glob('../../data/streetview_patches/positive/*.jpg'):
        patch = imread(filename)
        test(patch, filename, classifier, display = 'both') # use display = red, blue or both
    
    '''
    # Evaluate full images
    for filename in glob.glob('../../data/streetview_images/*.jpg'):
        image = imread(filename)
        
        segmented_image = binary_filtering(image)
        patches = extract_ROIs(segmented_image)
        
        classifier = get_classifier()
        
        for coords in patches:
            patch = image[coords[0]:coords[1], coords[2]:coords[3], :]
            test(patch, filename, classifier, display = 'both')
    '''
