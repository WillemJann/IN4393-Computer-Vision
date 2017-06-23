import os
import fnmatch
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import transform
from skimage.feature import hog

# Parameters
data_root = '../../data/dataset_training/'
extension = '*.jpg'
target_size = (75,75)

HOG_PIXELS_PER_CELL = (8,8)
HOG_CELLS_PER_BLOCK = (3,3)
HOG_HIST_NR_OF_BINS = 9
HOG_BLOCK_NORM = 'L2-Hys'

def count_images_in_path(path, extension):
    nr_of_images = 0
    for root, dirs, files in os.walk(path):
        for _ in fnmatch.filter(files, extension):
            nr_of_images += 1
    return nr_of_images

def calculate_hog_feature_size():
    cell_size = np.array(HOG_PIXELS_PER_CELL, dtype=np.float)
    block_size = np.array(HOG_CELLS_PER_BLOCK, dtype=np.float)
    block_overlap = np.ceil(block_size / 2)
    image_size = np.array(target_size, dtype=np.float)
    if block_size[0] == 1 and block_size[1] == 1:
        blocks_per_image = np.floor((image_size / cell_size - block_size) + 1)
    else:
        blocks_per_image = np.floor((image_size / cell_size - block_size) / (block_size - block_overlap) + 1)

    nr_of_features = blocks_per_image[0] * blocks_per_image[1] * block_size[0] * block_size[1] * HOG_HIST_NR_OF_BINS
    return int(nr_of_features)

def extract_hog_features(image):
    return hog(image, HOG_HIST_NR_OF_BINS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM, transform_sqrt=True)


def process_images():
    nr_of_images = count_images_in_path(data_root, extension)
    hog_features_size = calculate_hog_feature_size()

    features = np.zeros((nr_of_images, hog_features_size), dtype=np.float64)
    labels = []

    print('%s Images Found: %d' % (extension.strip('*.'), nr_of_images))

    print('Processing dir: %s' % data_root)
    print('|')
    index = 0
    for subdir in os.listdir(data_root):
        label = subdir
        print('|-- %s' % label)
        for path_to_file in glob.glob( os.path.join(os.path.join(data_root,subdir), extension) ):
            print('| |-- %s' % path_to_file)
            image = io.imread(path_to_file, as_grey=True)
            image = transform.resize(image, output_shape=target_size, order=1, mode='reflect')

            features[index,:] = extract_hog_features(image)
            labels.append(label)
            index+=1

    labels = np.array(labels)

    np.save('hog_features', features)
    np.save('truth_labels', labels)


process_images()