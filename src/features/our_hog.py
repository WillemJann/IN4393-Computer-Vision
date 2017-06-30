import numpy as np

from scipy import ndimage


HOG_CELL_SIZE = (8,8) # Should always fit in image size, we use image size = 75
HOG_BLOCK_SIZE = (3,3)
NR_OF_HIST_BINS = 9

def our_hog(image, cell_size = HOG_CELL_SIZE, block_size = HOG_BLOCK_SIZE, nr_of_hist_bins=NR_OF_HIST_BINS):
    pass


def calculate_gradient(image):
    # create a simple filter that approximates a derivative.
    filter = np.array([[1, 0, -1]])
    # convolve the filter with the image and convolve the transpose of the filter with the image
    # to get an approximation of the first order derivative in the x and y direction.
    gx = np.double(ndimage.convolve(image, np.transpose(filter)))
    gy = np.double(ndimage.convolve(image, filter))
    # calculate the gradient based on the derivative in the x and y direction.
    g = np.sqrt(gx **2 + gy ** 2)
    return g


def create_histogram(angles, weights, nr_of_bins=9, hist_range=(0, 180)):
    pass


def create_hog_cells(matrix, cell_shape=(8,8)):
    pass


def create_hog_blocks(cells, block_shape=(3,3)):
    pass