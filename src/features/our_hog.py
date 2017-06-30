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
    # Define the bin size and the lower bounds of each bin in the histogram
    bin_size = hist_range[1] / nr_of_bins
    bin_edges = range(hist_range[0], hist_range[1] + 1, bin_size)

    # Create a 1D histogram of size: nr_of_bins
    histogram = np.zeros(nr_of_bins, dtype=np.double)
    # The idea is that a histogram is created of all gradient values of a cell as this is a form of quantization.
    # The angles of the gradient define to which 2 bins this gradient belongs to.
    # The magnitudes of the gradient are the weights of how much that gradient contributes to the two bins.
    # This magnitude as weight is linearly interpolated between the two bins.

    # For each bin:
    # - Calculate which indices of the angles list falls into this bin
    # - For each index:
    #    - Get the angle at position 'index' in the angles list
    #    - Get the corresponding weight
    #    - Linearly interpolate the weight between the two neighbouring bins
    #    - Note that an angle of for instance 177 degrees contributes both to the bin 170-180 and 0-10,
    #      so bin indices should be defined circular
    for k in range(len(bin_edges)):
        indices_in_range = np.where(np.logical_and(angles >= bin_edges[k], angles < bin_edges[k] + bin_size))[0]
        for index in indices_in_range:
            angle = angles[index]
            weight = weights[index]

            diff = (angle - np.double(bin_edges[k]))
            histogram[k] += weight * (1 - (diff / np.double(bin_size)))
            histogram[(k + 1) % nr_of_bins] += weight * (diff / np.double(bin_size))
    return histogram


def create_hog_cells(matrix, cell_shape=(8,8)):
    # Calculate the number of cells that fit in the image.
    # If the image dimensions are not a multiple of the cell_shape, the maximum amount of
    # fully filled cells will be taken, the boundaries of the image where no complete cell fits,
    # will be discarded.
    nr_cells = (np.uint8(np.floor(np.float(matrix.shape[0]) / (np.float(cell_shape[0])))),
                 np.uint8(np.floor(np.float(matrix.shape[0]) / (np.float(cell_shape[0])))))
    # Create the output matrix which contains the image when split into cells
    cells = np.zeros((nr_cells[0], nr_cells[1], cell_shape[0], cell_shape[1]), dtype=np.double)
    # For each cell in the output matrix, copy the corresponding image values into the output matrix
    for row in range(nr_cells[0]):
        for col in range(nr_cells[1]):
            row_start = row * cell_shape[0]
            row_end = row_start + cell_shape[0]
            col_start = col * cell_shape[1]
            col_end = col_start + cell_shape[1]

            cell = matrix[row_start:row_end, col_start:col_end]
            cells[row, col, :, :] = cell
    return cells


def create_hog_blocks(cells, block_shape=(3,3)):
    pass