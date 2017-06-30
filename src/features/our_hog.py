import numpy as np

from scipy import ndimage

# HOG parameters
HOG_CELL_SIZE = (8, 8)
HOG_BLOCK_SIZE = (3, 3)
NR_OF_HIST_BINS = 9

# The Histogram of Oriented Gradient (HOG) algorithm consists of roughly 4 parts:
# - Firstly, the gradients in the 'x' and 'y' direction should be calculated
# - Secondly, the image is subdivided in cells, each consisting of the same amount of gradient values.
#   A histogram of gradient angles is calculated, where the magnitude of the corresponding gradient serves
#   as weight.
# - Thirdly, the histograms are normalized. Instead of normalizing the histogram of each cell,
#   the concatenated histograms of several cells together are normalized.
#   This group of cells is called a block and blocks are defined in a sliding window approach, where blocks
#   generally overlap each other.
# - Fourthly, all the histograms are concatenated to form one large feature vector, called the hog_descriptor.
def our_hog(image, cell_size = HOG_CELL_SIZE, block_size = HOG_BLOCK_SIZE, nr_of_hist_bins=NR_OF_HIST_BINS):
    # Firstly calculate the gradients of the image
    gx, gy = calculate_gradient(image)

    # Calculate the magnitude and angle from the gradient at each position.
    magnitudes = np.sqrt(gx **2 + gy ** 2)
    angles = np.rad2deg(np.arctan2(gy, gx)) % 180 # Angles in degrees, range [0,180]

    # Subdivide magnitude and angle image in areas of cell_size
    blocks_mags = create_hog_cells(magnitudes, cell_size)
    blocks_angles = create_hog_cells(angles, cell_size)

    # Calculate histogram per block
    shape = (blocks_mags.shape[0], blocks_mags.shape[1], nr_of_hist_bins)
    hist_range = (0, 180)
    histograms = np.zeros(shape, dtype=np.double)
    for i in range(shape[0]):
        for j in range(shape[1]):
            angles = blocks_angles[i, j, :, :].flatten()
            weights = blocks_mags[i, j, :, :].flatten()

            histograms[i, j, :] = create_histogram(angles, weights, nr_of_hist_bins, hist_range)

    # Normalize histograms per block (L2 norm)
    blocks = create_hog_blocks(histograms, block_size)
    blocks = blocks.reshape(blocks.shape[0], blocks.shape[1], -1)
    shape = blocks.shape
    normalized_histograms = np.zeros((shape[0], shape[1], shape[2]), dtype=np.double)
    for i in range(shape[0]):
        for j in range(shape[1]):
            eps = 1e-5
            block = blocks[i, j, :]
            normalized_histograms[i, j, :] = block / np.sqrt(np.sum(block ** 2) + eps ** 2)

    # Concatenate all normalized histograms to 1 large feature vector
    hog_features = normalized_histograms.flatten()
    return hog_features


# Calculates an approximation of the first order derivative in both x and y direction.
def calculate_gradient(image):
    # create a simple filter that approximates a derivative.
    filter = np.array([[1, 0, -1]])
    # convolve the filter with the image and convolve the transpose of the filter with the image
    # to get an approximation of the first order derivative in the x and y direction.
    gx = np.double(ndimage.convolve(image, np.transpose(filter)))
    gy = np.double(ndimage.convolve(image, filter))
    return gx, gy


# Create a 1D histogram of angles.
# This histogram is weighted in terms of gradients, which are linearly interpolated
# to define its contribution to each of the surrounding bins.
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


# This function divides an image in NxM equally shaped cells.
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


# This function creates MxN blocks of PxQ cells of size R
# In contrast to the create_hog_cells function, this function
# defines blocks in a sliding window fashion
def create_hog_blocks(cells, block_shape=(3,3)):
    # Define the amount of overlap in cells between two blocks.
    overlap = (np.uint8(np.floor(0.5 * block_shape[0])),
               np.uint8(np.floor(0.5 * block_shape[1])))
    # Define the amount of blocks to calculate.
    nr_blocks = (cells.shape[0] - 2 * overlap[0], cells.shape[1] - 2 * overlap[1])
    # Create an output matrix to store all the blocks in.
    blocks = np.zeros((nr_blocks[0], nr_blocks[1], block_shape[0], block_shape[1], cells.shape[2]),
                      dtype=np.double)
    # For each block in the image, copy the corresponding cell(s) and their content
    # and store it in the output matrix.
    for row in range(nr_blocks[0]):
        for col in range(nr_blocks[1]):
            row_end = row + block_shape[0]
            col_end = col + block_shape[1]
            cells_in_block = cells[row:row_end, col:col_end]

            blocks[row, col, :,:,:] = cells_in_block
    return blocks
