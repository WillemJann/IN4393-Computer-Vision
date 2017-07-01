import numpy as np
from skimage import color
from skimage import io
import matplotlib.pyplot as plt

import our_hog
from skimage.feature import hog


def compare_hogs(image):
    # Compare our hog implementation with SKImage hog implementation.
    nr_hist_bins = 9
    cell_size = (8,8)
    block_size = (3,3)

    our_descriptor = our_hog.hog(image, cell_size=cell_size, block_size=block_size, nr_of_hist_bins=nr_hist_bins)
    their_descriptor = hog(image, orientations=nr_hist_bins, pixels_per_cell=cell_size,
                           cells_per_block=block_size, block_norm='L2')

    # Compare our descriptor to theirs
    print 'Equal feature dimensions: ', our_descriptor.shape == their_descriptor.shape
    print 'Sum of absolute differences between descriptors: ', np.sum(np.abs(our_descriptor - their_descriptor))

    # Compare our gradients to their gradients
    sum_of_absolute_differences = compare_gradients(image)
    print 'Sum of absolute differences between our gradient and skimage gradient: ', sum_of_absolute_differences

    # Compare our descriptor when we use their gradients to their whole implementation.
    print ''
    print 'So there is a difference between our gradients.'
    print 'We use an approximation with a simple filter, they use a numpy function to get the gradient.'
    print 'So we add a testing function where we substitute our gradient for theirs and check for differences again:'
    print 'Difference between our and their gradient is now 0.'
    our_descriptor = their_gradient_hog(image, cell_size, block_size, nr_hist_bins)
    print 'Sum of absolute differences between descriptors: ', np.sum(np.abs(our_descriptor - their_descriptor))

    print ''
    print 'So there is still a difference between our implementation.'
    print 'Now also substitute our histogram per cell for their histogram per cell and test again.'
    our_descriptor = their_gradient_their_hist_hog(image, cell_size, block_size, nr_hist_bins)
    print 'Sum of absolute differences between descriptors: ', np.sum(np.abs(our_descriptor - their_descriptor))
    print ''
    print 'Now the results are equal. It appears that we use a different way to calculate our gradients.'
    print 'Besides that, we also use a different way to calculate our histogram. We use a simplified linear interpolation'
    print 'to distribute the weights. It is possible their implementation uses a different form of interpolation.'

def compare_gradients(image):
    # Show the original image
    plt.figure()
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('Original Image')

    # Calculate our gradient in x and y direction.
    gx, gy = our_hog.calculate_gradient(image)
    our_g = np.sqrt(gx ** 2 + gy ** 2)

    f, ax = plt.subplots(3, sharey=True)
    ax = ax.ravel()
    ax[0].imshow(gx, cmap=plt.cm.gray)
    ax[0].set_title('our_gradient in x-direction')
    ax[0].axis('off')
    ax[1].imshow(gy, cmap=plt.cm.gray)
    ax[1].set_title('our_gradient in y-direction')
    ax[1].axis('off')
    ax[2].imshow(our_g, cmap=plt.cm.gray)
    ax[2].set_title('our_gradient')
    ax[2].axis('off')


    # Calculate the gradient in x and y direction the way SKImage's hog implementation does.
    gy, gx = [np.ascontiguousarray(g, dtype=np.double) for g in np.gradient(image)]
    their_g = np.hypot(gx, gy)
    f, ax = plt.subplots(3, sharey=True)
    ax = ax.ravel()
    ax[0].imshow(gx, cmap=plt.cm.gray)
    ax[0].set_title('np.gradient in x-direction')
    ax[0].axis('off')
    ax[1].imshow(gy, cmap=plt.cm.gray)
    ax[1].set_title('np.gradient in y-direction')
    ax[1].axis('off')
    ax[2].imshow(their_g, cmap=plt.cm.gray)
    ax[2].set_title('np.gradient')
    ax[2].axis('off')

    plt.show()

    return np.sum(np.abs(our_g - their_g))

def their_gradient_hog(image, cell_size, block_size, nr_of_hist_bins):
    # Instead of our gradient, use SKImage gradient
    gy, gx = [np.ascontiguousarray(g, dtype=np.double) for g in np.gradient(image)]
    # Calculate the magnitude and angle from the gradient at each position.
    magnitudes = np.hypot(gx, gy)
    angles = np.rad2deg(np.arctan2(gy, gx)) % 180 # Angles in degrees, range [0,180]

    # Subdivide magnitude and angle image in areas of cell_size
    blocks_mags = our_hog.create_hog_cells(magnitudes, cell_size)
    blocks_angles = our_hog.create_hog_cells(angles, cell_size)

    # Calculate histogram per cell
    shape = (blocks_mags.shape[0], blocks_mags.shape[1], nr_of_hist_bins)
    hist_range = (0, 180)
    histograms = np.zeros(shape, dtype=np.double)
    for i in range(shape[0]):
        for j in range(shape[1]):
            angles = blocks_angles[i, j, :, :].flatten()
            weights = blocks_mags[i, j, :, :].flatten()

            histograms[i, j, :] = our_hog.create_histogram(angles, weights, nr_of_hist_bins, hist_range)

    # Normalize histograms per block (L2 norm)
    blocks = our_hog.create_hog_blocks(histograms, block_size)
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


def their_gradient_their_hist_hog(image, cell_size, block_size, nr_of_hist_bins):
    # Instead of our gradient, use SKImage gradient
    gy, gx = [np.ascontiguousarray(g, dtype=np.double) for g in np.gradient(image)]
    # Calculate the magnitude and angle from the gradient at each position.
    magnitudes = np.hypot(gx, gy)
    angles = np.rad2deg(np.arctan2(gy, gx)) % 180 # Angles in degrees, range [0,180]

    # Subdivide magnitude and angle image in areas of cell_size
    blocks_mags = our_hog.create_hog_cells(magnitudes, cell_size)
    blocks_angles = our_hog.create_hog_cells(angles, cell_size)

    # Use SKImage hoghistogram instead of our own implementation.
    from skimage.feature import _hoghistogram
    sy, sx = image.shape
    cx, cy = cell_size
    n_cellsx = int(sx // cx)  # number of cells in x
    n_cellsy = int(sy // cy)  # number of cells in y
    histograms = np.zeros((n_cellsy, n_cellsx, nr_of_hist_bins))
    _hoghistogram.hog_histograms(gx, gy, cx, cy, sx, sy, n_cellsx, n_cellsy,
                                 nr_of_hist_bins, histograms)

    # Normalize histograms per block (L2 norm)
    blocks = our_hog.create_hog_blocks(histograms, block_size)
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

if __name__ == "__main__":
    # Load image and normalize it
    image = io.imread('../../data/circles_normalized/A01-05.png')
    rgb = color.rgba2rgb(image, background=(0,0,0))
    gray = color.rgb2gray(rgb)
    image = np.float32(gray)/ 255.0

    # Compare our descriptor with skimage descriptor
    compare_hogs(image)