'''
    NOTES:
- Differences between our HOG and SKImage HOG:
 * Gradient, they use np.gradient, we use a simple sobel like filter.
 * Histogram, they use custom histogram, we use a simple np.histogram

- Solution:
 * add 'validate vs skimage' parameter, which if true enables their gradient method and their histogram
 * Create unit tests for various cell and block sizes and assert the outcome of our method vs their method:
    + Look at function cell_hog: https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_hoghistogram.pyx
 * Write in report that we validated, but we can prove our gradient calculation method is different, although visually
   it does the same thing and write that we have a simple histogram, because we wrote a simplified version.

- Create several neat functions to provide the HOG features.
'''

from skimage import filters
import numpy as np
from skimage import color
from skimage import io
from skimage.util import view_as_blocks
from skimage.util.shape import view_as_windows
from skimage import transform
import matplotlib.pyplot as plt
from scipy import ndimage

from skimage.feature import hog
HOG_CELL_SIZE = (8,8) # Should always fit in image size, we use image size = 75
HOG_BLOCK_SIZE = (3,3)
NR_OF_HIST_BINS = 9

def get_image():
    pass

def our_hog(cell_size= HOG_CELL_SIZE, signed_hist=False , nr_of_hist_bins=NR_OF_HIST_BINS):
    pass

def calculate_gradient(direction='x'):
    filter = np.array([[1, 0, -1]])
    # If 'y' is passed, calculate gradient in y direction.
    # return gradient in 'x' direction otherwise
    if direction== 'y':
        g = np.double(ndimage.convolve(image, np.transpose(filter)))
    else:
        g = np.double(ndimage.convolve(image, filter))
    return g

def create_hog_cells(matrix, cell_shape=(8,8)):
    nr_cells = (np.uint8(np.floor(np.float(matrix.shape[0]) / (np.float(cell_shape[0])))),
                 np.uint8(np.floor(np.float(matrix.shape[0]) / (np.float(cell_shape[0])))))
    cells = np.zeros((nr_cells[0], nr_cells[1], cell_shape[0], cell_shape[1]), dtype=np.double)
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
    overlap = (np.uint8(np.floor(0.5 * block_shape[0])),
               np.uint8(np.floor(0.5 * block_shape[1])))
    nr_blocks = (cells.shape[0] - 2 * overlap[0], cells.shape[1] - 2 * overlap[1])
    blocks = np.zeros((nr_blocks[0], nr_blocks[1], block_shape[0], block_shape[1], cells.shape[2]),
                      dtype=np.double)
    for row in range(nr_blocks[0]):
        for col in range(nr_blocks[1]):
            row_end = row + block_shape[0]
            col_end = col + block_shape[1]
            cells_in_block = cells[row:row_end, col:col_end]

            blocks[row, col, :,:,:] = cells_in_block
    return blocks

def create_histogram(angles, weights, nr_of_bins=9, hist_range=(0, 180)):
    bin_size = hist_range[1] / nr_of_bins
    bin_edges = range(hist_range[0], hist_range[1] + 1, bin_size)
    histogram = np.zeros(nr_of_bins, dtype=np.double)
    for k in range(len(bin_edges)):
        in_range = np.where(np.logical_and(angles >= bin_edges[k], angles < bin_edges[k] + bin_size))[0]
        for index in in_range:
            angle = angles[index]
            weight = weights[index]

            diff = (angle - np.double(bin_edges[k]))
            histogram[k] += weight * (1 - (diff / np.double(bin_size)))
            histogram[(k + 1) % NR_OF_HIST_BINS] += weight * (diff / np.double(bin_size))
    return histogram

# Load image and normalize it
image = io.imread('../../data/circles_normalized/A01-05.png')
#image = io.imread('test.png')
#image = transform.resize(image, (8,8))
rgb = color.rgba2rgb(image, background=(0,0,0))
gray = color.rgb2gray(rgb)
image = np.float32(gray)/ 255.0

plt.figure()
plt.axis('off')
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Original Image')

# Calculate gradient in x and y direction
filter = np.array([[1, 0, -1]])
gx = calculate_gradient(direction='x')
gy = calculate_gradient(direction='y')

# Calculate the gradient element-wise, based on the magnitude of grad_x and grad_y
# Calculate the angles element-wise in degrees between grad_x and grad_y
g = np.sqrt(gx **2 + gy ** 2) # magnitude

# SKImage code
#gy, gx = [np.ascontiguousarray(g, dtype=np.double) for g in np.gradient(image)]
#g = np.hypot(gx,gy)

theta = np.rad2deg(np.arctan2(gy, gx)) % 180 # angles in degrees

f, ax = plt.subplots(3, sharey=True)
ax = ax.ravel()
ax[0].imshow(gx, cmap=plt.cm.gray)
ax[0].set_title('gradient in x-direction')
ax[0].axis('off')
ax[1].imshow(gy, cmap=plt.cm.gray)
ax[1].set_title('gradient in y-direction')
ax[1].axis('off')
ax[2].imshow(g, cmap=plt.cm.gray)
ax[2].set_title('gradient')
ax[2].axis('off')

f, ax = plt.subplots(3, sharey=True)
gy, gx = [np.ascontiguousarray(g, dtype=np.double) for g in np.gradient(image)]
g = np.hypot(gx,gy)
ax = ax.ravel()
ax[0].imshow(gx, cmap=plt.cm.gray)
ax[0].set_title('np.gradient in x-direction')
ax[0].axis('off')
ax[1].imshow(gy, cmap=plt.cm.gray)
ax[1].set_title('np.gradient in y-direction')
ax[1].axis('off')
ax[2].imshow(g, cmap=plt.cm.gray)
ax[2].set_title('np.gradient')
ax[2].axis('off')

f, ax = plt.subplots(3, sharey=True)
gx = ndimage.convolve(image, filter)
gy = ndimage.convolve(image, np.transpose(filter))
g = np.sqrt(gx **2 + gy ** 2) # magnitude
ngy, ngx = [np.ascontiguousarray(g, dtype=np.double) for g in np.gradient(image)]
ng = np.hypot(gx,gy)
ax = ax.ravel()
ax[0].imshow(np.abs(gx-ngx), cmap=plt.cm.gray)
ax[0].set_title('gradient - np.gradient in x-direction')
ax[0].axis('off')
ax[1].imshow(np.abs(gy-ngy), cmap=plt.cm.gray)
ax[1].set_title('gradient - np.gradient in y-direction')
ax[1].axis('off')
ax[2].imshow(np.abs(g-ng), cmap=plt.cm.gray)
ax[2].set_title('gradient - np.gradient')
ax[2].axis('off')

plt.show()

# USE SKIMAGE VALUES:
# SKImage code
gy, gx = [np.ascontiguousarray(g, dtype=np.double) for g in np.gradient(image)]
g = np.hypot(gx,gy)
theta = np.rad2deg(np.arctan2(gy, gx)) % 180 # angles in degrees

# Use SKImage hoghistogram
from skimage.feature import _hoghistogram
sy, sx = image.shape
cx, cy = HOG_CELL_SIZE
bx, by = HOG_BLOCK_SIZE
n_cellsx = int(sx // cx)  # number of cells in x
n_cellsy = int(sy // cy)  # number of cells in y

test_histograms = np.zeros((n_cellsy, n_cellsx, NR_OF_HIST_BINS))
_hoghistogram.hog_histograms(ngx,ngy, cx, cy, sx, sy, n_cellsx, n_cellsy,
NR_OF_HIST_BINS, test_histograms)

# subdivide gradient image in areas of CELL_SIZE
blocks_g = create_hog_cells(g, HOG_CELL_SIZE)
blocks_theta = create_hog_cells(theta, HOG_CELL_SIZE)

# collapse the last two dimensions in one
#flatten_blocks_g = blocks_g.reshape(blocks_g.shape[0], blocks_g.shape[1], -1)
#flatten_blocks_theta = blocks_theta.reshape(blocks_theta.shape[0], blocks_theta.shape[1], -1)

# Calculate histogram per block
shape = (blocks_g.shape[0], blocks_g.shape[1], NR_OF_HIST_BINS)
hist_range = (0,180)
histograms = np.zeros(shape, dtype=np.double)

for i in range(shape[0]):
    for j in range(shape[1]):
        angles = blocks_theta[i,j,:,:].flatten()
        weights = blocks_g[i,j,:,:].flatten()

        histograms[i,j,:] = create_histogram(angles,weights, NR_OF_HIST_BINS, hist_range)

# Normalize histograms per windows (L2 norm)
blocks = create_hog_blocks(histograms, HOG_BLOCK_SIZE)
blocks = blocks.reshape(blocks.shape[0], blocks.shape[1], -1)
shape = blocks.shape
normalized_histograms = np.zeros((shape[0],shape[1],shape[2]), dtype=np.double)

for i in range(shape[0]):
    for j in range(shape[1]):
        eps = 1e-5
        block = blocks[i,j,:]
        normalized_histograms[i,j,:] = block / np.sqrt(np.sum(block **2) + eps ** 2)

# Concatenate all histograms to 1 large feature vector
hog_features = normalized_histograms.flatten()
hog_reference = hog(image, orientations=9, pixels_per_cell=HOG_CELL_SIZE,
                    cells_per_block=HOG_BLOCK_SIZE, block_norm='L2')

print 'Our hog shape: ',hog_features.shape
print 'Their hog shape: ',hog_reference.shape
print 'Max hog feature Difference: ', np.max(np.abs(hog_features-hog_reference))
print 'Max hist Difference: ', np.max(np.abs(histograms.ravel()-test_histograms.ravel()))
