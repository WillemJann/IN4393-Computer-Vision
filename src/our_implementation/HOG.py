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
import matplotlib.pyplot as plt
from scipy import ndimage

from skimage.feature import hog

HOG_CELL_SIZE = (5,5) # Should always fit in image size, we use image size = 75
HOG_BLOCK_SIZE = (3,3)
NR_OF_HIST_BINS = 9

def get_image():
    pass

def HOG(cell_size= HOG_CELL_SIZE, signed_hist=False , nr_of_hist_bins=NR_OF_HIST_BINS):

    pass

def calculate_gradient(direction='x'):
    if direction == 'x':
        pass
        #filters.sobel_h()
    elif direction== 'y':
        pass
        #filters.sobel_v()
    else:
        return None

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
rgb = color.rgba2rgb(image, background=(0,0,0))
gray = color.rgb2gray(rgb)
image = np.float32(gray)/ 255.0

plt.figure()
plt.axis('off')
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Original Image')

# Calculate gradient in x and y direction
filter = np.array([[1, 0, -1]])
#gx = np.double(ndimage.convolve(image, filter))
#gy = np.double(ndimage.convolve(image, np.transpose(filter)))

# Calculate the gradient element-wise, based on the magnitude of grad_x and grad_y
# Calculate the angles element-wise in degrees between grad_x and grad_y
#g = np.sqrt(gx **2 + gy ** 2) # magnitude

# SKImage code
gy, gx = [np.ascontiguousarray(g, dtype=np.double) for g in np.gradient(image)]
g = np.hypot(gx,gy)

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

# subdivide gradient image in areas of CELL_SIZE
blocks_g = view_as_blocks(g, HOG_CELL_SIZE)
blocks_theta = view_as_blocks(theta, HOG_CELL_SIZE)

# collapse the last two dimensions in one
flatten_blocks_g = blocks_g.reshape(blocks_g.shape[0], blocks_g.shape[1], -1)
flatten_blocks_theta = blocks_theta.reshape(blocks_theta.shape[0], blocks_theta.shape[1], -1)

# Calculate histogram per block
shape = (flatten_blocks_g.shape[0], flatten_blocks_g.shape[1], NR_OF_HIST_BINS)
hist_range = (0,180)
histograms = np.zeros(shape, dtype=np.double)

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

for i in range(shape[0]):
    for j in range(shape[1]):
        angles = flatten_blocks_theta[i,j,:]
        weights = flatten_blocks_g[i,j,:]

        # Not sure how to handle boundary conditions. for instance: angles right at an edge, should the weights be split
        # equally between the bins? I'm not sure how numpy handles this.
        #histogram, bin_edges= np.histogram(angles, bins=shape[2], range=hist_range, weights=weights, density=False)
        #histograms[i,j,:] = histogram
        histograms[i,j,:] = create_histogram(angles,weights, NR_OF_HIST_BINS, hist_range)
histograms = test_histograms


# Normalize histograms per windows (L2 norm)
window_shape = (HOG_BLOCK_SIZE[0], HOG_BLOCK_SIZE[1], NR_OF_HIST_BINS)
windows = view_as_windows(histograms, window_shape=window_shape, step=(1,1,NR_OF_HIST_BINS))
flatten_windows = windows.reshape(windows.shape[0], windows.shape[1], -1)
shape = flatten_windows.shape
normalized_histograms = np.zeros((shape[0],shape[1],shape[2]), dtype=np.double)

for i in range(shape[0]):
    for j in range(shape[1]):
        eps = 1e-5
        block = windows[i,j,:].ravel()
        normalized_histograms[i,j,:] = block / np.sqrt(np.sum(block **2) + eps ** 2)

# Concatenate all histograms to 1 large feature vector
hog_features = normalized_histograms.ravel()
hog_reference = hog(image, orientations=9, pixels_per_cell=HOG_CELL_SIZE,
                    cells_per_block=HOG_BLOCK_SIZE, block_norm='L2')

for i in range(3):
    for j in range(3):
        pass
        #print 'Our HOG function:'
        #print hog_features
        #print histograms[i,j,:]
        #print 'SKImage HOG function:'
        #print hog_reference
        #print test_histograms[i,j,:]
print 'Our hog shape: ',hog_features.shape
print 'Their hog shape: ',hog_reference.shape
print 'Max Difference: ', np.max(np.abs(hog_features-hog_reference))
#print hog_features
#print hog_reference
#print 'Max Difference: ', np.max(np.abs(histograms.ravel()-test_histograms.ravel()))
