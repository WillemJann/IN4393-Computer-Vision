from skimage import filters
import numpy as np
from skimage import color
from skimage import io
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt
from scipy import ndimage

HOG_CELL_SIZE = (3,3) # Should always fit in image size, we use image size = 75
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

# Load image and normalize it
image = io.imread('../../data/circles_normalized/A01-05.png')
rgb = color.rgba2rgb(image, background=(0,0,0))
gray = color.rgb2gray(rgb)
image = np.float32(gray)/ 255.0

plt.figure()
plt.axis('off')
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Original Image')

# Calculate gradient in x and y direction
filter = np.array([[-1, 0, 1]])
grad_x = ndimage.convolve(image, filter)
grad_y = ndimage.convolve(image, np.transpose(filter))

# Calculate the gradient element-wise, based on the magnitude of grad_x and grad_y
# Calculate the angles element-wise in degrees between grad_x and grad_y
g = np.sqrt(grad_x * grad_x + grad_y * grad_y) # magnitude
theta = np.rad2deg(np.arctan2(grad_x, grad_y)) # angles in degrees

f, ax = plt.subplots(3, sharey=True)
ax = ax.ravel()
ax[0].imshow(grad_x, cmap=plt.cm.gray)
ax[0].set_title('gradient in x-direction')
ax[0].axis('off')
ax[1].imshow(grad_y, cmap=plt.cm.gray)
ax[1].set_title('gradient in y-direction')
ax[1].axis('off')
ax[2].imshow(g, cmap=plt.cm.gray)
ax[2].set_title('gradient')
ax[2].axis('off')
plt.show()

# subdivide gradient image in areas of CELL_SIZE
blocks_g = view_as_blocks(g, HOG_CELL_SIZE)
blocks_theta = view_as_blocks(theta, HOG_CELL_SIZE)

# collapse the last two dimensions in one
flatten_blocks_g = blocks_g.reshape(blocks_g.shape[0], blocks_g.shape[1], -1)
flatten_blocks_theta = blocks_theta.reshape(blocks_theta.shape[0], blocks_theta.shape[1], -1)
print g.shape
print blocks_g.shape
print flatten_blocks_g.shape
print flatten_blocks_theta.shape

# Calculate histogram per block
shape = (flatten_blocks_g.shape[0], flatten_blocks_g.shape[1], NR_OF_HIST_BINS)
hist_range = (0,180)
signed_hist = False
if signed_hist:
    hist_range = (0,360)
histograms = np.zeros(shape, dtype=np.uint16)
for i in range(shape[0]):
    for j in range(shape[1]):
        angles = flatten_blocks_theta[i,j,:]
        weights = flatten_blocks_g[i,j,:]
        histogram, bin_edges= np.histogram(angles, bins=shape[2], range=hist_range, weights=weights)
        histograms[i,j,:] = histogram
        plt.figure()
        plt.bar(bin_edges[:-1], histogram, width = 20)
        plt.xlim(min(bin_edges), max(bin_edges))
        plt.show()
