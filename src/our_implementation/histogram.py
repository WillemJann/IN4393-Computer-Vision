import numpy as np
hist_range = (0,180)
NR_OF_HIST_BINS = 9

def histogram(angles, weights, nr_of_bins=9, hist_range=(0,180)):
    bin_size = hist_range[1] / nr_of_bins
    bin_edges = range(hist_range[0], hist_range[1] + 1, bin_size)
    histogram = np.zeros(nr_of_bins)
    for k in range(len(bin_edges)):
        in_range = np.where( np.logical_and(angles >= bin_edges[k], angles < bin_edges[k] + bin_size) )[0]
        for index in in_range:
            angle = angles[index]
            weight = weights[index]

            diff = (angle - np.float32(bin_edges[k]))
            histogram[k] += weight * (1 - (diff / np.float32(bin_size)))
            histogram[(k + 1) % NR_OF_HIST_BINS] += weight * (diff / np.float32(bin_size))
    return histogram

angles = np.array([179.0])
weights = np.array([1])
hist = histogram(angles,weights)
print hist
