import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
import numba as nb
import time

# # Plotting functionality for 2d embeddings
# def plot_2d(e1, e2):
#     x1 = e1[:,0]
#     y1 = e1[:,1]
#     x2 = e2[:,0]
#     y2 = e2[:,1]
#     plt.scatter(x1, y1, c='blue', label='e1')
#     plt.scatter(x2, y2, c='red', label='e2')
#     plt.legend(loc='best')
#     plt.title('Graph Embeddings')
#     plt.xlabel('0-dimension')
#     plt.ylabel('1-dimension')
#     plt.show()

# # Plotting functionality for 3d embeddings
# def plot_3d(e1, e2):
#     x1 = e1[:,0]
#     y1 = e1[:,1]
#     z1 = e1[:,2]
#     x2 = e2[:,0]
#     y2 = e2[:,1]
#     z2 = e2[:,2]
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x1, y1, z1, c='blue', label='e1')
#     ax.scatter(x2, y2, z2, c='red', label='e2')
#     ax.legend(loc='best')
#     ax.set_title('Graph Embeddings')
#     ax.set_xlabel('0-dimension')
#     ax.set_ylabel('1-dimension')
#     ax.set_zlabel('2-dimension')
#     plt.show()

# # Visualize 2d or 3d node embeddings
# def plot_embeddings(e1, e2):
#     n1 = e1.shape[0]
#     n2 = e2.shape[0]
#     if (n1 + n2 > 100):
#         print("Warning: plotting large number (", n1+n2, ") of data points")
#     assert(e1.shape[1] == e2.shape[1])
#     d = e2.shape[1]
#     assert(d == 2 or d == 3)
#     if d == 2:
#         plot_2d(e1, e2)
#     elif d == 3:
#         plot_3d(e1, e2)

# Normalize using max/min values across all dimensions
def normalize_all(e1, e2):
    max_val = max(e1.max(), e2.max())
    min_val = min(e1.min(), e2.min())
    e1_norm = (e1 - min_val) / (max_val - min_val)
    e2_norm = (e2 - min_val) / (max_val - min_val)
    return e1_norm, e2_norm

# Normalize each dimension independently (is this a good idea?? Close things could become farther apart??)
def normalize_ind(e1, e2):
    max_vals_e1 = e1.max(axis=0) # Get max vals along axis 0
    max_vals_e2 = e2.max(axis=0)
    max_vals = np.maximum(max_vals_e1, max_vals_e2) # Get max values, deciding between maxes of e1, e2 at each dimension

    min_vals_e1 = e1.min(axis=0) # Get min vals along axis 0
    min_vals_e2 = e2.min(axis=0)
    min_vals = np.minimum(min_vals_e1, min_vals_e2) # Get min values, deciding between mins of e1, e2 at each dimension

    e1_norm = (e1 - min_vals) / (max_vals - min_vals)
    e2_norm = (e2 - min_vals) / (max_vals - min_vals)
    return e1_norm, e2_norm

# Normalize graph embeddings to fall within unit hypercube (all dimensions in [0,1])
# Note: requires 2 graphs (to be compared) to retain relative similarities
def normalize_embeddings(e1, e2, mode='all'):
    if mode == 'all':
        return normalize_all(e1, e2)
    elif mode == 'ind':
        return normalize_ind(e1, e2)
    return None

# Generate binning of embedding e along dimension dim at a particular level
# MEMORY: requires numpy array of length 2^level
def get_binnings(e, dim, level):
    n_nodes, n_dims = e.shape
    assert(dim < n_dims)
    n_bins = 2**level
    return np.histogram(e[:, dim], n_bins, range=(0,1))[0]

# Return histogram intersection of 2 numpy arrays representing histograms
def hist_intersect(h1, h2):
    return np.sum(np.minimum(h1, h2))

# Compute I(H_g1^l, H_g2^l) from embeddings
# Added option to weight each dimension individually using numpy array of coefficient multipliers
# MEMORY: requires 2 numpy arrays of length 2^level
def compute_hist_intersect(e1, e2, level, dim_weighting=None):
    assert e1.shape[1] == e2.shape[1], 'Graph embeddings have different dimensionality'
    n_dims = e1.shape[1]
    hist_int = 0
    if dim_weighting is None:
        dim_weighting = np.ones(n_dims)
    for dim in range(n_dims):
        h1 = get_binnings(e1, dim, level)
        h2 = get_binnings(e2, dim, level)
        hist_int += dim_weighting[dim] * hist_intersect(h1, h2)
    return hist_int

# Exactly the same as compute_hist_intersect, but vectorized using numba. Requires bare numpy, no function calls
# MEMORY: requires 2 numpy arrays of length 2^level
@nb.njit
def compute_hist_intersect_vectorized(e1, e2, level, dim_weighting=None):
    assert e1.shape[1] == e2.shape[1], 'Graph embeddings have different dimensionality'
    n_dims = e1.shape[1]
    n_bins = 2**level
    hist_int = 0
    if dim_weighting is None:
        dim_weighting = np.ones(n_dims)
    for dim in range(n_dims):
        h1 = np.histogram(e1[:, dim], n_bins, range=(0,1))[0]
        h2 = np.histogram(e2[:, dim], n_bins, range=(0,1))[0]
        hist_int += dim_weighting[dim] * np.sum(np.minimum(h1, h2))
    return hist_int

# Compute similarity according to pyramid match graph kernel
# Time: O(n*d*L)
def pyramid_match_sim(e1, e2, L, regularize=False, vectorized=True, dim_weighting=None):

    # Check shapes
    n_nodes1 = e1.shape[0]
    assert n_nodes1 > 0, 'Graph 1 empty'
    n_nodes2 = e2.shape[0]
    assert n_nodes2 > 0, 'Graph 2 empty'

    # Put embeddings in unit hypercube
    e1, e2 = normalize_embeddings(e1, e2)

    # Begin with level L
    if vectorized:
        hist_int = compute_hist_intersect_vectorized(e1, e2, L, dim_weighting=dim_weighting)
    else:
        hist_int = compute_hist_intersect(e1, e2, L, dim_weighting=dim_weighting)
    sim = hist_int
    next_hist_int = hist_int

    # Loop over levels from small cells to large
    for level in reversed(list(range(L))): # Loop from level L-1 to 0
        weight = 1 / (2**(L - level))
        if vectorized:
            hist_int = compute_hist_intersect_vectorized(e1, e2, level, dim_weighting=dim_weighting)
        else:
            hist_int = compute_hist_intersect(e1, e2, level, dim_weighting=dim_weighting)
        sim += weight * (hist_int - next_hist_int)
        next_hist_int = hist_int

    # Option to regularize s.t. score is in interval (0,1]
    # NOTE: sim will never be zero (unless empty graph) b/c first level is entire hypercube
    if regularize:
        sim /= (n_nodes1 + n_nodes2)

    return sim

# Example of 2D plotting for different embedding normalizations
def test_2d_plot():
    e1 = np.array([[5, -2], [10, 3]])
    e2 = np.array([[6, 0], [-1, -5], [0, 0]])
    plot_embeddings(e1, e2)
    e1_norm, e2_norm = normalize_embeddings(e1, e2, mode='all')
    plot_embeddings(e1_norm, e2_norm)
    e1_norm, e2_norm = normalize_embeddings(e1, e2, mode='ind')
    plot_embeddings(e1_norm, e2_norm)

# Example of 3D plotting for different embedding normalizations
def test_3d_plot():
    e1 = np.array([[5, -2, 0], [10, 3, -1]])
    e2 = np.array([[6, 0, -1], [-1, -5, 9], [0, 0, 0]])
    plot_embeddings(e1, e2)
    e1_norm, e2_norm = normalize_embeddings(e1, e2, mode='all')
    plot_embeddings(e1_norm, e2_norm)
    e1_norm, e2_norm = normalize_embeddings(e1, e2, mode='ind')
    plot_embeddings(e1_norm, e2_norm)

# Test case with solutions worked out by hand (same embeddings as 2d plot example)
def test_2d_sim():
    e1 = np.array([[5, -2], [10, 3]])
    e2 = np.array([[6, 0], [-1, -5], [0, 0]])

    # Python:
    assert pyramid_match_sim(e1, e2, 2, vectorized=False) == 2.5, 'test_2d_sim, L=2'
    assert pyramid_match_sim(e1, e2, 1, vectorized=False) == 3, 'test_2d_sim, L=1'
    assert pyramid_match_sim(e1, e2, 0, vectorized=False) == 4, 'test_2d_sim, L=0'
    print("Success: test_2d_sim, L=0,1,2")

    # Vectorized (numba)
    assert pyramid_match_sim(e1, e2, 2, vectorized=True) == 2.5, 'test_2d_sim, L=2'
    assert pyramid_match_sim(e1, e2, 1, vectorized=True) == 3, 'test_2d_sim, L=1'
    assert pyramid_match_sim(e1, e2, 0, vectorized=True) == 4, 'test_2d_sim, L=0'
    print("Success: test_2d_sim vectorized, L=0,1,2")

# Demonstrate runtimes and linear scaling with n and d
def test_scaling():
    for n in [10**2]:
        for d in [10**1, 10**2, 10**3, 10**4, 10**5]:
            e1 = np.random.rand(n, d)
            e2 = np.random.rand(n, d)
            start = time.time()
            sim = pyramid_match_sim(e1, e2, 10, regularize=True)
            t = time.time() - start
            print('(n = ' + str(n) + ', d = ' + str(d) + '): ' + str(t))

# test_2d_plot()
# test_3d_plot()
# test_2d_sim()
test_scaling()