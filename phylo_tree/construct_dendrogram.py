import numpy as np
import scipy.cluster.hierarchy as hier
# import matplotlib.pyplot as plt

import sys
sys.path.append('graph_kernel/')
from pyramid_match import pyramid_match_sim

# Calculate flattened (condensed) vector index from nxn matrix representation
def square_to_condensed(i, j, n):
    assert i != j, "no diagonal elements in condensed matrix"
    if i < j:
        i, j = j, i
    return int(n*j - j*(j+1)/2 + i - 1 - j)

# Calculate similarity for each pair of embeddings in input list 'embeddings'
# Condensed: output 1d vector of similarities, flattened upper-triangular matrix with diagonal excluded (assumption: sim(x,x)=1)
# Not Condensed: output full 2d matrix of similarities, sim(x,x)=1 IFF regularize is set to true.
# NOTE: Careful condensed default parameters.  Condensed representation via squareform in scipy is technically for distances
# and excludes diagonals assuming they are 0 (i.e. dist(x,x)=0).  If using condensed form, must know to check diagonals accordingly.
def get_pairwise_sim(embeddings, L, regularize=False, condensed=False):
    n = len(embeddings)
    if condensed: # Upper triangular representation flattened into vector
        out = np.zeros(int(n*(n-1)/2)) # n choose 2 indices
        for i in range(n):
            for j in range(i+1, n):
                out[square_to_condensed(i, j, n)] = pyramid_match_sim(embeddings[i], embeddings[j], L, regularize=regularize)

    else: # Full nxn matrix representation
        out = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                out[i, j] = pyramid_match_sim(embeddings[i], embeddings[j], L, regularize=regularize)
    
    if regularize:
        out /= np.max(out)
        
    return out

# Get pairwise distances for each pair of embeddings in input list 'embeddings'. Read get_pairwise_sim()
# If sim is in interval (0, 1] (regularized), then dist will be in interval [0, 1)
# If sim is in interval (0, inf) (not regularized), then dist will be in interval (-inf, 0)
# NOTE: Careful changing regularize parameter.  
def get_pairwise_dist(embeddings, L, regularize=True, condensed=True):
    if regularize: # Can assume sim is in range (0,1]
        return 1 - get_pairwise_sim(embeddings, L, regularize=regularize, condensed=condensed)
    else: # Sim is in range (0,inf]
        return -1 * get_pairwise_sim(embeddings, L, regularize=regularize, condensed=condensed)

# Wrapper which returns linkage structure of dendrogram
# Linkage is a string specifying the linkage method (e.g. single, complete, average, centroid, etc.)
def build_dendrogram(pairwise_dist, linkage):
    return hier.linkage(pairwise_dist, method=linkage, metric=None)

# Test case to create and visualize dendrogram
def test_dendrogram():
    e1 = np.array([[5, -2], [10, 3]])
    e2 = np.array([[6, 0], [-1, -5], [0, 0]])
    e3 = np.array([[9, -2], [7, 0], [-4, -2], [-2, 0]])
    e4 = np.array([[5, 2], [1, 5], [-9, -2], [7, 6], [1, -2]])
    embeddings = [e1, e2, e3, e4]

    pairwise_dist = get_pairwise_dist(embeddings, 3)
    print(pairwise_dist)
    single_link = build_dendrogram(pairwise_dist, 'single')
    print(single_link)
    # complete_link = build_dendrogram(pairwise_dist, 'complete')
    # average_link = build_dendrogram(pairwise_dist, 'average')
    # centroid_link = build_dendrogram(pairwise_dist, 'centroid')
    # plt.figure()
    # plt.subplot(2,2,1)
    # dn = hier.dendrogram(single_link)
    # plt.title('Single Link')
    # plt.subplot(2,2,2)
    # dn = hier.dendrogram(complete_link)
    # plt.title('Complete Link')
    # plt.subplot(2,2,3)
    # dn = hier.dendrogram(average_link)
    # plt.title('Average Link')
    # plt.subplot(2,2,4)
    # dn = hier.dendrogram(centroid_link)
    # plt.title('Centroid Link')
    # plt.show()


# Test case to show that condensed matrix representation matches what scipy creates/expects
def test_condensed_mat():
    from scipy.spatial.distance import squareform
    e1 = np.array([[5, -2], [10, 3]])
    e2 = np.array([[6, 0], [-1, -5], [0, 0]])
    e3 = np.array([[9, -2], [7, 0], [-4, -2], [-2, 0]])
    e4 = np.array([[5, 2], [1, 5], [-9, -2], [7, 6], [1, -2]])
    embeddings = [e1, e2, e3, e4]
    
    # squareform assumes that it's using distance and dist(x,x)=0, so must add identity to get sim(x,x)=1
    sqf_sim = squareform(get_pairwise_sim(embeddings, 3, regularize=True, condensed=True)) + np.identity(len(embeddings))
    mat_sim = get_pairwise_sim(embeddings, 3, regularize=True, condensed=False)
    assert np.allclose(sqf_sim, mat_sim), 'test_condensed_mat(): similarity'
    print('Success: test_condensed_mat() similarity')

    sqf_dist = squareform(get_pairwise_dist(embeddings, 3, regularize=True, condensed=True))
    mat_dist = get_pairwise_dist(embeddings, 3, regularize=True, condensed=False)
    assert np.allclose(sqf_dist, mat_dist), 'test_condensed_mat(): distance'
    print('Success: test_condensed_mat() distance')

# test_dendrogram()
# test_condensed_mat()