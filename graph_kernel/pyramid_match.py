import numpy as np

# Normalize graph embeddings to fall within unit hypercube
# Note: requires 2 graphs (to be compared) to retain relative similarities
def normalize_embeddings(e1, e2):
    return e1, e2

# Generate histogram for embedding e at a particular level
def get_histogram(e, level):
    return np.zeros(10)

# Compute similarity according to pyramid match graph kernel
# Labels optional
def pyramid_match_sim(e1, e2, num_levels, labels=None):

    # Put embeddings in unit hypercube
    e1, e2 = normalize_embeddings(e1, e2)

    # Loop over levels from small cells to large
    for level in reversed(list(range(num_levels))): # Loop from level L to 0
        h1 = get_histogram(e1, level)
        h2 = get_histogram(e2, level)

    return 0

# Simple test case
def test_sim():
    e1 = np.array([[0.1, 0.1], [0.1, 0.9]])
    e2 = np.array([[0.5, 0.1], [0.7, 0.9]])
    sim = pyramid_match_sim(e1, e2, 2)

test_sim()