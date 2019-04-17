import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_location = 'data/flye_output/'
edge_list_location = '/20-repeat/edge_list'
content_list_location = '/20-repeat/node_list'
embedding_location = '/20-repeat/saved_embeddings_%s_%s_%d.npy'
distance_mat_location = 'data/distance_mat_%s_%s_%d.npy'

bacteria_list = os.listdir(data_location)
bacteria = np.random.choice(bacteria_list)
print(bacteria)
emb1 = np.load(data_location + bacteria + (embedding_location % ('node2vec', 'biovec', 100)))
emb2 = np.load(data_location + bacteria + (embedding_location % ('node2vec', 'dna2vec', 100)))
print(np.allclose(emb1, emb2))
print(emb1.shape)
print(emb2.shape)