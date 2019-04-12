from embedding.generate_embeddings import generate_embeddings
from phylo_tree.construct_dendrogram import get_pairwise_dist, build_dendrogram
import os
import numpy as np

data_location = 'data/flye_output/'
edge_list_location = '/20-repeat/edge_list'
content_list_location = '/20-repeat/node_list'
embedding_location = '/20-repeat/saved_embeddings.npy'
distance_mat_location = 'data/distance_mat.npy'

bacteria = os.listdir(data_location)
limit = 3 #np.inf # For debugging purposes
genomes = []
embeddings = []

for genome in bacteria:
    if not os.path.isfile(data_location + genome + embedding_location):
        edge_list = []
        with open(data_location + genome + edge_list_location) as edges:
            line = edges.readline()
            while line:
                nodes = line.split()
                edge_list.append((int(nodes[0]), int(nodes[1])))
                line = edges.readline()

        contigs = {}
        repeats = {}
        with open(data_location + genome + content_list_location) as content:
            content.readline()
            content.readline()
            line2 = content.readline()

            current_contig = ''
            current_node = None
            current_repeat = None

            while line2:
                if line2[0] == '>':
                    if current_node is not None:
                        contigs[current_node] = current_contig
                        repeats[current_node] = current_repeat
                    parts = line2.split()
                    current_node = int(parts[0][1:])
                    current_repeat = int(parts[3])
                else:
                    current_contig += line2.strip()
                line2 = content.readline()

            contigs[current_node] = current_contig
            repeats[current_node] = current_repeat

        genomes.append((genome, edge_list, contigs, repeats))
        print("Read files for " + genome)
    else:
        embeddings.append((genome, np.load(data_location + genome + embedding_location)))
        print("Loaded embeddings for " + genome)

emb_list = []
for genome, edges, contigs, repeats in genomes:
    if len(genomes) >= limit:
        break
    print('Generating embeddings for ' + genome)
    emb = generate_embeddings(edges, contigs, repeats, 'node2vec', 'dna2vec')
    emb_list.append(emb)
    embeddings.append((genome, emb, None)) # Name, embeddings, weight vector (optional)
    np.save(data_location + genome + embedding_location, emb)
    print('Done')
    
if not os.path.isfile(distance_mat_location):
    print('Calculating all pairwise distances')
    pairwise_dist = get_pairwise_dist(emb_list, 8)
    np.save(distance_mat_location, pairwise_dist)
else:
    print('Loading pairwise distances')
    pairwise_dist = np.load(distance_mat_location)
print('Done')
