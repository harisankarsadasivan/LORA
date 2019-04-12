from embedding.generate_embeddings import generate_embeddings
import os

data_location = 'data/flye_output/'
edge_list_location = '/20-repeat/edge_list'
content_list_location = '/20-repeat/node_list'

bacteria = os.listdir(data_location)

genomes = []

for genome in bacteria:
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

embeddings = []

for genome, edges, contigs, repeats in genomes:
    print('Generating embeddings for ' + genome)
    embeddings.append(generate_embeddings(edges, contigs, repeats, 'node2vec', 'dna2vec'))
    print('Done')

