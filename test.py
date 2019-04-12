from embedding.generate_embeddings import generate_embeddings

edges = [(1,2),(2,3),(3,1),(4,2)]
contigs = {1:'AACTATGCTGCT', 2:'ATGTATGCATGC', 3:'TTTGTTTCTTTA', 4:'AACCTGTTAA'}
generate_embeddings(edges, contigs, 'node2vec', 'biovec')

