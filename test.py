from embedding.generate_embeddings import generate_embeddings

edges = [(1,2),(2,3),(3,1),(4,2)]
contigs = {1:'AACTATGCTGCT', 2:'ATGTATGCATGC', 3:'TTTGTTTCTTTA', 4:'AACCTGTTAA'}
repeats = {1:1, 2:1, 3:1, 4:1}
generate_embeddings(edges, contigs,repeats,  'node2vec', 'dna2vec')

