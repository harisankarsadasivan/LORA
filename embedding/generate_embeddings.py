import networkx as nx
from .biovec import models
from .dna2vec.multi_k_model import MultiKModel
from node2vec import node2vec
from struc2vec.src import struc2vec, graph
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np

_2vec_params = {'p': 1,
                   'q': 1,
                   'num_walks': 10,
                   'walk_len': 80,
                   'dimensions': 128,
                   'window_size': 10,
                   'workers': 8,
                   'iter': 1}

biovec_model_path = 'embedding/biovec/streptomyces_avermitillis.model'
dna2vec_model_path = 'embedding/dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
n = 8

def generate_embeddings(edge_list, contigs, repeats, struct_embedding_type, content_embedding_type, dimensions=100):
    structure_embeddings = generate_structural_embeddings(edge_list, struct_embedding_type, dimensions)
    content_embeddings = generate_content_embeddings(contigs, content_embedding_type, repeats)

    final_embeds = []


    for k,v in structure_embeddings.items():
        content_collapsed = np.concatenate((content_embeddings[k][0],content_embeddings[k][1],content_embeddings[k][2], v), axis=0)
        final_embeds.append(content_collapsed)
    return np.asarray(final_embeds)

def generate_structural_embeddings(edge_list, embed_type, dimensions):
    if embed_type == 'node2vec':
        G = nx.DiGraph()
        for start, end in edge_list:
            G.add_edge(start, end)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

        G = node2vec.Graph(G, True, _2vec_params['p'], _2vec_params['q'])
        G.preprocess_transition_probs()
        walks = G.simulate_walks(_2vec_params['num_walks'], _2vec_params['walk_len'])
        walks = [map(str, walk) for walk in walks]
        model = Word2Vec(walks, size=dimensions, window=_2vec_params['window_size'],
                         min_count=0, sg=1, workers=_2vec_params['workers'], iter=_2vec_params['iter'])

    elif embed_type == 'struc2vec':
        G = graph.load_edgelist_local(edge_list)
        G = struc2vec.Graph(G, True, _2vec_params['workers'], untilLayer=None)
        G.preprocess_neighbors_with_bfs()
        G.calc_distances_all_vertices(compactDegree=False)
        G.create_distances_network()
        G.preprocess_parameters_random_walk()

        G.simulate_walks(_2vec_params['num_walks'], _2vec_params['walk_len'])

        walks = LineSentence('random_walks.txt')
        model = Word2Vec(walks, size=dimensions, window=_2vec_params['window_size'], min_count=0, hs=1, sg=1,
                         workers=_2vec_params['workers'], iter=_2vec_params['iter'])

    vocab, vectors = model.wv.vocab, model.wv.vectors
    structure = {}
    for word, vocab_ in sorted(vocab.items(), key=lambda item: -item[1].count):
        row = vectors[vocab_.index]
        structure[int(word)] = row

    return structure


def generate_content_embeddings(contigs, embed_type, repeats):
    if embed_type == 'biovec':
        biovec = models.load_protvec(biovec_model_path)
        content = {}
        for node, contig in contigs.items():
            content[node] = biovec.to_vecs(contig) * repeats[node]
        return content
    elif embed_type == 'dna2vec':
        mk_model = MultiKModel(dna2vec_model_path)
        content = {}
        for node, contig in contigs.items():
            if len(contig) < n:
                content[node] = mk_model.vector(contig) * repeats[node]
            else:
                content[node] = to_vecs(contig, mk_model) * repeats[node]
        return content

def split_to_kmers(seq):
    a, b, c = zip(*[iter(seq)] * n), zip(*[iter(seq[1:])] * n), zip(*[iter(seq[2:])] * n)
    str_ngrams = []
    for ngrams in [a, b, c]:
        x = []
        for ngram in ngrams:
            x.append("".join(ngram))
        str_ngrams.append(x)
    return str_ngrams

def to_vecs(seq, mk_model):
    ngram_patterns = split_to_kmers(seq)

    protvecs = []
    for ngrams in ngram_patterns:
        ngram_vecs = []
        for ngram in ngrams:
            try:
                ngram_vecs.append(mk_model.vector(ngram))
            except:
                raise Exception("Model has never trained this n-gram: " + ngram)
        protvecs.append(sum(ngram_vecs))
    return protvecs
