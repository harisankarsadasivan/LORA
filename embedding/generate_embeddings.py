import networkx as nx
import subprocess
import random
from .biovec import models
from .dna2vec.multi_k_model import MultiKModel
import numpy as np

_2vec_params = {'p': 1,
                   'q': 1,
                   'num_walks': 10,
                   'walk_len': 80,
                   'dimensions ': 128,
                   'window_size': 10,
                   'workers': 8,
                   'iter' :1}

biovec_model_path = 'biovec/streptomyces_avermitillis.model'
dna2vec_model_path = 'dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
n = 7

node2vec_command = 'python2 node2vec/main.py --input embed_input.txt --output embed_output.emb'
struc2vec_command = 'python2 struc2vec/src/main.py --input embed_input.txt --output embed_output.emb'

def generate_embeddings(edge_list, contigs, struct_embedding_type, content_embedding_type):
    structure_embeddings = generate_structural_embeddings(edge_list, struct_embedding_type)
    content_embeddings = generate_content_embeddings(contigs, content_embedding_type)

    final_embeds = []

    for k,v in structure_embeddings.items():
        content_collapsed = np.stack((content_embeddings[k][0],content_embeddings[k][1],content_embeddings[k][2], v), axis=0)
        final_embeds.append(content_collapsed)
    final_embeds = np.asarray(final_embeds)
    print(final_embeds)

def generate_structural_embeddings(edge_list, embed_type):
    with open('embed_input.txt', 'w') as input:
        for start, end in edge_list:
            input.write(str(start) + " " + str(end) + "\n")

    if embed_type == 'node2vec':
        process = subprocess.Popen(node2vec_command.split(), stdout=subprocess.PIPE)
        process.communicate()

    elif embed_type == 'struc2vec':
        process = subprocess.Popen(struc2vec_command.split(), stdout=subprocess.PIPE)
        process.communicate()

    with open('embed_output.emb') as output:
        line = output.readline().split()
        n = int(line[0])
        d = int(line[1])
        structure =  {}
        for i in range(n):
            line = output.readline()
            indv_embed = line.split()
            embedding = np.zeros((d,))
            for j in range(1,d+1):
                embedding[j-1] = indv_embed[j]
            structure[int(indv_embed[0])] = embedding
    return structure


def generate_content_embeddings(contigs, embed_type):
    if embed_type == 'biovec':
        biovec = models.load_protvec(biovec_model_path)
        content = {}
        for node, contig in contigs.items():
            content[node] = biovec.to_vecs(contig)
        return content
    elif embed_type == 'dna2vec':
        mk_model = MultiKModel(dna2vec_model_path)
        content = {}
        for node, contig in contigs.items():
            if(len(contig) < n):
                content[node] = mk_model.vector(contig)
            else:
                content[node] = to_vecs(contig, mk_model)
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
