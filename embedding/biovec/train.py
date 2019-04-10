import biovec

pv = biovec.ProtVec("big_ass_genome.fa")
pv["GCT"]
pv.to_vecs("ATGCGCGCTA")
pv.save("new_model")
