import models
biovec_model_path = 'streptomyces_avermitillis.model'
bv = models.load_protvec(biovec_model_path)
print(bv.to_vecs("ACCCTT"))
