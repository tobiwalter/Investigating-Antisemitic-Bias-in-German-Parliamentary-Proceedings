import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from twec.twec import TWEC
from gensim.models.word2vec import Word2Vec

aligner = TWEC(size=200, siter=5, diter=5, workers=1, ns=5)

#train the compass
aligner.train_compass("examples/reichstag", overwrite=False) # keep an eye on the overwrite behaviour

# Train the slices
#slice_1 = aligner.train_slice("examples/reichstag/rt_slice_1_processed", save=True)
slice_2 = aligner.train_slice("examples/reichstag/rt_slice_2_processed", save=True)
slice_3 = aligner.train_slice("examples/reichstag/rt_slice_3_processed", save=True)
slice_4 = aligner.train_slice("examples/reichstag/rt_slice_4_processed", save=True)
slice_5 = aligner.train_slice("examples/reichstag/rt_slice_5_processed", save=True)


