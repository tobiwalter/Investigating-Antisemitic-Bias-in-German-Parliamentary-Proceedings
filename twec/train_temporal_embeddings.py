import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from twec.twec import TWEC
from gensim.models.word2vec import Word2Vec

aligner = TWEC(size=200, siter=5, diter=5, workers=1, ns=10)

#train the compass
aligner.train_compass("examples/compass", overwrite=True) # keep an eye on the overwrite behaviour

# Train the slices
slice_1 = aligner.train_slice("examples/compass/slice_1", save=True)
slice_2 = aligner.train_slice("examples/compass/slice_2", save=True)
slice_3 = aligner.train_slice("examples/compass/slice_3", save=True)
slice_4 = aligner.train_slice("examples/compass/slice_4", save=True)
slice_5 = aligner.train_slice("examples/compass/slice_5", save=True)

