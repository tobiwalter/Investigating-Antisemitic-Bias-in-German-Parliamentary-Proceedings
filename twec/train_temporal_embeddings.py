import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from twec.twec import TWEC
from gensim.models.word2vec import Word2Vec

aligner = TWEC(size=200, siter=10, diter=10, workers=4,
	test='.\\examples\\testing', ns=0, hs=1)


#train the compass
aligner.train_compass("examples/training/compass.txt", overwrite=False) # keep an eye on the overwrite behaviour

# slice_one = aligner.train_slice("examples/training_org/arxiv_14.txt", save=True)
# slice_two = aligner.train_slice("examples/training_org/arxiv_9.txt", save=True)
# Train the slices
# slice_1 = aligner.train_slice("examples/training/slice_1_train.txt", save=True)
# slice_2 = aligner.train_slice("examples/training/slice_2_train.txt", save=True)
# slice_3 = aligner.train_slice("examples/training/slice_3_train.txt", save=True)
# slice_4 = aligner.train_slice("examples/training/slice_4_train.txt", save=True)
# slice_5 = aligner.train_slice("examples/training/slice_5_train.txt", save=True)
# Evaluate 

aligner.evaluate() 