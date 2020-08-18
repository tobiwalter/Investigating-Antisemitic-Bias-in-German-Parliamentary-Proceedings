import os
from twec.twec import TWEC
from gensim.models.word2vec import Word2Vec
os.chdir('./..')

aligner = TWEC(size=200, siter=5, diter=5, workers=1, ns=5, opath='entire')

#train the compass
aligner.train_compass("compass_new/harm", overwrite=False) # keep an eye on the overwrite behaviour

# Train the slices

#kaiserreich_1 = aligner.train_slice("reichstag/kaiserreich_1_processed", save=True)
#kaiserreich_2 = aligner.train_slice("reichstag/kaiserreich_2_processed", save=True)
#weimar = aligner.train_slice("reichstag/weimar_processed", save=True)
#cdu_1 = aligner.train_slice("compass_new/slice_1_new", save=True)
spd_1 = aligner.train_slice("compass_new/slice_2_new", save=True)
cdu_2 = aligner.train_slice("compass_new/slice_3_new", save=True)
spd_2 = aligner.train_slice("compass_new/slice_4_new", save=True)
cdu_3 = aligner.train_slice("compass_new/slice_5_new", save=True)
