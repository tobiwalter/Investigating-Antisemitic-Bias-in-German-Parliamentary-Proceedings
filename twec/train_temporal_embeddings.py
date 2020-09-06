import os
import sys
from twec.twec import TWEC
from gensim.models.word2vec import Word2Vec
os.chdir('./..')

def main():
	aligner = TWEC(size=200, siter=5, diter=5, workers=1, ns=5, opath=f'{sys.argv[1]}_aligned')

	if sys.argv[1] == 'RT':
		#Train the compass
		aligner.train_compass("reichstag", overwrite=False) # keep an eye on the overwrite behaviour

		# Train the slices
		kaiserreich_1 = aligner.train_slice("reichstag/kaiserreich_1_processed", save=True)
		kaiserreich_2 = aligner.train_slice("reichstag/kaiserreich_2_processed", save=True)
		weimar = aligner.train_slice("reichstag/weimar_processed", save=True)

	elif sys.argv[1] == 'BRD':
		aligner.train_compass("bundestag", overwrite=False) # keep an eye on the overwrite behaviour

		cdu_1 = aligner.train_slice("bundestag/cdu_1", save=True)
		spd_1 = aligner.train_slice("bundestag/spd_1", save=True)
		cdu_2 = aligner.train_slice("bundestag/cdu_2", save=True)
		spd_2 = aligner.train_slice("bundestag/spd_2", save=True)
		cdu_3 = aligner.train_slice("bundestag/cdu_3", save=True)


if __name__ == "__main__":
	main()
