# -*- coding: utf-8 -*-
import pickle
import os 
import sys
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from pathlib import Path
from representations.utils import load_corpus, filter_target_set, create_target_sets

MOD_DIR = os.path.dirname(os.path.abspath(__file__))
data_folder = Path(MOD_DIR) / 'models'

JEWISH = 'rabbi, synagoge, koscher, sabbat, orthodox, judentum, jude, jüdisch, mose, talmud, israel, abraham, zionistisch, zionistische, zionismus, israelitisch, israelis, israelisch, israels, israeli, mosaisch, rabbiner, 	beschneidung, zentralrat, holocaust, leviten'.split(', ')

CHRISTIAN = 'taufe, katholizismus, auferstehung, christentum, evangelisch, evangelium, jesus, christ, christlich, katholisch, kirche, pfarrer, ostern, bibel, christlich-abendländisch, christenheit, abendland, weltkirchenrat'.split(', ')

CATHOLIC = 'katholizismus, katholisch, katholik, papst, römisch-katholisch, jesuiten, jesuitenorden, ultramontanismus, ultramontanen, zentrumspartei, pius, enzyklika, päpstliche, diözese, bischofskonferenz, kurie, franziskus'.split(', ')

PROTESTANT = 'protestant, protestantisch, protestantismus, evangelisch, evangelium, landeskirche, oberkirchenrat, lutherisch, evangelisch-lutherisch, reformiert, lutheraner, kirchentag, landesbischof, diakonie, ekd, evangelisch, baptist, anglikaner, anglikanisch, uniert, diakon, diakoniewerk'.split(', ')

VÖLKISCH = 'patriotisch, vaterlandsliebe, volksbewußtsein, volksgeist, germanische, nationalbewußtsein, vaterländisch, reichstreu, nationalgesinnt, nationalstolz, deutschnational, königstreu, volksdeutsch, nationalgefühl, abendländisch, patriot, patrioten, patriotismus, volksbewusstsein, volksempfinden, volkstum'.split(', ')

UNVÖLKISCH = 'nichtdeutsch, fremdländisch, fremd, undeutsch, vaterlandslos, reichsfeind, landesverräter, reichsfeindlich, unpatriotisch, antideutsch, deutschfeindlich, umstürzler, heimatlos, staatenlos, nichtdeutsch, separatistisch, separatistische, staatsfeindlich, staatsfeindliche, klassenkämpferisch, sozialschmarotzer'.split(', ')		

# NEW_WORDS = 'theil, teil, nothwendig, notwendig, berathung, beratung, landwirthschaft, landwirtschaft, thatsächlich, tatsache, nöthig, nötig, ergiebt, ergibt, rot, roth, thun, endgiltig, endgültig, forcirt, forciert, entwickelung, entwicklung, gleichgiltig, gleichgültig, konstatire, konstatiere, commission, kommission, controle, kontrolle, vertheidigung, verteidigung, qualification, qualifikation'.split(', ')


# NEW_WORDS = 'modern, antimodern, volkstum, volksthum, barbarei, entartung, entarthung, kulturkampf, kulturprotestantismus'.split(', ')

# model =KeyedVectors.load(str(data_folder / sys.argv[1]))
# model_vocab = model.wv.vocab
#Load all models and retrieve their vocab
# slice_1 = KeyedVectors.load(str(data_folder /'w2v_slice_1_lem'))
# vocab_1 = slice_1.wv.vocab

# slice_2 = KeyedVectors.load(str(data_folder /'w2v_slice_2_lem'))
# vocab_2 = slice_2.wv.vocab

slice_3 = KeyedVectors.load(str(data_folder /'w2v_slice_3_lem'))
vocab_3 = slice_3.wv.vocab

slice_4 = KeyedVectors.load(str(data_folder /'w2v_slice_4_lem'))
vocab_4 = slice_4.wv.vocab

# slice_5 = KeyedVectors.load(str(data_folder /'w2v_slice_5_lem'))
# vocab_5 = slice_5.wv.vocab


def create_word_counts_df(target_words):

	counts = pd.DataFrame(index = ['slice_3','slice_4'], columns= target_words)

	# counts.loc['slice_1'] = [vocab_1.get(word).count if word in vocab_1 else 0 for word, count in counts.loc['slice_1'].iteritems()]

	# counts.loc['slice_2'] = [vocab_2.get(word).count if word in vocab_2 else 0 for word, count in counts.loc['slice_2'].iteritems()]

	counts.loc['slice_3'] = [vocab_3.get(word).count if word in vocab_3 else 0 for word, count in counts.loc['slice_3'].iteritems()]

	counts.loc['slice_4'] = [vocab_4.get(word).count if word in vocab_4 else 0 for word, count in counts.loc['slice_4'].iteritems()]

	# counts.loc['slice_5'] = [vocab_5.get(word).count if word in vocab_5 else 0 for word, count in counts.loc['slice_5'].iteritems()]

	return counts 

völkisch_counts = create_word_counts_df(VÖLKISCH)
unvölkisch_counts = create_word_counts_df(UNVÖLKISCH)
jewish_counts = create_word_counts_df(JEWISH)
christian_counts = create_word_counts_df(CHRISTIAN)
catholic_counts = create_word_counts_df(CATHOLIC)
protestant_counts = create_word_counts_df(PROTESTANT)
# new_words_counts = create_word_counts_df(NEW_WORDS)

path_to_bundestagsprotokolle = 'data\\word_counts\\Bundestagsprotokolle'

# if not os.path.exists(os.path.join(MOD_DIR, path_to_bundestagsprotokolle)):
# 	os.mkdir(os.path.join(MOD_DIR, path_to_bundestagsprotokolle))

jewish_counts.to_csv(os.path.join(path_to_bundestagsprotokolle,f'{sys.argv[1]}_jewish_counts_harm.csv'))
christian_counts.to_csv(os.path.join(path_to_bundestagsprotokolle,f'{sys.argv[1]}_christian_counts_harm.csv'))
catholic_counts.to_csv(os.path.join(path_to_bundestagsprotokolle,f'{sys.argv[1]}_catholic_counts_harm.csv'))
protestant_counts.to_csv(os.path.join(path_to_bundestagsprotokolle,f'{sys.argv[1]}_protestant_counts_harm.csv'))
völkisch_counts.to_csv(os.path.join(path_to_bundestagsprotokolle,f'{sys.argv[1]}_völkisch_counts_harm.csv'))
unvölkisch_counts.to_csv(os.path.join(path_to_bundestagsprotokolle,f'{sys.argv[1]}_unvölkisch_counts_harm.csv'))
# new_words_counts.to_csv('data/word_counts/new_word_counts.csv')

# new_words_counts.to_csv('data/word_counts/_word_counts.csv')
