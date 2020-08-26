# -*- coding: utf-8 -*-
import re
import json
import pickle
import numpy as np
from pathlib import Path
import os
import codecs
from scipy import sparse
from typing import List, Dict, Tuple

DATA_FOLDER = Path('./data')
MODELS_FOLDER = Path('./models')
VOCAB_FOLDER = DATA_FOLDER / 'vocab'

# term sets

## target terms 

JEWISH_RT = ["jude", "juedisch", "judentum", "orthodox", "israel", "mosaisch","israelitisch","israelit", "rothschild", "talmud", "synagoge", "abraham", "rabbiner", "zionistisch"]

JEWISH_BRD = 'judentum, jude, juedisch, israel, israelisch, synagoge, koscher, orthodox, rabbiner, zentralrat'.split(', ')

CHRISTIAN_RT = ["christ", "christlich", "christentum", "katholizismus", "katholisch", "evangelisch", "evangelium", "auferstehung", "kirche" , "jesus", "taufe", "pfarrer", "bibel", "ostern"]

CHRISTIAN_BRD = 'christ, christlich, christentum, evangelisch, evangelium, jesus, katholisch, kirche, pfarrer, taufe, abendland'.split(', ')

PROTESTANT_BRD = "protestant, protestantisch, evangelisch, evangelium, landeskirche, kirchentag, ekd, landesbischof, lutherisch, diakonie".split(', ')

PROTESTANT_RT = ["protestant", "protestantisch", "protestantismus", "evangelisch", "evangelium", "landeskirche", "lutherisch", "evangelisch-lutherisch", "oberkirchenrat", "reformiert"]

CATHOLIC_BRD = "katholisch, katholik, papst, roemisch-katholisch, enzyklika, paepstlich, bischofskonferenz, dioezese, franziskus, kurie".split(', ')

CATHOLIC_RT = ["katholizismus", "katholisch", "katholik", "papst", "jesuiten", "ultramontanismus", "ultramontanen", "jesuitenorden", "roemisch-katholisch", "zentrumspartei"]

## attribute terms defining different attribute dimensions

# sentiment
PLEASANT = 'streicheln, Freiheit, Gesundheit, Liebe, Frieden, Freude, Freund, Himmel, loyal, Vergnuegen, Diamant, sanft, ehrlich, gluecklich, Regenbogen, Diplom, Geschenk, Ehre, Wunder, Sonnenaufgang, Familie, Lachen, Paradies, Ferien'.lower().split(', ') 

UNPLEASANT = 'Missbrauch, Absturz, Schmutz, Mord, Krankheit, Tod, Trauer, vergiften, stinken, Angriff, Katastrophe, Hass, verschmutzen, Tragoedie, Scheidung, Gefaengnis, Armut, haesslich, Krebs, toeten, faul, erbrechen, Qual'.lower().split(', ') 

# nationalism/patriotism

VOLKSTREU_RT = 'patriotisch, germanisch, vaterlaendisch, deutschnational, reichstreu, vaterlandsliebe, nationalgesinnt, nationalstolz, koenigstreu, volksgeist, nationalbewusstsein, volksbewusstsein, staatstreu, nationalgefuehl'.split(', ')

VOLKSUNTREU_RT = 'unpatriotisch, undeutsch, vaterlandslos, antideutsch, dissident, landesverraeter, reichsfeindlich, reichsfeind, deutschfeindlich, fremd, fremdlaendisch, nichtdeutsch, staatsfeindlich, heimatlos'.split(', ')

VOLKSTREU_BRD = 'patriotisch, vaterlandsliebe, germanisch, nationalbewusstsein, vaterlaendisch, nationalgefuehl, volkstum, patriotismus, patriot, staatstreu'.split(', ')

VOLKSUNTREU_BRD = 'nichtdeutsch, vaterlandslos, landesverraeter, antideutsch, heimatlos, separatistisch, staatsfeindlich, fremd, staatenlos, dissident'.split(', ')   

# economy

ECONOMIC_PRO = 'geben, großzuegigkeit, großzuegig, selbstlos, genuegsam, großmut, uneigennuetzig, sparsam, proletariat, armut, industriearbeiter'.split(', ')

ECONOMIC_CON = 'nehmen, gier, gierig, egoistisch, habgierig, habsucht, eigennuetzig, verschwenderisch, bourgeoisie, wohlstand, bankier, wucher'.split(', ')

# conspiracy

CONSPIRATORIAL_PRO = 'loyal, kamerad, ehrlichkeit, ersichtlich, aufrichtig, vertrauenswuerdig, wahr, ehrlich, unschuldig, freundschaftlich, hell, zugaenglich, machtlos, ohnmacht, untertan'.split(', ')

CONSPIRATORIAL_CON = 'illoyal, spitzel, verrat, geheim, betruegerisch, hinterlistig, unwahr, zweifelhaft, verbrecher, bedrohlich, dunkel, geheimnis, einflussreich, weltmacht, herrschaft, verschwoerung'.split(', ')

# ethics

ETHIC_PRO = 'bescheiden, sittlich, anstaendig, tugendhaft, charakterfest, wuerdig, treu, moralisch, ehrlich, gesittet, gewissenhaft, vorbildlich'.split(', ')

ETHIC_CON = 'unbescheiden, unsittlich, unanstaendig, luestern, korrupt, unwuerdig, untreu, unmoralisch, unehrlich, verdorben, gewissenlos, barbarisch'.split(', ')

# religion

RELIGIOUS_PRO = 'glaeubige, geistlich, engel, heilig, fromm, geheiligt, goettlich, ehrwuerdig, treu, glaeubig, religioes'.split(', ')

RELIGIOUS_CON = 'atheist, weltlich, teufel, irdisch, atheistisch, heidnisch, gottlos, verflucht, untreu, unglaeubig, irreligioes, gotteslaesterung'.split(', ')

# racism
RACIST_PRO = 'normal, ueberlegenheit, gleichheit, angenehm, freundlich, ehrenwert, sympathie, akzeptiert, besser, national, rein, ueberlegen, sauber, ehrenhaft'.split(', ')

RACIST_CON = 'seltsam, unterlegenheit, ungleichheit, unangenehm, boshaft, schaendlich, hass, abgelehnt, schlechter, fremdlaendisch, unrein, unterlegen, schmutzig, verseucht, schaedlich, niedertraechtig'.split(', ')

# unipolar attribute sets - can only be used for ECT 

OUTSIDER_WORDS = 'unaufrichtig, komisch, boshaft, unberechenbar, primitiv, beaengstigend, hinterlistig, energisch, truegerisch, \
neidisch, gierig, abscheulich, verachtenswert, brutal, ungeheuer, berechnend, grausam, gemein, intolerant, aggressiv'\
.lower().split(', ') 

JEWISH_STEREOTYPES_NOUNS = 'Gier, Wucherer, Drueckeberger, Kriegsgewinnler, Macht, Einfluss, Kriegstreiber, Luegner, \
Weltherrschaft, Kommunismus, Kapitalismus, Liberalismus, Aussenseiter'.lower().split(', ')

JEWISH_STEREOTYPES_CHARACTER = 'egoistisch, fremd, dekadent, hasserfuellt, habgierig, geldgierig, penetrant, hinterlistig, \
intellektuell, pervers, hinterhaeltig, betruegerisch, gebeugt, bucklig'.split(', ')

JEWISH_STEREOTYPES_POLITICAL = 'liberalistisch, modern, materialistisch, liberal, undeutsch, unpatriotisch, saekular, \
sozialistisch, links, bolschewistisch'.split(', ')

JEWISH_OCCUPATIONS = 'Pfandleiher, Geldleiher, Kaufmann, Haendler, Bankier, Finanzier, Steuereintreiber, Zoellner, \
Troedelhaendler'.lower().split(', ') 



class CreateSlice:
    """
    Read in pre-processed files before feeding them to word2vec model

    Attributes
    ----------
    dirname (str): The name of the directory storing the protocols
    protocol_type (str): whether to process a collection of Reichstag or BRD protocols
    """
    
    def __init__(self, dirname: str, profiling=False):
        self.dirname = str(DATA_FOLDER / dirname)
        self.profiling = profiling

    def __iter__(self):
        for fn in os.listdir(self.dirname):
            text = open(os.path.join(self.dirname, fn), encoding='utf-8').readlines()
            # for corpus profiling
            if self.profiling:
                yield text
            else: 
                for sentence in text:
                    yield sentence.split()

class CreateCorpus:
    """
    Read in pre-processed files before feeding them to word2vec model

    Attributes
    ----------
    top_dir (str): The name of the top directory storing the protocols
    protocol_type (str): whether to process a collection of Reichstag or BRD protocols
    """

    def __init__(self,top_dir, profiling=False):
        self.top_dir = str(DATA_FOLDER / top_dir)
        self.profiling = profiling
    def __iter__(self):
        """Iterate over all documents, yielding a document (=list of utf-8 tokens) at a time."""
        for root, dirs, files in os.walk(self.top_dir):
            for file in filter(lambda file: file.endswith('.txt'), files):
                text = open(os.path.join(root, file), encoding='utf-8').readlines()
                # for the purpose of corpus profiling
                if self.profiling:
                    yield text
                else:
                    for line in text:
                        yield line.split()

def save_corpus(corpus: List , corpus_path: str):
    """Save each protocol from a corpus to disk."""
    if not (DATA_FOLDER / corpus_path).exists():
        os.makedirs(DATA_FOLDER / corpus_path)
    for num,doc in enumerate(corpus):
        write_lines((DATA_FOLDER / corpus_path / f'{num+1}_sents.txt'), doc)

def save_vocab(model, filepath: str):
    """Save the word:index mappings from word2vec to disk."""
    words = sorted([w for w in model.wv.vocab], key=lambda w: model.wv.vocab.get(w).index)
    index = {w: i for i, w in enumerate(words)}
    with codecs.open(str(VOCAB_FOLDER / filepath) + '.json',"w", encoding='utf-8') as f:
        f.write(json.dumps(index))

def write_lines(path: str, lines: List):
    """Write document lines stored as a list to disk"""
    f = codecs.open(path, "w", encoding='utf8')
    for l in lines:
        f.write(str(l) + "\n")
    f.close()

def filter_terms(target_set: List, input_repr):
    """Filter out target terms that do not reach the minimum count. """
    return [word for word in target_set if word in input_repr]

def create_attribute_sets(dict, kind, incl_unipolar=False):
    """
    Create all attribute sets for a specific time period

    :param input_repr: input representation of the text 
    :param kind: version of attributes to create - either for RT or BRD 
    :param incl_unipolar: whether to include unipolar attribute sets 
    """
    attribute_sets = {
    'sentiment_pro' : filter_terms(PLEASANT, dict),
    'sentiment_con' : filter_terms(UNPLEASANT, dict),
	'economic_pro' : filter_terms(ECONOMIC_PRO, dict),
	'economic_con' : filter_terms(ECONOMIC_CON, dict),
	'conspiratorial_pro' : filter_terms(CONSPIRATORIAL_PRO, dict),
	'conspiratorial_con' : filter_terms(CONSPIRATORIAL_CON, dict),
    'religious_pro' : filter_terms(RELIGIOUS_PRO, dict),
    'religious_con' : filter_terms(RELIGIOUS_CON, dict),
    'racist_pro' : filter_terms(RACIST_PRO, dict),
    'racist_con' : filter_terms(RACIST_CON, dict),
    'ethic_pro' : filter_terms(ETHIC_PRO, dict),
    'ethic_con' : filter_terms(ETHIC_CON, dict)}

    if incl_unipolar:

        attribute_sets['outsider_words'] = filter_terms(OUTSIDER_WORDS, dict)
        attribute_sets['jewish_occupations'] = filter_terms(JEWISH_OCCUPATIONS, dict),
        attribute_sets['jewish_nouns'] = filter_terms(JEWISH_STEREOTYPES_NOUNS, dict),
        attribute_sets['jewish_character'] = filter_terms(JEWISH_STEREOTYPES_CHARACTER, dict),
        attribute_sets['jewish_political'] = filter_terms(JEWISH_STEREOTYPES_POLITICAL, dict)

    if kind == 'BRD':
        attribute_sets['patriotism_pro'] = filter_terms(VOLKSTREU_BRD, dict)
        attribute_sets['patriotism_con'] = filter_terms(VOLKSUNTREU_BRD, dict)
    elif kind == 'RT':            
        attribute_sets['patriotism_pro'] = filter_terms(VOLKSTREU_RT, dict)
        attribute_sets['patriotism_con'] = filter_terms(VOLKSUNTREU_RT, dict)
    else: 
        raise ValueError('parameter ''kind'' must be specified to either RT for Reichstag proceedings or BRD for Bundestag proceedings.')

    return attribute_sets

def convert_attribute_set(dimension):
    if dimension in ('sentiment', 'random'):
      return ('sentiment_pro', 'sentiment_con')
    elif dimension == 'sentiment_flipped':
      return ('sentiment_con', 'sentiment_pro')
    elif dimension == 'patriotism':
      return ('patriotism_pro', 'patriotism_con')
    elif dimension == 'economic':
      return ('economic_pro', 'economic_con')
    elif dimension == 'conspiratorial':
      return ('conspiratorial_pro', 'conspiratorial_con')
    elif dimension == 'racist':
      return ('racist_pro', 'racist_con')
    elif dimension == 'religious':
      return ('religious_pro', 'religious_con')
    elif dimension == 'ethic':
      return ('ethic_pro', 'ethic_con')

      
def create_target_sets(dict, kind): 
    """
    Create all target sets for this study
    :param dict: trained word vectors 
    :param kind: kind of attributes to create - either RT or BRD 
    """
    if kind == 'RT':
        target_sets = {
        'jewish' : filter_terms(JEWISH_RT, dict),
        
        'christian' : filter_terms(CHRISTIAN_RT, dict),
        
        'catholic' : filter_terms(CATHOLIC_RT, dict),
        
        'protestant' : filter_terms(PROTESTANT_RT, dict)
                    }
    elif kind == 'BRD':
        target_sets = {
        'jewish' : filter_terms(JEWISH_BRD, dict),
        
        'christian' : filter_terms(CHRISTIAN_BRD, dict),
        
        'catholic' : filter_terms(CATHOLIC_BRD, dict),
        
        'protestant' : filter_terms(PROTESTANT_BRD, dict)
                    }
    else:
        print('parameter ''kind'' must be specified to either RT for Reichstag proceedings or BRD for Bundestag proceedings.')
    # Join them together to form bias words
    return target_sets

def inverse(matrix):
  return np.linalg.inv(matrix)

def load_embedding_dict(vocab_path="", vector_path="", dict_path="", glove=False, postspec=False):
  """Load embedding dict for WEAT test

  :param vocab_path:
  :param vector_path:
  :return: embd_dict
  """

  if dict_path != "":
    embd_dict = utils.load_dict(dict_path)
    return embd_dict
  else:
    embd_dict = {}
    vocab = load_vocab(vocab_path)
    vectors = load_vectors(vector_path)
    for term, index in vocab.items():
      embd_dict[term] = vectors[index]
    assert len(embd_dict) == len(vocab)
    return embd_dict

def load_lines(filepath):
  return [l.strip() for l in list(codecs.open(filepath, "r", encoding = 'utf8', errors = 'replace').readlines(sizehint=None))]

def load_vocab(path, inverse = False):
  vocab = json.load(open(path,"r"))
  if inverse:
    vocab_inv = {v : k for k, v in vocab.items()}
    return vocab, vocab_inv
  else:
    return vocab

def load_vectors(path, normalize = False):
  if path.endswith('npz'):
    vecs = sparse.load_npz(path).toarray()
  else:
    vecs = np.load(path)
  if normalize:
    vecs_norm = vecs / np.transpose([np.linalg.norm(vecs, 2, 1)])
    return vecs, vecs_norm
  else:
    return vecs

def load_embeddings(path, word2vec=True, rdf2vec=False):
    """
    >>> load_embeddings("/work/anlausch/glove_twitter/glove.twitter.27B.200d.txt")
    :param path:
    :param word2vec:
    :param rdf2vec:
    :return:
    """
    embbedding_dict = {}
    if word2vec == False and rdf2vec == False:
        with codecs.open(path, "rb", "utf8", "ignore") as infile:
            for line in infile:
                try:
                    parts = line.split()
                    word = parts[0]
                    nums = [float(p) for p in parts[1:]]
                    embbedding_dict[word] = nums
                except Exception as e:
                    print(line)
                    continue
        return embbedding_dict
    elif word2vec == True:
        #Load Google's pre-trained Word2Vec model.
        if os.name != 'nt':
            model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
            # model = gensim.models.KeyedVectors.load_word2vec_format(path, encoding = 'utf-8', unicode_errors = 'ignore', binary=True)
        else:
            try:
              model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
            except UnicodeDecodeError:
              model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)


            # model = gensim.models.Word2Vec.load_word2vec_format(path, encoding = 'utf-8', binary=True, unicode_errors= 'ignore')
        return model
    elif rdf2vec == True:
        #Load Petars model.
        model = gensim.models.Word2Vec.load(path)
    return model



