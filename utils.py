import re
import json
import pickle
import numpy as np
from pathlib import Path
import os
import codecs
# import numba

DATA_FOLDER = Path('./data')
MODELS_FOLDER = Path('./models')
VOCAB_FOLDER = DATA_FOLDER / 'vocab'

# term sets

## target terms 

JEWISH_RT = ["jude", "juedisch", "judentum", "orthodox", "israel", "mosaisch","israelitisch","israelit", "rothschild", "talmud", "synagoge", "abraham", "rabbiner", "zionistisch"]

JEWISH_BRD = 'judentum, jude, juedisch, israel, israels, israeli, synagoge, koscher, orthodox, rabbiner, zentralrat'.split(', ')

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

VOLKSTREU_RT = 'patriotisch, vaterlandsliebe, volksbewusstsein, volksgeist, germanisch, deutschnational, nationalbewusstsein, vaterlaendisch, reichstreu, nationalgesinnt, nationalstolz, koenigstreu'.split(', ')

VOLKSUNTREU_RT = 'nichtdeutsch, fremdlaendisch, fremd, undeutsch, vaterlandslos, reichsfeind, landesverraeter, reichsfeindlich, unpatriotisch, antideutsch, deutschfeindlich, umstuerzler'.split(', ')   

VOLKSTREU_BRD = 'patriotisch, vaterlandsliebe, germanisch, nationalbewusstsein, vaterlaendisch, nationalgefuehl, volkstum, patriotismus, patriot'.split(', ')

VOLKSUNTREU_BRD = 'nichtdeutsch, vaterlandslos, landesverraeter, antideutsch, heimatlos, separatistisch, staatsfeindlich, fremd, staatenlos'.split(', ')   

# economy

ECONOMIC_PRO = 'geben, großzuegigkeit, großzuegig, selbstlos, genuegsam, großmut, uneigennuetzig, sparsam, proletariat, armut, industriearbeiter'.split(', ')

ECONOMIC_CON = 'nehmen, gier, gierig, egoistisch, habgierig, habsucht, eigennuetzig, verschwenderisch, bourgeoisie, wohlstand, boerse, wucher'.split(', ')

# conspiracy

CONSPIRATORIAL_PRO = 'loyal, kamerad, ehrlichkeit, ersichtlich, aufrichtig, vertrauenswuerdig, wahr, ehrlich, unschuldig, freundschaftlich, hell, zugaenglich, machtlos, ohnmacht, untertan'.split(', ')

CONSPIRATORIAL_CON = 'illoyal, spitzel, verrat, geheim, betruegerisch, hinterlistig, unwahr, zweifelhaft, verbrecher, bedrohlich, dunkel, geheimnis, einflussreich, weltmacht, herrschaft, verschwoerung'.split(', ')

# ethics

ETHIC_PRO = 'bescheiden, sittlich, anstaendig, tugendhaft, charakterfest, wuerdig, treu, moralisch, ehrlich, gesittet, gewissenhaft, vorbildlich'.split(', ')

ETHIC_CON = 'unbescheiden, unsittlich, unanstaendig, luestern, korrupt, unwuerdig, untreu, unmoralisch, unehrlich, verdorben, gewissenlos, barbarisch'.split(', ')

# religion

RELIGIOUS_PRO = 'glaeubige, geistlich, engel, heilig, fromm, geheiligt, goettlich, ehrwuerdig, treu, glaeubig, religioes'.split(', ')

RELIGIOUS_CON = 'atheist, weltlich, teufel, irdisch, atheistisch, heidnisch, gottlos, verflucht, schaendlich, untreu, unglaeubig, irreligioes, gotteslaesterung'.split(', ')

# racism
RACIST_PRO = 'normal, ueberlegenheit, gleichheit, angenehm, freundlich, ehrenwert, sympathie, akzeptiert, besser, national, rein, ueberlegen, sauber, ehrenhaft'.split(', '

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
    
    def __init__(self, dirname, profiling=False):
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
    """
    def __init__(self,top_dir, profiling=False):
        self.top_dir = str(DATA_FOLDER / top_dir)
        self.profiling = profiling
    """Iterate over all documents, yielding a document (=list of utf8 tokens) at a time."""
    def __iter__(self):
        for root, dirs, files in os.walk(self.top_dir):
            for file in filter(lambda file: file.endswith('.txt'), files):
                text = open(os.path.join(root, file), encoding='utf-8').readlines()
                # for corpus profiling
                if self.profiling:
                    yield text
                else:
                    for line in text:
                        yield line.split()


def save_corpus(corpus, corpus_path):
    if not (DATA_FOLDER / corpus_path).exists():
        os.makedirs(DATA_FOLDER / corpus_path)
    for num,doc in enumerate(corpus):
        write_lines((DATA_FOLDER / corpus_path / f'{num+1}_sents.txt'), doc)

def save_vocab(model, filepath):
    words = sorted([w for w in model.wv.vocab], key=lambda w: model.wv.vocab.get(w).index)
    index = {w: i for i, w in enumerate(words)}
    json_repr = json.dumps(index)
    with codecs.open(str(VOCAB_FOLDER / filepath) + '.json',"w", encoding='utf-8') as f:
        f.write(json_repr)


def write_lines(path, list):
    f = codecs.open(path, "w", encoding='utf8')
    for l in list:
        f.write(str(l) + "\n")
    f.close()

def filter_target_set(target_set, word_vectors):
    """
    Filter out all target terms that did not reach the minimum count and are thus not included in the embeddings space
    """
    return [word for word in target_set if word in word_vectors]

def create_attribute_sets(word_vectors, kind):
    """
    Create all attribute sets for this study

    :param word_vectors: trained word vectors 
    :param kind: kind of attributes to create - either RT or BRD 
    """
    attribute_sets = {
        'pleasant' : filter_target_set(PLEASANT, word_vectors),
        'unpleasant' : filter_target_set(UNPLEASANT, word_vectors),
	'economic_pro' : filter_target_set(ECONOMIC_PRO, word_vectors),
	'economic_con' : filter_target_set(ECONOMIC_CON, word_vectors),
	'conspiratorial_pro' : filter_target_set(CONSPIRATORIAL_PRO, word_vectors),
	'conspiratorial_con' : filter_target_set(CONSPIRATORIAL_CON, word_vectors),
    'religious_pro' : filter_target_set(RELIGIOUS_PRO),
    'religious_con' : filter_target_set(RELIGIOUS_CON),
    'racist_pro' : filter_target_set(RACIST_RPO),
    'racist_con' : filter_target_set(RACIST_CON),
        'outsider_words' : filter_target_set(OUTSIDER_WORDS, word_vectors), 
        'jewish_occupations' : filter_target_set(JEWISH_OCCUPATIONS, word_vectors),
        'jewish_nouns' : filter_target_set(JEWISH_STEREOTYPES_NOUNS, word_vectors),
        'jewish_character' : filter_target_set(JEWISH_STEREOTYPES_CHARACTER, word_vectors),
        'jewish_political' : filter_target_set(JEWISH_STEREOTYPES_POLITICAL, word_vectors)
                    }
    if kind == 'BRD':
        attribute_sets['volkstreu'] = filter_target_set(VOLKSTREU_BRD, word_vectors)
        attribute_sets['volksuntreu'] = filter_target_set(VOLKSUNTREU_BRD, word_vectors)
    elif kind == 'RT':            
        attribute_sets['volkstreu'] = filter_target_set(VOLKSTREU_RT, word_vectors)
        attribute_sets['volksuntreu'] = filter_target_set(VOLKSUNTREU_RT, word_vectors)
    else:
        print('parameter ''kind'' must be specified to either RT for Reichstag proceedings or BRD for Bundestag proceedings.')

    return attribute_sets

def convert_attribute_set(label):
    if label in ('sentiment', 'random'):
      return ('pleasant', 'unpleasant')
    elif label == 'sentiment_flipped':
      return ('unpleasant', 'pleasant')
    elif label == 'patriotism':
      return ('volkstreu', 'volksuntreu')
    elif label == 'economic':
      return ('economic_pro', 'economic_con')
    elif label == 'conspiratorial':
      return ('conspiratorial_pro', 'conspiratorial_con')
<<<<<<< HEAD
    elif label == 'racist':
      return ('racist_pro', 'racist_con')
    elif label == 'religious':
      return ('religious_pro', 'religious_con')
=======
  
>>>>>>> 0e9e5c6cdefc8b5688524df068bdeac4b79673fa
      
def create_target_sets(word_vectors, kind): 
    """
    Create all target sets for this study
    :param word_vectors: trained word vectors 
    :param kind: kind of attributes to create - either RT or BRD 
    """
    if kind == 'RT':
        target_sets = {
        'jewish' : filter_target_set(JEWISH_RT, word_vectors),
        
        'christian' : filter_target_set(CHRISTIAN_RT, word_vectors),
        
        'catholic' : filter_target_set(CATHOLIC_RT, word_vectors),
        
        'protestant' : filter_target_set(PROTESTANT_RT, word_vectors)
                    }
    elif kind == 'BRD':
        target_sets = {
        'jewish' : filter_target_set(JEWISH_BRD, word_vectors),
        
        'christian' : filter_target_set(CHRISTIAN_BRD, word_vectors),
        
        'catholic' : filter_target_set(CATHOLIC_BRD, word_vectors),
        
        'protestant' : filter_target_set(PROTESTANT_BRD, word_vectors)
                    }
    else:
        print('parameter ''kind'' must be specified to either RT for Reichstag proceedings or BRD for Bundestag proceedings.')
    # Join them together to form bias words
    return target_sets

# @numba.jit
def inverse(matrix):
  return np.linalg.inv(matrix)

def create_labels(targets_1, targets_2):
    labels = []
    for word in targets_1:
        labels.append(1)
    for word in targets_2:
        labels.append(0)
    labels = np.array(labels)
    return labels


def load_embedding_dict(vocab_path="", vector_path="", word_vectors_path="", glove=False, postspec=False):
  """
  >>> _load_embedding_dict()
  :param vocab_path:
  :param vector_path:
  :return: embd_dict
  """
  if word_vectors_path != "":
    embd_dict = utils.load_word_vectors(word_vectors_path)
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

def write_list(path, list):
    f = codecs.open(path,'w',encoding='utf8')
    for l in list:
        f.write(l + "\n")
    f.close()

def load_vocab(path, inverse = False):
  vocab = json.load(open(path,"r"))
  if inverse:
    vocab_inv = {v : k for k, v in vocab.items()}
    return vocab, vocab_inv
  else:
    return vocab

def load_vectors(path, normalize = False):
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



