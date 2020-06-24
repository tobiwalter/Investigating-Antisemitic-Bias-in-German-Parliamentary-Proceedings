import re
import json
import pickle
from pathlib import Path
import os
import codecs

DATA_FOLDER = Path('./data')
MODELS_FOLDER = Path('./obj')
VOCAB_FOLDER = DATA_FOLDER / 'vocab'

# term sets

JEWISH_RT = ["jude", "juedisch", "judentum", "orthodox", "israel", "mosaisch","israelitisch","israelit", "rothschild", "talmud", "synagoge", "abraham", "rabbiner", "zionistisch"]

JEWISH_BRD = 'judentum, jude, juedisch, israel, israels, israeli, synagoge, koscher, orthodox, rabbiner, zentralrat'.split(', ')

CHRISTIAN_RT = ["christ", "christlich", "christentum", "katholizismus", "katholisch", "evangelisch", "evangelium", "auferstehung", "kirche" , "jesus", "taufe", "pfarrer", "bibel", "ostern"]

CHRISTIAN_BRD = 'christ, christlich, christentum, evangelisch, evangelium, jesus, katholisch, kirche, pfarrer, taufe, abendland'.split(', ')

PROTESTANT_BRD = "protestant, protestantisch, evangelisch, evangelium, landeskirche, kirchentag, ekd, landesbischof, lutherisch, diakonie".split(', ')

PROTESTANT_RT = ["protestant", "protestantisch", "protestantismus", "evangelisch", "evangelium", "landeskirche", "lutherisch", "evangelisch-lutherisch", "oberkirchenrat", "reformiert"]

CATHOLIC_BRD = "katholisch, katholik, papst, roemisch-katholisch, enzyklika, paepstlich, bischofskonferenz, dioezese, franziskus, kurie".split(', ')

CATHOLIC_RT = ["katholizismus", "katholisch", "katholik", "papst", "jesuiten", "ultramontanismus", "ultramontanen", "jesuitenorden", "roemisch-katholisch", "zentrumspartei"]

PLEASANT = 'streicheln, Freiheit, Gesundheit, Liebe, Frieden, Freude, Freund, Himmel, loyal, Vergnuegen, Diamant, sanft, ehrlich, gluecklich, Regenbogen, Diplom, Geschenk, Ehre, Wunder, Sonnenaufgang, Familie, Lachen, Paradies, Ferien'.lower().split(', ') 

UNPLEASANT = 'Missbrauch, Absturz, Schmutz, Mord, Krankheit, Tod, Trauer, vergiften, stinken, Angriff, Katastrophe, Hass, verschmutzen, Tragoedie, Scheidung, Gefaengnis, Armut, haesslich, Krebs, toeten, faul, erbrechen, Qual'.lower().split(', ') 

# sets patriotic/non-patriotic words 

VOLKSTREU_RT = 'patriotisch, vaterlandsliebe, volksbewusstsein, volksgeist, germanisch, deutschnational, nationalbewusstsein, vaterlaendisch, reichstreu, nationalgesinnt, nationalstolz, koenigstreu'.split(', ')

VOLKSUNTREU_RT = 'nichtdeutsch, fremdlaendisch, fremd, undeutsch, vaterlandslos, reichsfeind, landesverraeter, reichsfeindlich, unpatriotisch, antideutsch, deutschfeindlich, umstuerzler'.split(', ')   

VOLKSTREU_BRD = 'patriotisch, vaterlandsliebe, germanisch, nationalbewusstsein, vaterlaendisch, nationalgefuehl, volkstum, patriotismus, patriot'.split(', ')

VOLKSUNTREU_BRD = 'nichtdeutsch, vaterlandslos, landesverraeter, antideutsch, heimatlos, separatistisch, staatsfeindlich, fremd, staatenlos'.split(', ')   

ECONOMIC_PRO = 'geben, großzuegigkeit, großzuegig, selbstlos,  genuegsam, großmut, uneigennuetzig, sparsam, bourgeoisie, großmut, bescheiden'.split(', ')

ECONOMIC_CON = 'nehmen, gier, gierig, egoistisch, habgierig, eigennuetzig, verschwenderisch, proletariat, habsucht, wucher'.split(', ')

CONSPIRATORIAL_PRO = 'treu, moralisch, ehrlich, loyal, aufrichtig, ehrenwert, offensichtlich, machtlos, ohnmacht'.split(', ')

CONSPIRATORIAL_CON = 'untreu, unmoralisch, unehrlich, verraeterisch, hinterlistig, betruegerisch, geheim, einflussreich, macht'.split(', ')

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
            if self.profiling:
                yield text
            else: 
                for sentence in text:
                    yield sentence.split()

class CreateCorpus:
    """
    Read in pre-process files before feeding them to word2vec model 
    """
    def __init__(self,top_dir, profiling=False):
        self.top_dir = top_dir
        self.profiling = profiling
    """Iterate over all documents, yielding a document (=list of utf8 tokens) at a time."""
    def __iter__(self):
        for root, dirs, files in os.walk(self.top_dir):
            for file in filter(lambda file: file.endswith('.txt'), files):
                text = open(os.path.join(root, file), encoding='utf-8').readlines()
                if self.profiling:
                    yield text
                else:
                    for line in text:
                        yield line

def write_lines(path, list):
    f = codecs.open(path, "w", encoding='utf8')
    for l in list:
        f.write(str(l) + "\n")
    f.close()

def save_corpus(corpus, corpus_path):
    if not os.path.exists(str(DATA_FOLDER/ corpus_path)):
        os.makedirs(str(DATA_FOLDER/ corpus_path))
    for num,doc in enumerate(corpus):
        write_lines(str(DATA_FOLDER/ corpus_path / f'{num+1}_sents.txt'), doc)
    words = sorted([w for w in model.wv.vocab], key=lambda w: model.wv.vocab.get(w).index)
    index = {w: i for i, w in enumerate(words)}
    json_repr = json.dumps(index)
    with open(str(VOCAB_FOLDER / filepath) + '.json',"w", encoding='utf-8') as f: 
        f.write(json_repr)

def load_corpus(filepath):
    with open(str(MODELS_FOLDER / filepath), 'rb') as f:
        corpus = pickle.load(f)
        return corpus

def assign_label(word, attributes):
    if word in jewish_words:
        return 'jewish'
    elif word in christian_words:
        return 'christian'
    else: return attributes

def filter_target_set(target_set, embeddings):
    '''Filter out all target words that did not reach min_count and hence are not in the embeddings'''
    return [word for word in target_set if word in embeddings]

def create_attribute_sets(embeddings, kind):
    attribute_sets = {
        'pleasant' : filter_target_set(PLEASANT, embeddings),
        'unpleasant' : filter_target_set(UNPLEASANT, embeddings),
        'outsider_words' : filter_target_set(OUTSIDER_WORDS, embeddings), 
        'jewish_occupations' : filter_target_set(JEWISH_OCCUPATIONS, embeddings),
        'jewish_nouns' : filter_target_set(JEWISH_STEREOTYPES_NOUNS, embeddings),
        'jewish_character' : filter_target_set(JEWISH_STEREOTYPES_CHARACTER, embeddings),
        'jewish_political' : filter_target_set(JEWISH_STEREOTYPES_POLITICAL, embeddings)
                    }
    if kind == 'BT':
        attribute_sets['volkstreu'] = filter_target_set(VOLKSTREU_BRD, embeddings)
        attribute_sets['volksuntreu'] = filter_target_set(VOLKSUNTREU_BRD, embeddings)
    elif kind == 'RT':            
        attribute_sets['volkstreu'] = filter_target_set(VOLKSTREU_RT, embeddings)
        attribute_sets['volksuntreu'] = filter_target_set(VOLKSUNTREU_RT, embeddings)
    
    return attribute_sets


def create_target_sets(word_vectors, kind): 
    if kind == 'RT':
        target_sets = {
        'jewish' : filter_target_set(JEWISH_RT, word_vectors),
        
        'christian' : filter_target_set(CHRISTIAN_RT, word_vectors),
        
        'catholic' : filter_target_set(CATHOLIC_RT, word_vectors),
        
        'protestant' : filter_target_set(PROTESTANT_RT, word_vectors)
                    }
    elif kind == 'BT':
        target_sets = {
        'jewish' : filter_target_set(JEWISH_BRD, word_vectors),
        
        'christian' : filter_target_set(CHRISTIAN_BRD, word_vectors),
        
        'catholic' : filter_target_set(CATHOLIC_BRD, word_vectors),
        
        'protestant' : filter_target_set(PROTESTANT_BRD, word_vectors)
                    }
    else:
        print('Need to specify kind - either RT for Reichstag proceedings or BRD for Bundestag proceedings.')
    # Join them together to form bias words
    return target_sets

def load_embeddings(embeddings_path):
    """
    >>> load_embeddings("/work/anlausch/glove_twitter/glove.twitter.27B.200d.txt")
    :param path:
    :return:
    """
    embbedding_dict = {}
        #Load Google's pre-trained Word2Vec model.
    if os.name != 'nt':
        model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    else:
        model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
    return model

def load_embedding_dict(vocab_path="", vector_path="", embeddings_path="", glove=False, postspec=False):
  """
  >>> _load_embedding_dict()
  :param vocab_path:
  :param vector_path:
  :return: embd_dict
  """
  if embeddings_path != "":
    embd_dict = utils.load_embeddings(embeddings_path)
    return embd_dict
  else:
    embd_dict = {}
    vocab = json.load(open(vocab_path, 'r'))
    vectors = np.load(vector_path)
    for term, index in vocab.items():
      embd_dict[term] = vectors[index]
    assert len(embd_dict) == len(vocab)
    return embd_dict
