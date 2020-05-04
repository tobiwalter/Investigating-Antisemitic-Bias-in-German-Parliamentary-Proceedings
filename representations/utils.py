import re
import json
import pickle
from pathlib import Path
import os

DATA_FOLDER = Path('./data')
MODELS_FOLDER = Path('./obj')
VOCAB_FOLDER = DATA_FOLDER / 'vocab'

# sets of pleasant/unpleasant words

PLEASANT = 'streicheln, Freiheit, Gesundheit, Liebe, Frieden, Freude, Freund, Himmel, loyal, Vergnügen, Diamant, sanft, ehrlich, \
glücklich, Regenbogen, Diplom, Geschenk, Ehre, Wunder, Sonnenaufgang, Familie, Lachen, Paradies, Ferien'.lower().split(', ') 

UNPLEASANT = 'Mißbrauch, Absturz, Schmutz, Mord, Krankheit, Tod, Trauer, vergiften, stinken, Angriff, Katastrophe, Haß, \
verschmutzen, Tragödie, Scheidung, Gefängnis, Armut, häßlich, Krebs, töten, faul, erbrechen, Qual'.lower().split(', ') 

# sets patriotic/non-patriotic words 

VOLKSTREU = 'patriotisch, vaterlandsliebe, volksbewußtsein, volksgeist, germanische, deutschnational, nationalbewußtsein, \
vaterländisch, reichstreu, nationalgesinnt, nationalstolz, deutschnational, königstreu'.split(', ')

VOLKSUNTREU = 'nichtdeutsch, fremdländisch, fremd, undeutsch, vaterlandslos, reichsfeind, landesverräter, reichsfeindlich, \
unpatriotisch, antideutsch, deutschfeindlich, umstürzler'.split(', ')   

OUTSIDER_WORDS = 'unaufrichtig, komisch, boshaft, unberechenbar, primitiv, beängstigend, hinterlistig, energisch, trügerisch, \
neidisch, gierig, abscheulich, verachtenswert, brutal, ungeheuer, berechnend, grausam, gemein, intolerant, aggressiv'\
.lower().split(', ') 

JEWISH_STEREOTYPES_NOUNS = 'Gier, Wucherer, Drückeberger, Kriegsgewinnler, Macht, Einfluß, Kriegstreiber, Lügner, \
Weltherrschaft, Kommunismus, Kapitalismus, Liberalismus, Außenseiter'.lower().split(', ')

JEWISH_STEREOTYPES_CHARACTER = 'egoistisch, fremd, dekadent, haßerfüllt, habgierig, geldgierig, penetrant, hinterlistig, \
intellektuell, pervers, hinterhältig, betrügerisch, gebeugt, bucklig'.split(', ')

JEWISH_STEREOTYPES_POLITICAL = 'liberalistisch, modern, materialistisch, liberal, undeutsch, unpatriotisch, säkular, \
sozialistisch, links, bolschewistisch'.split(', ')

JEWISH_OCCUPATIONS = 'Pfandleiher, Geldleiher, Kaufmann, Händler, Bankier, Finanzier, Steuereintreiber, Zöllner, \
Trödelhändler'.lower().split(', ') 


class CreateSlice:
    
    def __init__(self, dirname, profiling=False):
        self.dirname = str(DATA_FOLDER / dirname)
        self.profiling = profiling

    def __iter__(self):
        for fn in os.listdir(self.dirname):
            text = open(os.path.join(self.dirname, fn), encoding='utf-8').readlines()
            if self.profiling == True:
                yield text
            else: 
                for sentence in text:
                    yield sentence.split()

class CreateCorpus:
    """
    Read in pre-process files before feeding them to word2vec model 
    """
    def __init__(self,top_dir):
        self.top_dir = top_dir
        
    """Iterate over all documents, yielding a document (=list of utf8 tokens) at a time."""
    def __iter__(self):
        for root, dirs, files in os.walk(self.top_dir):
            for file in filter(lambda file: file.endswith('.txt'), files):
                text = open(os.path.join(root, file), encoding='utf-8').readlines()
                for line in text:
                    yield line

def save_corpus(corpus, filepath):
    with open(str(MODELS_FOLDER / filepath) + '.pkl', 'wb') as f:
        pickle.dump(corpus, f, pickle.HIGHEST_PROTOCOL)

def save_vocab(model, filepath):
    words = sorted([w for w in model.wv.vocab], key=lambda w: model.wv.vocab.get(w).index)
    index = {w: i for i, w in enumerate(words)}
    json_repr = json.dumps(index)
    with open(str(VOCAB_FOLDER / filepath) + '.json',"w", encoding='utf-8') as f:    
        f.write(json_repr)
        f.close()

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

def create_attribute_sets(embeddings):
    attribute_sets = {
        'pleasant' : filter_target_set(PLEASANT, embeddings),
        'unpleasant' : filter_target_set(UNPLEASANT, embeddings),
        'outsider_words' : filter_target_set(OUTSIDER_WORDS, embeddings), 
        'jewish_occupations' : filter_target_set(JEWISH_OCCUPATIONS, embeddings),
        'jewish_nouns' : filter_target_set(JEWISH_STEREOTYPES_NOUNS, embeddings),
        'jewish_character' : filter_target_set(JEWISH_STEREOTYPES_CHARACTER, embeddings),
        'jewish_political' : filter_target_set(JEWISH_STEREOTYPES_POLITICAL, embeddings),
        'volkstreu' : filter_target_set(VOLKSTREU, embeddings),
        'volksuntreu' : filter_target_set(VOLKSUNTREU, embeddings)
         }
    
    return attribute_sets


def create_target_sets(word_vectors): 
    target_sets = {
    'jewish' : filter_target_set('rabbi, synagoge, koscher, sabbat, orthodox, judentum, jude, jüdisch, mose, talmud, israel, abraham, zionistisch'.split(', '), word_vectors),
    
    'christian' : filter_target_set('taufe, katholizismus, christentum, evangelisch, evangelium, jesus, christ, christlich, katholisch, kirche, pfarrer, ostern, bibel'.split(', '), word_vectors),
    
    'catholic' : filter_target_set('katholizismus, katholisch, katholik, papst, römisch-katholisch, jesuiten, jesuitenorden, ultramontanismus, ultramontanen, zentrumspartei'.split(', '), word_vectors),
    
    'protestant' : filter_target_set('protestant, protestantisch, protestantismus, evangelisch, evangelium, landeskirche, oberkirchenrat, lutherisch, evangelisch-lutherisch, reformiert'.split(', '), word_vectors)
                }
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
