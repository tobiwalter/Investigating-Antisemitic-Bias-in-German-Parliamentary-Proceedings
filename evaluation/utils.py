 # -*- coding: utf-8 -*_

from nltk.corpus import wordnet as wn
from sys import stdin
import codecs
import numpy as np
import pickle
import os

# word sets

JEWISH_RT = 'rabbi, synagoge, koscher, sabbat, orthodox, judentum, jude, juedisch, mose, talmud, israel, abraham, zionistisch'.split(', ')

JEWISH_BRD = 'synagoge, koscher, orthodox, judentum, jude, juedisch, israel, israels, israeli, rabbiner, zentralrat'.split(', ')

CHRISTIAN_RT = 'taufe, katholizismus, christentum, evangelisch, evangelium, jesus, christ, christlich, katholisch, kirche, pfarrer, ostern, bibel'.split(', ')

CHRISTIAN_BRD = 'taufe, christentum, evangelisch, evangelium, jesus, christ, christlich, katholisch, kirche, pfarrer, abendland'.split(', ')

PROTESTANT_BRD = "protestant, protestantisch, evangelisch, evangelium, landeskirche, kirchentag, ekd, landesbischof, lutherisch, diakonie".split(', ')

PROTESTANT_RT = 'protestant, protestantisch, protestantismus, evangelisch, evangelium, landeskirche, oberkirchenrat, lutherisch, evangelisch-lutherisch, reformiert'.split(', ')

CATHOLIC_BRD = "katholisch, katholik, papst, roemisch-katholisch, enzyklika, paepstliche, bischofskonferenz, dioezese, franziskus, kurie".split(', ')

CATHOLIC_RT = 'katholizismus, katholisch, katholik, papst, roemisch-katholisch, jesuiten, jesuitenorden, ultramontanismus, ultramontanen, zentrumspartei'.split(', ')

# sets patriotic/non-patriotic words 

VOLKSTREU_RT = 'patriotisch, vaterlandsliebe, volksbewusstsein, volksgeist, germanische, deutschnational, nationalbewusstsein, \
vaterlaendisch, reichstreu, nationalgesinnt, nationalstolz, koenigstreu'.split(', ')

VOLKSUNTREU_RT = 'nichtdeutsch, fremdlaendisch, fremd, undeutsch, vaterlandslos, reichsfeind, landesverraeter, reichsfeindlich, \
unpatriotisch, antideutsch, deutschfeindlich, umstuerzler'.split(', ')   

VOLKSTREU_BRD = 'patriotisch, vaterlandsliebe, germanische, nationalbewusstsein, vaterlaendisch, nationalgefuehl, volkstum, patriotismus, patriot'.split(', ')

VOLKSUNTREU_BRD = 'nichtdeutsch, vaterlandslos, landesverraeter, antideutsch, heimatlos, separatistische, staatsfeindliche, fremd, staatenlos'.split(', ')   

# IO

def load_lines(filepath):
  return [l.strip() for l in list(codecs.open(filepath, "r", encoding = 'utf8', errors = 'replace').readlines())]

def write_list(path, list):
	f = codecs.open(path,'w',encoding='utf8')
	for l in list:
		f.write(l + "\n")
	f.close()

def load_vocab(path, inverse = False):
  vocab = pickle.load(open(path,"rb"))
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

# WN

def add_lemmas(syn, word, lista):
  for l in syn.lemmas():
    if l.name() != word and not "_" in l.name(): 
      lista.append(l.name())  

def fetch(word):
  synsets = wn.synsets(word)
  syns_list = [] 
  hypers_list = []
  cohyp_list = []
  hypo_list = [] 
  if synsets:
    # synonyms
    syn = synsets[0]
    add_lemmas(syn, word, syns_list) 
    
    # hypernyms
    hypers = syn.hypernyms()
    for h in hypers:
      add_lemmas(h, word, hypers_list)

      # cohyponyms
      cohyps = h.hyponyms()
      for ch in cohyps:
        if ch != syn:
          add_lemmas(ch, word, cohyp_list)
  
    # hyponyms
    hypos = syn.hyponyms()
    
    for h in hypos:
      add_lemmas(h, word, hypo_list)
  
  syns_list = list(set(syns_list)) 
  hypers_list = list(set(hypers_list))
  cohyp_list = list(set(cohyp_list))
  hypo_list = list(set(hypo_list))
  return syns_list, hypers_list, cohyp_list, hypo_list

def fetch_list(word_list):    
  results = {}
  for word in word_list:
    synl, hyperl, cohypl, hypol = fetch(word)
    print(word)
    print("Syns: " + " ".join(synl))
    print("Hypers: " + " ".join(hyperl))
    print("Cohyps: " + " ".join(cohypl))
    print("Hypos: " + " ".join(hypol))
    results[word] = (synl, hyperl, hypol, cohypl)
  return results

def flatten_all_augments(results):
  flat_list = []
  for w in results:
    flat_list.extend(results[w][0])
    flat_list.extend(results[w][1])  
    flat_list.extend(results[w][2])
    flat_list.extend(results[w][3])
  flat_list = list(set(flat_list))
  flat_list = [x for x in flat_list if x not in results]
  return flat_list
      
# vector spaces

def similarity(w1, w2, vocab, vectors, normalized = False):
  if w1 and w2 in vocab:
    cs = np.dot(vectors[vocab[w1]], vectors[vocab[w2]])
    if not normalized:
      cs = cs / (np.linalg.norm(vectors[vocab[w1]]) * np.linalg.norm(vectors[vocab[w2]])) 
  else:
    return None

def diffs_search(vocab_inv, vecs, vecs_norm, seed, limit = 2000, sim_thold = 0.3):
  pairs = []
  if not limit:
    limit = len(vecs)

  for i in range(limit):
    print(i)
    # assuming vectors are L2-normalized (cos = dot)
    vec_src = vecs_norm[i]
    # finding indices where the similarity with the query vector is above 
    cosines = np.dot(vec_src, np.transpose(vecs_norm)) # np.sqrt(np.linalg.norm(vec_src - vecs, axis = 1)) #
    inds = np.where(cosines > sim_thold)[0]
    if len(inds) == 1:
      continue
    inds = np.delete(inds, np.where(inds == i)[0])

    diffs_i = vecs[i] - np.array([vecs[ind] for ind in inds])
    sims_diffs = np.linalg.norm(diffs_i - seed, axis = 1)
    ind_min = np.argmin(sims_diffs)
    dist_min = sims_diffs[ind_min]
    
    real_ind_min = inds[ind_min]
    pair = (vocab_inv[i], vocab_inv[real_ind_min], dist_min)
    pairs.append(pair)

  scores = np.array([x[2] for x in pairs])
  order = np.argsort(scores)
  sorted_pairs = [pairs[p] for p in order]
  return sorted_pairs
    
def projection_bolukbasi(seed_words, vocab, vocab_inv, vecs_norm):
  seed_dif = vecs_norm[vocab[seed_words[0]]] - vecs_norm[vocab[seed_words[1]]]
  scores = np.dot(seed_dif, np.transpose(vecs_norm))
  order = np.argsort(scores)
  sorted_pairs = [(vocab_inv[i], scores[i]) for i in order]
  return sorted_pairs


