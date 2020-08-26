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
SPECIFICATIONS_FOLDER = DATA_FOLDER / 'specifications'

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

def load_specifications(attribute_set):
    with open(SPECIFICATIONS_FOLDER / f'{attribute_set}.txt', 'r') as f:
        specifications = f.read().lower().split('\n')
    return specifications

def create_attribute_sets(input_repr, kind):
    """
    Create all attribute sets for a specific time period

    :param input_repr: input representation of the text 
    :param kind: version of attributes to create - either for RT or BRD 
    """

    domains = ['sentiment_pro', 'sentiment_con', 'economic_pro', 'economic_con', 'conspiratorial_pro', 'conspiratorial_con', 'religious_pro','religious_con',
     'racist_pro', 'racist_con', 'ethic_pro', 'ethic_con']
    attribute_sets = {d: filter_terms(load_specifications(d), input_repr) for d in domains}

    if kind == 'BRD':
        attribute_sets['patriotic_pro'] = filter_terms(load_specifications('patriotic_pro_brd'), input_repr)
        attribute_sets['patriotic_con'] = filter_terms(load_specifications('patriotic_con_brd'), input_repr)
    elif kind == 'RT':            
        attribute_sets['patriotic_pro'] = filter_terms(load_specifications('patriotic_pro_rt'), input_repr)
        attribute_sets['patriotic_con'] = filter_terms(load_specifications('patriotic_con_rt'), input_repr)
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

      
def create_target_sets(input_repr, kind): 
    """
    Create all target sets for this study

    :param input_repr: trained word vectors 
    :param kind: kind of attributes to create - either RT or BRD 
    """

    targets = ['jewish', 'christian', 'protestant', 'catholic']
    if kind == 'RT':
        target_sets =  {t: filter_terms(load_specifications(f'{t}_rt'), input_repr) for t in targets}
    elif kind == 'BRD':
        target_sets =  {t: filter_terms(load_specifications(f'{t}_brd'), input_repr) for t in targets}
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



