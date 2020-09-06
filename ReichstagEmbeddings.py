# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
from gensim.models import KeyedVectors
import json
import heapq
import os

ROOT_DIR = os.path.dirname(os.path.abspath('__file__'))
VOCAB_FOLDER = Path(ROOT_DIR) / 'data/vocab'

class Embeddings:
  """Class to load embedding model for semantic shift plotting

  Credits: This implementation is based on https://github.com/williamleif/histwords/blob/master/representations/embedding.py
  """
  def __init__(self, model, index, normalize=True):
    self.emb = model.wv
    self.index = index
    self.dim = self.emb.vectors.shape[1]
    if normalize:
      self.normalize()

  def __getitem__(self, word):
      if word not in self.index:
          raise KeyError
      else:
          return self.emb.vectors[self.index[word]] 

  def __contains__(self,word):
      return word in self.index

  @classmethod
  def load(cls, path):
    # print(str(cls.model_folder / path) + '.model')
    vocab_path = str(cls.vocab_folder /os.path.splitext(os.path.basename(path))[0]) + '.json'
    try:
      model =  KeyedVectors.load(path)
    except FileNotFoundError:
      print('Model not found!')
    index = json.load(open(vocab_path, "r"))
    return cls(model, index)

  def reindex(self):
    words = sorted([w for w in self.emb.vocab], key=lambda w: self.emb.vocab.get(w).index)
    self.index = {w: i for i, w in enumerate(words)}

  def diff(self, word1, word2):
      v = self.emb.vectors[self.index[word1]] - self.emb.vectors[self.index[word2]]
      return v/np.linalg.norm(v)

  def normalize(self):
      self.emb.vectors /= np.linalg.norm(self.emb.vectors, axis=1)[:, np.newaxis]

  def closest(self, w, topn=10):
    """
    Assumes the vectors have been normalized.
    """
    scores = self.emb.vectors.dot(self.emb[w])
    return heapq.nlargest(topn, zip(scores, self.index))

