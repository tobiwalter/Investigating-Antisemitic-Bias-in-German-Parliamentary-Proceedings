# -*- coding: utf-8 -*-
import os 
from pathlib import Path
import pickle
from collections import defaultdict
import itertools
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# topic model
from gensim.corpora.dictionary import Dictionary
import spacy
# !python -m spacy download de_core_news_sm

# word embeddings
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
from sklearn.decomposition import PCA

# import internals
import representations.utils
import json
import heapq

# Seed lists
# 1) Targets:
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)

class ReichstagEmbeddings:
  model_folder = Path('twec/model')
  vocab_folder = Path('twec/vocab')
# For now: no method to train word vectors, only to load them --> training is done by executing train_embeddings.py
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
    try:
      model =  KeyedVectors.load_word2vec_format(str(cls.model_folder / path) + '.txt', binary=True)
    except FileNotFoundError:
      model =  KeyedVectors.load(str(cls.model_folder / path) + '.model')

    index = json.load(open(str(cls.vocab_folder / path) + '.json', "r"))
    return cls(model, index)

  def reindex(self):
    words = sorted([w for w in self.emb.vocab], key=lambda w: self.emb.vocab.get(w).index)
    self.index = {w: i for i, w in enumerate(words)}

  def diff(self, word1, word2):
      v = self.emb.vectors[self.index[word1]] - self.emb.vectors[self.index[word2]]
      return v/np.linalg.norm(v)

  def normalize(self):
      self.emb.vectors /= np.linalg.norm(self.emb.vectors, axis=1)[:, np.newaxis]

  def closest(self, w, n=10):
    """
    Assumes the vectors have been normalized.
    """
    scores = self.emb.vectors.dot(self.emb[w])
    return heapq.nlargest(n, zip(scores, self.index))


  def filter_target_set(self,target_set):
    return [word for word in target_set if word in self.wv]

  def create_attribute_sets(self):
    attribute_sets = {
        'pleasant' : self.filter_target_set(PLEASANT, self.wv),
        'unpleasant' : self.filter_target_set(UNPLEASANT, self.wv),
        'outsider_words' : self.filter_target_set(OUTSIDER_WORDS, self.wv), 
        'jewish_stereotypes' : self.filter_target_set(JEWISH_STEREOTYPES, self.wv),
                      }
    return attribute_sets

  def create_target_sets(self):
    target_sets = {
      'jewish' : self.filter_target_set(JEWISH, self.wv),
      'christian' : self.filter_target_set(CHRISTIAN, self.wv),
      'protestant' : self.filter_target_set(PROTESTANT, self.wv),
      'catholic' : self.filter_target_set(CATHOLIC, self.wv)
                  }

  def get_word_indices(self):
      
      words = list(self.wv.vocab)
      word_indices = {word : word.index for word in words}
      return word_indices

  def get_wordlist_similarity(self, bias_word, target_set):
      total_similarity = 0
      for word in target_set:
        total_similarity += self.wv.similarity(bias_word, word)
      return round(total_similarity/len(target_set),4)


  def create_bias_df(self, bias_words, attribute_sets):
    # Create index for data frame consisting of target concepts and attribute words 
    bias_index = pd.MultiIndex.from_product(iterables= [list(attribute_sets.keys()), bias_words], names= ['target set', 'bias word'])

    # Data frame with to be filled with statistics for each bias word in relation to each attribute set
    bias_df = pd.DataFrame(index= bias_index, columns= ['rank', 'similarity', 'label'])

    # Fill data frame 
    for index,row in bias_df.iterrows():
        row['similarity'] = self.get_wordlist_similarity(index[1], attribute_sets[index[0]])
        row['similarity'] = row.astype('float')
        row['label'] = 'jewish' if index[1] in jewish_words else 'christian'
        
    return bias_df


  def create_bias_df2(self, bias_words_1, bias_words_2, attribute_sets):
      df = pd.DataFrame(index = list(attribute_sets.keys()), columns = ['sim_jewish', 'sim_christian'])
      for index,row in df.iterrows():
          test_attributes = [word for word in attribute_sets[index]]
          row['sim_jewish'] = self.wv.n_similarity(test_attributes, bias_words_1)
          row['sim_christian'] = self.wv.n_similarity(test_attributes, bias_words_2)

      return df


def most_similar_words_df(self,bias_words, topn=5):
    columns = ['most_similar_' + str(i) for i in range(1,topn+1)]
    df = pd.DataFrame(index=bias_words, columns= columns)

    for index,row in df.iterrows():
        most_similar_words = self.wv.most_similar(index, topn=topn)
        for i in range(1,topn+1):
            row['most_similar_' + str(i)] = most_similar_words[i-1][0]
            # 
    return df


def plot_avg_similarity(bias_df, bias_word_1, bias_word_2):
    # Create boolean masks -> easier for plotting 
    mask_jewish = bias_df['label'] == 'jewish'
    mask_christian = bias_df['label'] == 'christian'

    # Use slicing in order to only include according rows into relevant data frames 
    idx = pd.IndexSlice
    jewish_df = bias_df.loc[idx[mask_jewish]]
    christian_df = bias_df.loc[idx[mask_christian]]

    jewish_stats_agg = jewish_df.unstack(level=0).agg('mean')
    christian_stats_agg = christian_df.unstack(level=0).agg('mean')
    
    fig, ax = plt.subplots(figsize=(12,8))
    labels = jewish_stats_agg['similarity'].index  
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, jewish_stats_agg['similarity'].values, width)
    ax.bar(x + width/2, christian_stats_agg['similarity'].values, width)
    ax.set_xlabel('target set')
    ax.set_ylabel('similarity')
    ax.set_title('Avg similarity for target concepts')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(['jewish', 'christian'])

    plt.show()


def reduce_dimension_pca(self):
    pca = PCA(n_components=2)
    result = pca.fit_transform(self[self.wv.vocab])
    return result


def plot_word_embeddings(self, bias_words_1, bias_words_2,
  attributes, figsize = (20,15)):
    
    reduced_embeddings = self.reduce_dimension_pca()    
    words_to_plot = bias_words_1 + bias_words_2 + attributes
    label =  [utils.assign_label(word,attributes) for word in words_to_plot]
    selected_indices = [(word, word_indices[word]) for word in words_to_plot]

    fig,ax = plt.subplots(figsize=figsize)
    sns.scatterplot(reduced_embeddings1[[t[1] for t in selected_indices], 0], reduced_embeddings1[[t[1] for t in selected_indices], 1], hue=label)
    plt.title('Word embeddings on the first 2 principal components - selected words zoomed in')
    for tup in selected_indices:
        plt.annotate(tup[0], xy=(reduced_embeddings1[tup[1], 0], reduced_embeddings1[tup[1], 1]))


# def main():        

#   embeddings = ReichtagEmbeddings('')

# if __name__ == '__main__':
#   main()

