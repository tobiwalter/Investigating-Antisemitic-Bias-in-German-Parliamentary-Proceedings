# -*- coding: utf-8 -*-''
from gensim.models.word2vec import Word2Vec, PathLineSentences
from gensim.models import KeyedVectors
from gensim import utils
from pathlib import Path
import numpy as np
import pickle
import os
import itertools
import collections
from ReichstagEmbeddings import ReichstagEmbeddings

# Define class for sequential embedding
class SequentialEmbedding:
    def __init__(self, slice_embeds, **kwargs):
        self.embeds = slice_embeds
 
    @classmethod
#     def load(cls, path, slices, **kwargs):
#         embeds = collections.OrderedDict()
#         for s in range(1,slices+1):
#             embeds[s] =  KeyedVectors.load(path + "slice_%s.model" % str(s), **kwargs).wv
#         return SequentialEmbedding(embeds)
    def load(cls, path, slices, **kwargs):
        embeds = collections.OrderedDict()
        for s in range(1,slices+1):
            embeds[s] = ReichstagEmbeddings.load(f'{path}_{s}')
             # KeyedVectors.load(path + "slice_%s.model" % str(s), **kwargs)
        return SequentialEmbedding(embeds)

    def get_embed(self, slice):
        return self.embeds[str(slice)]
    
    def get_time_sims(self, word1, word2):
       time_sims = collections.OrderedDict()
       for slice, embed in self.embeds.items():
           time_sims[slice] = embed.emb.similarity(word1, word2)
       return time_sims

    def get_nearest_neighbors(self, word, n=3):
        neighbour_set = set([])
        for embed in self.embeds.values():
            closest = embed.emb.most_similar(word,topn=n)
            for neighbour,score in closest:
                neighbour_set.add(neighbour)
        return neighbour_set
    
    def get_seq_closest(self, word, start_slice, num_slices=5, n=10):
        closest = collections.defaultdict(float)
        for slice in range(start_slice, start_slice + num_slices):
            embed = self.embeds[slice]
            slice_closest = embed.emb.most_similar(word,topn=n*10)
            for neigh, score in slice_closest:
                closest[neigh] += score
        return sorted(closest, key = lambda word : closest[word], reverse=True)[0:n]
    
    def get_intra_word_similarites(self, word, slices=[1,2,3,4,5]):
        """
        How similar are vectors of the same word trained in different slices?
        """
        print(f'Similarity vals for word {word} ...')
        print('\n')
        for s1,s2 in itertools.combinations(slices,2):
            print(f'between slice {s1} and {s2}:')
            print(np.dot(twec.embeds[s1][word], twec.embeds[s2][word]))
            print('\n')


