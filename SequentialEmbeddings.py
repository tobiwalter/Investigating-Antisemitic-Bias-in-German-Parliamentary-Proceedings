# -*- coding: utf-8 -*-''
import numpy as np
import itertools
import collections
from gensim.models import KeyedVectors

# Define class for sequential embedding
class SequentialEmbedding:
    def __init__(self, embedding_spaces, **kwargs):
        self.embeds = embedding_spaces
 
    @classmethod
    def load(cls, path, num_slices, **kwargs):
        embeds = collections.OrderedDict()
        for i in range(1,num_slices+1): 
            embeds[i] = KeyedVectors.load(f'{path}_{i}.model')
        return SequentialEmbedding(embeds)

    def get_embed(self, slice):
        return self.embeds[str(slice)]
    
    def get_time_sims(self, word1, word2):
       time_sims = collections.OrderedDict()
       for slice, embed in self.embeds.items():
           time_sims[slice] = embed.wv.similarity(word1, word2)
       return time_sims

    def get_nearest_neighbors(self, word, n=3):
        neighbour_set = set([])
        for embed in self.embeds.values():
            closest = embed.wv.most_similar(word,topn=n)
            for neighbour,score in closest:
                neighbour_set.add(neighbour)
        return neighbour_set
    
    def get_seq_closest(self, word, start_slice, num_slices=5, n=10):
        closest = collections.defaultdict(float)
        for slice in range(start_slice, start_slice + num_slices):
            embed = self.embeds[slice]
            slice_closest = embed.wv.most_similar(word,topn=n*10)
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
            print(np.dot(self.embeds[s1].wv[word], self.embeds[s2].wv[word]))
            print('\n')


