# -*- coding: utf-8 -*-
import helpers
import sys

"""
Let's examine the closest neighbors for a word over time
"""
import numpy as np
import matplotlib.pyplot as plt
from SequentialEmbeddings import SequentialEmbedding
import argparse

# We accept a list of words from command line
# to generate graphs for.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot semantic shift of words")
    parser.add_argument('-w','--words', nargs='+', help='List of words to plot', required=True)
    parser.add_argument("-n", "--neighbors", type=int, default=15, help="Number of neighbors to plot", required=True)
    parser.add_argument("-p", "--protocol_type", type=str, default=None, help="Run tests for Reichstagsprotokolle or Bundestagsprotokolle?", required=True) 
    parser.add_argument('--twec', dest='twec', action='store_true')
    args = parser.parse_args()
    words = args.words
    n = args.neighbors

    if args.protocol_type == 'RT':
      if args.twec:
          embeddings = SequentialEmbedding.load('twec/Reichstag/slice',5)
      else:
          embeddings = SequentialEmbedding.load('rt_slice',5)
    if args.protocol_type == 'BT':
      if args.twec:
          embeddings = SequentialEmbedding.load('twec/Bundestag/slice',4)
      else:
          embeddings = SequentialEmbedding.load('slice_h',4)    

    for word1 in words:
        helpers.clear_figure()
        try:
          time_sims, lookups, nearests, sims = helpers.get_time_sims(embeddings, word1, topn=n)
       
          words = list(lookups.keys())
          values = [ lookups[word] for word in words]
          fitted = helpers.fit_tsne(values)
          if not len(fitted):
              print(f"Couldn't model word {word1}")
              continue

          # draw the words onto the graph
          cmap = helpers.get_cmap(len(time_sims))
          annotations = helpers.plot_words(word1, words, fitted, cmap, sims,len(embeddings.embeds)+1)

          if annotations:
              helpers.plot_annotations(annotations)
          
          if args.twec:
              helpers.savefig(f"{args.protocol_type}/twec/{word1}_annotated_{n}")
          else:
              helpers.savefig(f"{args.protocol_type}/{word1}_annotate_{n}")

          for year, sim in time_sims.items():
              print(year, sim)
        except KeyError:
            print(f'{word1}  is not in the embedding space.')

