# -*- coding: utf-8 -*-

"""
Let's examine the closest neighbors for a word over time
"""
import sys
import os
import helpers
from gensim.models import KeyedVectors
import argparse

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
print(ROOT_DIR)
tpath = os.path.abspath(os.path.join(ROOT_DIR, "../"))
sys.path.append(tpath)
print(os.getcwd())
print(tpath)
# os.chdir(tpath)
from SequentialEmbeddings import SequentialEmbedding

def main():
    parser = argparse.ArgumentParser(description="Plot semantic shift of words")
    parser.add_argument('-w','--words', nargs='+', help='List of words to plot', required=True)
    parser.add_argument("-n", "--neighbors", type=int, default=15, help="Number of neighbors to plot", required=True)
    parser.add_argument("--protocol_type", type=str, help="Whether to run test for Reichstagsprotokolle (RT) or Bundestagsprotokolle (BRD)", required=True)
    parser.add_argument("--model_folder", type=str, help="Folder where word2vec models are located")
    
    args = parser.parse_args()
    words = args.words
    n = args.neighbors

    if args.protocol_type == 'RT':
      # embeds = {1 : KeyedVectors.load('model/kaiserreich_1'),
      #           2 : KeyedVectors.load('model/kaiserreich_2'),
      #           3 : KeyedVectors.load('model/weimar')
      #           }

      # embeddings = SequentialEmbeddings(embeds)
      embeddings = SequentialEmbedding.load(os.path.join(args.model_folder, 'rt_slice'),5)


    if args.protocol_type == 'BRD':
      embeddings = SequentialEmbedding.load(os.path.join(args.model_folder, 'slice'),4)

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

          helpers.savefig(f"{args.protocol_type}/{word1}_annotated_{n}")
          for year, sim in time_sims.items():
              print(year, sim)
        except KeyError:
          print(f'{word1} is not in the embedding space.')


if __name__ == '__main__':
  main()