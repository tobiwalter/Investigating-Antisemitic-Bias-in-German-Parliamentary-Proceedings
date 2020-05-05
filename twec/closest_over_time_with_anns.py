import helpers
import sys

"""
Let's examine the closest neighbors for a word over time
"""
import numpy as np
import matplotlib.pyplot as plt
from SequentialEmbeddings import SequentialEmbedding

# We accept a list of words from command line
# to generate graphs for.


WORDS = helpers.get_words()
N = int(helpers.get_n())

if __name__ == "__main__":
    embeddings = SequentialEmbedding.load('slice',4)
    for word1 in WORDS:
        helpers.clear_figure()
        time_sims, lookups, nearests, sims = helpers.get_time_sims(embeddings, word1, topn=N)
     
        words = list(lookups.keys())
        values = [ lookups[word] for word in words]
        fitted = helpers.fit_tsne(values)
        if not len(fitted):
            print(f"Couldn't model word {word1}")
            continue

        # draw the words onto the graph
        cmap = helpers.get_cmap(len(time_sims))
        annotations = helpers.plot_words(word1, words, fitted, cmap, sims,5)

        if annotations:
            helpers.plot_annotations(annotations)

        helpers.savefig(f"{word1}_annotated_{N}")
        for year, sim in time_sims.items():
            print(year, sim)
