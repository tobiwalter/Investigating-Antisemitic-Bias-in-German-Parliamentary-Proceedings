# -*- coding: utf-8 -*-
import sys
import os
import time
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import collections
import heapq
from sklearn.manifold import TSNE

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

tpath = os.path.abspath(os.path.join(ROOT_DIR, "../"))
sys.path.append(tpath)
os.chdir(tpath)

from plot_utils import set_size
from SequentialEmbeddings import SequentialEmbedding

CMAP_MIN=0
WIDTH = 360
FIG_DIM = set_size(WIDTH)

def get_cmap(n, name='Set1'):
    return plt.cm.get_cmap(name, n+CMAP_MIN)

def get_index(model):
    words = sorted([w for w in model.wv.vocab], key=lambda w: model.wv.vocab.get(w).index)
    index = {w: i for i, w in enumerate(words)}
    return index

def closest(model, w, topn=10):
    """
    Assumes the vectors have been normalized.
    """
    scores = model.wv.vectors.dot(model[w])
    index = get_index(model)
    return heapq.nlargest(topn, zip(scores, index))

def get_time_sims(seq_embedding, word1, topn=15):
    start = time.time()
    time_sims = collections.OrderedDict()
    lookups = {}
    nearests = {}
    sims = {}
    for period, embed in seq_embedding.embeds.items():
        nearest = []
        nearests[f"{word1}|{period}"]= nearest
        time_sims[period] = []

        # ww = f"{word1}|{period}"
        # sim = 1.00
        # nearest.append((sim, ww))
        # time_sims[period].append((sim, ww))
        # lookups[ww] = embed[word1]
        # sims[ww] = sim

        # for word, sim in embed.wv.most_similar(word1, topn=topn):
        for sim, word in embed.closest(word1, topn=topn):
          ww = f"{word}|{period}"
          nearest.append((sim, ww))
          if sim > 0.3:
                time_sims[period].append((sim, ww))
                lookups[ww] = embed[word]
                sims[ww] = sim

    print(f"GET TIME SIMS FOR {word1} TOOK {time.time() - start}")
    return time_sims, lookups, nearests, sims


def clear_figure():
    plt.clf()

def fit_tsne(values):
    if not values:
        return
    start = time.time()
    mat = np.array(values)
    model = TSNE(n_components=2, random_state=0, learning_rate=150, init='pca')
    fitted = model.fit_transform(mat)
    print(f"FIT TSNE TOOK {time.time() - start}")

    return fitted

def assing_period(word, protocol_type):
    if protocol_type == 'BRD':
        numbers = {'1': 'CDU I',
           '2': 'SPD I',
           '3': 'CDU II',
           '4': 'SPD II',
           '5': 'CDU III'
           }

    elif protocol_type == 'RT':
            numbers = {'1': 'KS I',
           '2': 'KS II',
           '3': 'Weimar'
           }
    number = word.split("|")[1].strip() 
    return numbers.get(number)

def plot_words(word1, words, fitted, cmap, sims, n, protocol_type):
    # TODO: remove this and just set the plot axes directly
    label = [assing_period(word, protocol_type) for word in words]
    colors = [cmap(i - 1 + CMAP_MIN) for i in range(1,n)]
    fig, ax = plt.subplots(figsize=FIG_DIM)
    sns.scatterplot(fitted[:,0], fitted[:,1], alpha=0, hue=label, palette=colors, legend='brief')

    plt.suptitle(f"{word1}", fontsize=8, y=0.05)
    fig.set_size_inches(FIG_DIM[0], (FIG_DIM[0]/1.6))
    plt.axis('off')
    plt.tight_layout()
    ax.legend(loc='best', fontsize=3,framealpha=0.75, frameon=True, markerscale=0.2)


    annotations = []
    isArray = type(word1) == list
    for i in range(len(words)):
        pt = fitted[i]

        ww,period = [w.strip() for w in words[i].split("|")]
        color = cmap((int(period)) - 1 + CMAP_MIN)
        word = ww
        sizing = sims[words[i]] * 10
        # word1 is the word we are plotting against
        if ww == word1 or (isArray and ww in word1):
            annotations.append((ww, period, pt))
            word = period
            color = 'black'
            sizing = 4

        plt.text(pt[0], pt[1], word, color=color, size=int(sizing))

    return annotations


def plot_annotations(annotations):
    # draw the movement between the word through the decades as a series of
    # annotations on the graph
    annotations.sort(key=lambda w: w[1], reverse=True)
    prev = annotations[0][-1]
    for ww, period, ann in annotations[1:]:
        plt.annotate('', xy=prev, xytext=ann,
            arrowprops=dict(facecolor='blue', shrink=0.1, alpha=0.15,width=1, headwidth=10))
        print(prev, ann)
        prev = ann

def savefig(name, protocol_type, n):

    directory = os.path.join(ROOT_DIR, f"output/{protocol_type}_historic")
    if not os.path.exists(directory):
        os.makedirs(directory)

    fname = os.path.join(directory, name)

    plt.savefig(f'{fname}_{n}_control.pdf', format='pdf', bbox_inches='tight')
