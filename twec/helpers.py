# -*- coding: utf-8 -*-
import sys
import os
import time
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.manifold import TSNE

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

tpath = os.path.abspath(os.path.join(ROOT_DIR, "../"))
sys.path.append(tpath)
os.chdir(tpath)
from SequentialEmbeddings import SequentialEmbedding

CMAP_MIN=0
def get_cmap(n, name='Set1'):
    return plt.cm.get_cmap(name, n+CMAP_MIN)

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

        for word, sim in embed.wv.most_similar(word1, topn=topn):
          ww = f"{word}|{period}"
          nearest.append((sim, ww))
          if sim > 0.3:
                time_sims[period].append((sim, ww))
                lookups[ww] = embed[word]
                sims[ww] = sim

    print(f"GET TIME SIMS FOR {word1} TOOK {time.time() - start}")
    return time_sims, lookups, nearests, sims


def clear_figure():
    plt.figure(figsize=(20,20))
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

numbers = {'1': 'one',
           '2': 'two',
           '3': 'three',
           '4': 'four',
           '5': 'five'
           }

def assing_period(word):
    number = word.split("|")[1].strip() 
    return numbers.get(number)

def plot_words(word1, words, fitted, cmap, sims, n):
    # TODO: remove this and just set the plot axes directly
    label = [assing_period(word) for word in words]
    colors = [cmap(i - 1 + CMAP_MIN) for i in range(1,n)]
    print(label)
    print(colors)
    sns.scatterplot(fitted[:,0], fitted[:,1], alpha=0, hue=label, palette= colors)
    # plt.scatter(fitted[:,0], fitted[:,1], alpha=0, hue=label)
    plt.suptitle(f"{word1}", fontsize=30, y=0.1)
    plt.axis('off')


    annotations = []
    isArray = type(word1) == list
    for i in range(len(words)):
        pt = fitted[i]

        ww,period = [w.strip() for w in words[i].split("|")]
        color = cmap((int(period)) - 1 + CMAP_MIN)
        word = ww
        sizing = sims[words[i]] * 30

        # word1 is the word we are plotting against
        if ww == word1 or (isArray and ww in word1):
            annotations.append((ww, period, pt))
            word = period
            color = 'black'
            sizing = 15

        plt.text(pt[0], pt[1], word, color=color, size=int(sizing))

    return annotations


def plot_annotations(annotations):
    # draw the movement between the word through the decades as a series of
    # annotations on the graph
    annotations.sort(key=lambda w: w[1], reverse=True)
    prev = annotations[0][-1]
    for ww, period, ann in annotations[1:]:
        plt.annotate('', xy=prev, xytext=ann,
            arrowprops=dict(facecolor='blue', shrink=0.1, alpha=0.3,width=2, headwidth=15))
        print(prev, ann)
        prev = ann

def savefig(name):

    directory = os.path.join(ROOT_DIR, "output")
    if not os.path.exists(directory):
        os.makedirs(directory)

    fname = os.path.join(directory, name)

    plt.savefig(fname, bbox_inches=0)
