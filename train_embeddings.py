# -*- coding: utf-8 -*-
import os 
import logging
import json
from pathlib import Path
import multiprocessing as mp
from gensim.models.word2vec import Word2Vec, PathLineSentences
from gensim.models.fasttext import FastText as FT_gensim
import utils
import argparse
import time

parser = argparse.ArgumentParser(description='Train word embedding models for Reichstag proceedings')
parser.add_argument('--format', type=str, default='gensim',
	help='Save word2vec model in word2vec or gensim format')
parser.add_argument('--proceedings', type=str, help='folder containing pre-processed Reichstag proceedings docs')
parser.add_argument('--model_path', type=str, help='path to store trained model')
parser.add_argument('--vocab_path', type=str, help='path to store model vocab and indices')
parser.add_argument('--model_type', type=str, default='word2vec', help='type of embedding space to train embeddings')
parser.add_argument('-s', '--size', type=int, default=200, help='dimension of word embeddings')
parser.add_argument('-w', '--window', type=int, default=5, help='window size to define context of each word')
parser.add_argument('-m', '--min_count', type=int, default=5, help='minimum frequency of a word to be included in vocab')
parser.add_argument('-t', '--threads', type=int, default=mp.cpu_count(), help='number of worker threads to train word embeddings')
parser.add_argument('-sg', '--sg', type=int, default=0, help='architecture to be used for training: 0 = CBOW; 1 = SG')
parser.add_argument('-hs', '--hs', type=int, default=0, help='use hierarchical softmax for trainng')
parser.add_argument('-ns', '--ns', type=int, default=5, help='number of samples to use for negative sampling')


args = parser.parse_args()
logging.basicConfig(
    filename=args.model_path.strip() + '.result', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# option 1)
#sentences = PathLineSentences(args.proceedings)

class CreateCorpus(object):
    
    def __init__(self,top_dir):
        self.top_dir = top_dir
        
    """Iterate over all documents, yielding a document (=list of utf8 tokens) at a time."""
    def __iter__(self):
        for root, dirs, files in os.walk(self.top_dir):
            for file in filter(lambda file: file.endswith('.txt'), files):
                text = open(os.path.join(root, file), encoding='utf-8').readlines()
                for sentence in text:
                    yield sentence.split()

# option 2)
class CreateSlice:
    
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fn in os.listdir(self.dirname):
            text = open(os.path.join(self.dirname, fn)).readlines()
            for sentence in text:
                yield sentence.split()

# sentences = ReichstagCorpus(args.proceedings)

start = time.time()
logging.info(f'Training started at: {start}')
if args.model_type == 'word2vec':
	model = Word2Vec(sentences=CreateCorpus(args.proceedings), size=args.size, window=args.window, min_count=args.min_count, workers=args.threads, sg=args.sg, hs=args.hs, negative=args.ns)

elif args.model_type == 'fasttext':
	model = FT_gensim(size=args.size, window=args.window, min_count=args.min_count, workers=args.threads, sg=args.sg, hs=args.hs,negative=args.ns)

	# build the vocabulary
	model.build_vocab(sentences)

	# train the model
	model.train(sentences,epochs=model.epochs,
	                   total_examples=model.corpus_count, 
	                   total_words=model.corpus_total_words)

elapsed = time.time()
logging.info(f'Training finished. Took {elapsed-start} s')
logging.info(f'Vocab size: {len(model.wv.vocab)}')
# Save model to disk
if args.format == 'w2v':
	model.wv.save_word2vec_format(args.model_path + '.txt', binary=True)
else:
	model.wv.save(args.model_path)


# Save vocab to disk 
utils.save_vocab(model, args.vocab_path)












