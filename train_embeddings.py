# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
import multiprocessing as mp
from gensim.models.word2vec import Word2Vec, LineSentence
from gensim.models.fasttext import FastText as FT_gensim
from utils import CreateCorpus, save_vocab
import argparse
import time

PYTHONHASHSEED = 0
MODELS_FOLDER = Path('./models')

parser = argparse.ArgumentParser(description='Train word embedding models from parliamentary protocols')
parser.add_argument('--format', nargs='?', choices=['gensim', 'w2v'], default='gensim', help='Save word2vec model in original word2vec or gensim format', required=True)
parser.add_argument('--protocols', type=str, help='Folder containing pre-processed parliamentary protocols')
parser.add_argument('--model_path', type=str, help='path to store trained model')
parser.add_argument('--vocab_path', type=str, help='path to store model vocab and indices')
parser.add_argument('--model_architecture', nargs='?', choices=['word2vec', 'fasttext'], default='word2vec', help='type of embedding space to train', required=True)
parser.add_argument('-s', '--size', type=int, default=200, help='dimension of word embeddings')
parser.add_argument('-w', '--window', type=int, default=5, help='window size to define context of each word')
parser.add_argument('-m', '--min_count', type=int, default=5, help='minimum frequency of a word to be included in vocab')
parser.add_argument('-t', '--threads', type=int, default=mp.cpu_count(), help='number of worker threads to train word embeddings')
parser.add_argument('-sg', '--sg', type=int, default=0, help='w2v architecture to be used for training: 0 = CBOW; 1 = SG')
parser.add_argument('-hs', '--hs', type=int, default=0, help='use hierarchical softmax for training if hs == 1, else do not use')
parser.add_argument('-ns', '--ns', type=int, default=5, help='number of samples to use for negative sampling')

args = parser.parse_args()
logging.basicConfig(
    filename=args.model_path.strip() + '.result', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

start = time.time()
logging.info(f'Training started at: {start}')

if args.protocols.endswith('.txt'):
    sentences = LineSentence(args.protocols)
else:
    # Read in each document to be processed by word2vec
    sentences = list(CreateCorpus(args.protocols))

if args.model_architecture == 'word2vec':
	model = Word2Vec(sentences=sentences, size=args.size, window=args.window, min_count=args.min_count, workers=args.threads, sg=args.sg, hs=args.hs, negative=args.ns)
elif args.model_architecture == 'fasttext':
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
if args.format == 'gensim':
	model.wv.save(f'{MODELS_FOLDER /args.model_path}',separately= ['vectors'])
elif args.format == 'w2v':
	model.wv.save_word2vec_format(f'{MODELS_FOLDER / args.model_path}.txt', binary=True)
	
# Save vocab to disk 
save_vocab(model, args.vocab_path)
