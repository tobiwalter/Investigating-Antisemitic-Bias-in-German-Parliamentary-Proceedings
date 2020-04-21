# -*- coding: utf-8 -*-
import csv
import os 
import glob
from eval import eval_simlex, eval_k_means, eval_svm
from weat import XWEAT, load_vocab_goran
from pathlib import Path
import json
import pickle 
import argparse
import numpy as np
from gensim.models import KeyedVectors

debie_path = os.path.dirname(os.path.abspath(__file__))
working_path = os.path.abspath(os.path.join(debie_path, "../.."))



weat_tests = [XWEAT().weat_1, XWEAT().weat_2, XWEAT().weat_3, XWEAT().weat_4, XWEAT().weat_6]

def main():
  parser = argparse.ArgumentParser(description="Running Simlex test")
  parser.add_argument("--vocab_file", type=str, default=None, help="vocab path file", required=True)
  parser.add_argument("--vector_file", type=str, default=None, help="vector path file", required=True)

  args = parser.parse_args()
  vocab_files = glob.glob(os.path.join(working_path, args.vocab_file))
  vector_files = glob.glob(os.path.join(working_path,args.vector_file))
  for t in zip(vocab_files, vector_files):
      file_name = os.path.splitext(os.path.basename(t[0]))[0]
      vocab = json.load(open(t[0], 'r'))
      vectors =  KeyedVectors.load_word2vec_format(t[1], binary=True).vectors
      with open(os.path.join(debie_path, f'kmeans\\{file_name}.txt'), 'w') as f:
        for test in weat_tests:
          f.write(f'K-means score {test.__name__ }: ')
          targets_1 = test()[0]
          targets_2 = test()[1]
          k_means_score = eval_k_means(targets_1, targets_2, vectors, vocab)
          f.write(str(k_means_score))
          f.write('\n')
      f.close()
if __name__ == '__main__':
    main()
