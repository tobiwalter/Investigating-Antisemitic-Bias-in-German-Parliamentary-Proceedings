# -*- coding: utf-8 -*-
import sys
sys.path.append('./..')
from utils import load_vocab, load_vectors
import os 
import glob
from eval import eval_k_means
from weat import XWEAT
from pathlib import Path
import json
import argparse
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
vocab_path = Path((os.path.join(ROOT_DIR, "../data/vocab")))
models_path = Path((os.path.join(ROOT_DIR, "../models")))

weat_tests = [XWEAT().weat_1, XWEAT().weat_2, XWEAT().weat_3, XWEAT().weat_4, XWEAT().weat_6]

def main():
  parser = argparse.ArgumentParser(description="Running K-means test")
  parser.add_argument("--vocab_file_pattern", type=str, default=None, help="vocab path file or file pattern in case of multiple files", required=True)
  parser.add_argument("--vector_file_pattern", type=str, default=None, help="vector path file or file pattern in case of multiple files", required=True)
  parser.add_argument("--protocol_type", type=str, help="Whether to run test for Reichstagsprotokolle (RT) or Bundestagsprotokolle (BRD)", required=True)

  args = parser.parse_args()

  vocab_files = glob.glob(str(vocab_path / args.vocab_file_pattern))
  vector_files = glob.glob(str(models_path/ args.vector_file_pattern))
  print(vocab_files)
  print(vector_files)
  for t in zip(vocab_files, vector_files):
      file_name = os.path.splitext(os.path.basename(t[0]))[0]
      vocab = load_vocab(t[0])
      vectors = load_vectors(t[1])
      with open(os.path.join(ROOT_DIR, f'kmeans/{file_name}.txt'), 'w') as f:
        for test in weat_tests:
          f.write(f'K-means score {test.__name__ }: ')
          targets_1 = test(args.protocol_type)[0]
          targets_2 = test(args.protocol_type)[1]
          k_means_score = eval_k_means(targets_1, targets_2, vectors, vocab)
          f.write(str(k_means_score))
          f.write('\n')
      f.close()
if __name__ == '__main__':
    main()
