# -*- coding: utf-8 -*-
import os
from eval import eval_simlex
from pathlib import Path
import json
import glob
import numpy as np
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
vocab_path = Path((os.path.join(ROOT_DIR, "../data/vocab")))
models_path = Path((os.path.join(ROOT_DIR, "../models")))

# Get simlex pairs
with open(os.path.join(ROOT_DIR, 'MSimLex999_German.csv'), encoding = 'utf-8') as f:

    text = f.readlines()

    simlex_pairs = []
    for line in text:
      elements = line.split(',')
      simlex_pair = [elements[0], elements[1], elements[-1].strip()]
      simlex_pairs.append(simlex_pair)

simlex_pairs.pop(0)

def main():
  parser = argparse.ArgumentParser(description="Running Simlex test")
  parser.add_argument("--vocab_file_pattern", type=str, default=None, help="vocab path file or file pattern in case of multiple files", required=True)
  parser.add_argument("--vector_file_pattern", type=str, default=None, help="vector path file or file pattern in case of multiple files", required=True)
  parser.add_argument("--output_file", type=str, default=None, help="file to write output to", required=True)

  args = parser.parse_args()
  vocab_files = glob.glob(str(vocab_path / args.vocab_file_pattern))
  vector_files = glob.glob(str(models_path/ args.vector_file_pattern))

  with open(os.path.join(ROOT_DIR, f'simlex/{args.output_file}'), 'w') as f:
    for t in zip(vocab_files, vector_files):
      file_name = os.path.splitext(os.path.basename(t[0]))[0][4:]
      vocab = load_vocab(t[0])
      vectors = load_vectors(t[1])
      simlex_score = eval_simlex(simlex_pairs, vocab, vectors)
      f.write('{}: {}'.format(file_name,simlex_score))
      f.write('\n')
    f.close()

if __name__ == '__main__':
  main()
