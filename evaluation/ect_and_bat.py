import sys
sys.path.append('./..')
from utils import *

import os
import codecs
import numpy as np
import json
import time
import glob
import logging
import argparse
from pathlib import Path
from eval import embedding_coherence_test, bias_analogy_test
from weat import XWEAT


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
vocab_path = Path((os.path.join(ROOT_DIR, "../data/vocab")))
models_path = Path((os.path.join(ROOT_DIR, "../models")))

weat_tests = [XWEAT().weat_1, XWEAT().weat_2, XWEAT().weat_3, XWEAT().weat_4, 
XWEAT().weat_5, XWEAT().weat_6, XWEAT().weat_7]

EMB_DIM = 200
def run_bat(vectors, vocab, weat_terms):
  logging.basicConfig(level=logging.INFO)
  logging.info('BAT test started:')
  start_time = time.time()
  result = bias_analogy_test(vectors, vocab, weat_terms[0], weat_terms[1], 
    weat_terms[2], weat_terms[3])
  elapsed = time.time()
  logging.info(f'Time for BAT: {elapsed-start_time}')
  logging.info(f'Result : {result}')
  return result

def run_ect(vectors, vocab, weat_terms, attributes):
  logging.basicConfig(level=logging.INFO)
  logging.info(f'ECT test started with {attributes}')
  start_time = time.time()
  result = embedding_coherence_test(EMB_DIM,vectors, vocab, weat_terms[0], weat_terms[1], attributes)
  elapsed = time.time()
  logging.info('Time for ECT: {}'.format(elapsed-start_time))
  logging.info(result)
  return result

def main():
  parser = argparse.ArgumentParser(description="Running BAT or ECT")
  parser.add_argument("--test_type", type=str, help="Specify BAT or ECT depending on which test shall be run", required=True)
  parser.add_argument("--protocol_type", type=str, help="Whether to run test for Reichstagsprotokolle (RT) or Bundestagsprotokolle (BRD)", required=True)
  parser.add_argument("--output_file", type=str, default=None, help="File to store the results)", required=True)
  parser.add_argument("--vocab_file_pattern", type=str, default=None, help="vocab path file or file pattern in case of multiple files", required=True)
  parser.add_argument("--vector_file_pattern", type=str, default=None, help="vector path file or file pattern in case of multiple files", required=True)
 
  args = parser.parse_args()
	
  if not args.test_type in ['ECT', 'BAT']:
    parser.print_help()
    sys.exit(2)

  vocab_files = glob.glob(str(vocab_path / args.vocab_file_pattern))
  vector_files = glob.glob(str(models_path/ args.vector_file_pattern))

  attribute_sets = {
          'pleasant_unplesant' : PLEASANT + UNPLEASANT,
	  'conspiratorial' : CONSPIRATORIAL_PRO + CONSPIRATORIAL_CON,
	  'economic' : ECONOMIC_PRO + ECONOMIC_CON,
          'outsider_words' : OUTSIDER_WORDS,
          'jewish_nouns' : JEWISH_STEREOTYPES_NOUNS,
          'jewish_character' : JEWISH_STEREOTYPES_CHARACTER,
          'jewish_political' : JEWISH_STEREOTYPES_POLITICAL ,
          'jewish_occupations' : JEWISH_OCCUPATIONS
          }
  if args.protocol_type == 'RT':
    attribute_sets['volkstreu_volksuntreu'] = VOLKSTREU_RT + VOLKSUNTREU_RT
  elif args.protocol_type == 'BRD':
    attribute_sets['volkstreu_volksuntreu'] = VOLKSTREU_BRD + VOLKSUNTREU_BRD
  
  if args.output_file:
    with codecs.open(str(ROOT_DIR) + args.output_file, "w", "utf8") as f:
      for t in zip(vocab_files, vector_files):
        file_name = os.path.splitext(os.path.basename(t[0]))[0]
        f.write(file_name + ':')
        f.write("\n")
        vocab = load_vocab(t[0])
        vectors = load_vectors(t[1])
        for test in weat_tests:
          f.write('{} {}: '.format(args.test_type, test.__name__ ))
          weat_terms = test(args.protocol_type)

          if args.test_type == 'BAT':
            result = run_bat(vectors, vocab, weat_terms) 
            f.write(str(result))
            f.write("\n")
          elif args.test_type == 'ECT':
            emb_size = EMB_DIM
            for att in attribute_sets:
          # if args.attributes:
              atts_to_test = attribute_sets[att]
              result = run_ect(vectors, vocab, weat_terms, atts_to_test)
              f.write("Config: ")
              f.write(str(att) + " and ")
              f.write(str(emb_size) + "\n")
              f.write("Result: ")
              f.write(str(result))
              f.write("\n")
      f.close()

if __name__ == "__main__":
  main()

  
