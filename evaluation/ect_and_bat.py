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

weat_tests = [XWEAT().weat_1, XWEAT().weat_2, XWEAT().weat_3, XWEAT().weat_4]

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
  parser.add_argument("--test_type", nargs='?', choices = ['ECT', 'BAT'], help="Specify BAT or ECT depending on which test shall be run", required=True)
  parser.add_argument("--protocol_type", nargs='?', choices = ['RT', 'BRD'], help="Whether to run test for Reichstagsprotokolle (RT) or Bundestagsprotokolle (BRD)",
 required=True)
  parser.add_argument("--output_file", type=str, default=None, help="File to store the results)", required=True)
  parser.add_argument("--vocab_file", type=str, default=None, help="path to vocab file", required=True)
  parser.add_argument("--vector_file", type=str, default=None, help="path to vector file", required=True)
 
  args = parser.parse_args()
	
  if not args.test_type in ['ECT', 'BAT']:
    parser.print_help()
    sys.exit(2)
	
  vocab = load_vocab(str(vocab_path / args.vocab_file_pattern))
  vectors = load_vectors(str(models_path/ args.vector_file_pattern))
  attribute_sets = create_attribute_sets(vocab, args.protocol_type)
  for test in weat_tests:
    logging.info('{} {}: '.format(args.test_type, test.__name__ ))
    weat_terms = test(args.protocol_type)

    results = {}
    for test in weat_tests:
        results[test.__name__] = {}
        dims = ['sentiment', 'patriotism', 'economic', 'conspiratorial', 'religious', 'racist', 'ethic']
        for dim in dims:
            weat_terms = test(dim, args.protocol_type) 
            if test_type == 'BAT':
                result = run_bat(model, vocab, weat_terms) 
                logging.info(f'{test.__name__} - {dim}: {result}')
                results[test.__name__][dim] = result 
            elif test_type == 'ECT':
                result = run_ect(model, vocab, weat_terms, attribute_sets[f'{dim}_pro'] + attribute_sets[f'{dim}_con'])
                logging.info(f'{test.__name__} - {dim}: {result}')
                results[test.__name__][dim] = result 

    if test_type == 'BAT':
      res_df = pd.DataFrame(results).T.round(3)

    elif test_type == 'ECT':
      res_df = pd.DataFrame(index=pd.MultiIndex.from_product([dims, ['corr', 'p']]),
                         columns=results.keys()).T
      for k1,v1 in results.items():
          for k2, v2 in v1.items():
              res_df.loc[k1, (k2, 'corr')] = results[k1][k2].correlation
              res_df.loc[k1, (k2, 'p')] = results[k1][k2].pvalue

    res_df.to_csv(f'{args.output_file}.csv', index=True, header=True)


if __name__ == "__main__":
  main()

  
