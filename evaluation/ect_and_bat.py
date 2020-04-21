import sys
import os
import codecs
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import random
import json
import time
import glob
import logging
import argparse
from pathlib import Path
from gensim.models import KeyedVectors
from eval import embedding_coherence_test, bias_analogy_test
from weat import XWEAT


debie_path = Path('C:\\Users\\Tobias\\Documents\\Uni Mannheim\\Master Thesis\\notebooks\\DEBIE\\evaluation')

working_path = Path('C:\\Users\\Tobias\\Documents\\Uni Mannheim\\Master Thesis\\notebooks')

weat_tests = [XWEAT().weat_1, XWEAT().weat_2, XWEAT().weat_3, XWEAT().weat_4, 
XWEAT().weat_6]

sys.path.append(os.path.abspath(debie_path))
# Target and attribute sets 

pleasant = 'streicheln, Freiheit, Gesundheit, Liebe, Frieden, Freude, Freund, Himmel, loyal, Vergnügen, Diamant, sanft, ehrlich, glücklich, Regenbogen, Diplom, Geschenk, Ehre, Wunder, Sonnenaufgang, Familie, Lachen, Paradies, Ferien'\
.lower().split(', ') 

unpleasant = 'Mißbrauch, Absturz, Schmutz, Mord, Krankheit, Tod, Trauer, vergiften, stinken, Angriff, Katastrophe, Haß, verschmutzen, Tragödie, Scheidung, Gefängnis, Armut, häßlich, Krebs, töten, faul, erbrechen, Qual'\
.lower().split(', ') 

outsider_words = 'unaufrichtig, komisch, boshaft, unberechenbar, primitiv, beängstigend, hinterlistig, energisch, trügerisch, neidisch, gierig, abscheulich, brutal, ungeheuer, berechnend, grausam, gemein, intolerant, aggressiv'.lower().split(', ') 

jewish_stereotypes = 'Gier, habgierig, geldgierig, hinterlistig, intellektuell, Wucherer, Macht, Einfluß, Kriegstreiber, pervers, Lügner, Weltherrschaft, Kommunismus, Kapitalismus, hinterhältig, betrügerisch, gebeugt, bucklig'.lower().split(', ') 

jewish_stereotypes_nouns = 'Gier, Wucherer, Drückeberger, Kriegsgewinnler, Macht, Einfluß, Kriegstreiber, Lügner, Weltherrschaft, Kommunismus, Kapitalismus, Liberalismus, Außenseiter'.lower().split(', ')

jewish_stereotypes_character = 'egoistisch, fremd, dekadent, haßerfüllt, habgierig, geldgierig, penetrant, hinterlistig, intellektuell, pervers, hinterhältig, betrügerisch, gebeugt, bucklig'.split(', ')

jewish_stereotypes_political = 'liberalistisch, modern, materialistisch, liberal, undeutsch, unpatriotisch, säkular, sozialistisch, links, bolschewistisch'.split(', ')

jewish_occupations = 'Pfandleiher, Geldleiher, Kaufmann, Händler, Bankier, Finanzier, Steuereintreiber, Zöllner, Trödelhändler'.lower().split(', ') 

volkstreu = 'patriotisch, vaterlandsliebe, volksbewußtsein, volksgeist, germanische, deutschnational, nationalbewußtsein, vaterländisch, reichstreu, nationalgesinnt, nationalstolz, deutschnational, königstreu'.split(', ')

volksuntreu = 'nichtdeutsch, fremdländisch, fremd, undeutsch, vaterlandslos, reichsfeind, landesverräter, reichsfeindlich, unpatriotisch, antideutsch, deutschfeindlich, umstürzler'.split(', ')


EMB_DIM = 200

def run_bat(vectors, vocab, weat_targets):
  logging.basicConfig(level=logging.INFO)
  logging.info('BAT test started:')
  start_time = time.time()
  result = bias_analogy_test(vectors, vocab, weat_targets[0], weat_targets[1], 
    weat_targets[2], weat_targets[3])
  elapsed = time.time()
  logging.info('Time for BAT: {}'.format(elapsed-start_time))
  logging.info('Result : %r' % result)
  return result

def run_ect(emb_size, vectors, vocab, weat_targets, attributes):
  logging.basicConfig(level=logging.INFO)
  logging.info('ECT test started with %r:' % attributes)
  start_time = time.time()
  result = embedding_coherence_test(emb_size,vectors, vocab, weat_targets[0], weat_targets[1], attributes)
  elapsed = time.time()
  logging.info('Time for ECT: {}'.format(elapsed-start_time))
  logging.info(result)
  return result

def main():
  parser = argparse.ArgumentParser(description="Running BAT or ECT")
  parser.add_argument('--test_type', type=str, help='Run BAT or ECT', required=True)
  parser.add_argument("--output_file", type=str, default=None, help="File to store the results)", required=True)
  parser.add_argument("--vocab_file_pattern", type=str, default=None, help="vocab path file or file pattern in case of multiple files", required=True)
  parser.add_argument("--vector_file_pattern", type=str, default=None, help="vector path file or file pattern in case of multiple files", required=True)
 
  args = parser.parse_args()

  vocab_files = glob.glob(str(working_path / args.vocab_file_pattern))
  vector_files = glob.glob(str(working_path / args.vector_file_pattern))

  attribute_sets = {
        'pleasant_unplesant' : pleasant + unpleasant,
        'volkstreu_volksuntreu' : volkstreu + volksuntreu,
        'outsider_words' : outsider_words,
        'jewish_nouns' : jewish_stereotypes_nouns,
        'jewish_character' : jewish_stereotypes_character,
        'jewish_political' : jewish_stereotypes_political,
        'jewish_occupations' : jewish_occupations
        }

  if args.output_file:
    with codecs.open(str(debie_path) + args.output_file, "w", "utf8") as f:
      for t in zip(vocab_files, vector_files):
        file_name = os.path.splitext(os.path.basename(t[0]))[0][4:]
        f.write(file_name + ':')
        f.write("\n")
        vocab = json.load(open(t[0], 'r'))
        # vectors = np.load(t[1], allow_pickle=True)
        # TO-DO: change from hardcoded value to emb_size 
        # vectors = np.loadtxt(t[1], skiprows=1 , usecols= (np.arange(1,EMB_DIM+1)))
        vectors = KeyedVectors.load_word2vec_format(t[1], binary=True).vectors
        
        for test in weat_tests:
          f.write('{} {}: '.format(args.test_type, test.__name__ ))
          weat_targets = test()

          if args.test_type == 'BAT':
            result = run_bat(vectors, vocab, weat_targets) 
            f.write(str(result))
            f.write("\n")
          elif args.test_type == 'ECT':
            emb_size = 200
            for att in attribute_sets:
          # if args.attributes:
              atts_to_test = attribute_sets[att]
              result = run_ect(emb_size,vectors, vocab, weat_targets, atts_to_test)
              f.write("Config: ")
              f.write(str(att) + " and ")
              f.write(str(emb_size) + "\n")
              f.write("Result: ")
              f.write(str(result))
              f.write("\n")
      f.close()

if __name__ == "__main__":
  main()

  
