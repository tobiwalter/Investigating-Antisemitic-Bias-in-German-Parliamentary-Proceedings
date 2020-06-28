import sys
sys.path.append('./..')
from utils import *
print(PLEASANT)

import numpy as np
import random
from itertools import filterfalse
from itertools import combinations
import codecs
import os
import logging
import argparse
import time
from collections import OrderedDict
import math

class XWEAT(object): 
  """
  Credits: This implementation is based on https://github.com/umanlp/XWEAT/blob/master/weat.py

  Perform WEAT (Word Embedding Association Test) bias tests on a language model.
  Follows from Caliskan et al 2017 (10.1126/science.aal4230).
  """

  def __init__(self):
      self.embd_dict = None
      self.vocab = None
      self.embedding_matrix = None

  def set_embd_dict(self, embd_dict):
      self.embd_dict = embd_dict


  def convert_by_vocab(self, items):
      """Converts a sequence of [tokens|ids] using the vocab."""
      output = [self.vocab[item] for item in items if item in self.vocab]
      return output

  def _build_vocab_dict(self, vocab):
      self.vocab = OrderedDict()
      vocab = set(vocab)
      index = 0
      for term in vocab:
        if term in self.embd_dict:
          self.vocab[term] = index
          index += 1
        else:
          logging.warning("Not in vocab %s", term)

  def _build_embedding_matrix(self):
      self.embedding_matrix = []
      for term, index in self.vocab.items():
        if term in self.embd_dict:
          self.embedding_matrix.append(self.embd_dict[term])
        else:
          raise AssertionError("This should not happen.")
      self.embd_dict = None

  def mat_normalize(self,mat, norm_order=2, axis=1):
    return mat / np.transpose([np.linalg.norm(mat, norm_order, axis)])

  def cosine(self, a, b):
    norm_a = self.mat_normalize(a)
    norm_b = self.mat_normalize(b)
    cos = np.dot(norm_a, np.transpose(norm_b))
    return cos

  def _init_similarities(self, similarity_type):
     if similarity_type == "cosine":
      self.similarities = self.cosine(self.embedding_matrix, self.embedding_matrix)
     else:
      raise NotImplementedError()

  def weat_1(self, protocol_type):
      """
      WEAT 1 - terms representing judaism and christianism + sets of pleasant/unplesant wordargs.protocol_typeselfs
      """
      if protocol_type == 'BT':
        targets_1 = CHRISTIAN_BRD
        targets_2 = JEWISH_BRD

      elif protocol_type == 'RT':
        targets_1 = CHRISTIAN_RT
        targets_2 = JEWISH_RT

      attributes_1 = PLEASANT 
      attributes_2 = UNPLEASANT
      return targets_1, targets_2, attributes_1, attributes_2


  def weat_2(self, protocol_type):
      """
      WEAT 2 - terms representing catholicism and protestantism + sets of pleasant/unplesant words
      """
      if protocol_type == 'BT':
        targets_1 = PROTESTANT_BRD
        targets_2  = CATHOLIC_BRD

      elif protocol_type == 'RT':
        targets_1 = PROTESTANT_RT
        targets_2  = CATHOLIC_RT
        attributes_1 = PLEASANT 
        attributes_2 = UNPLEASANT
      return targets_1, targets_2, attributes_1, attributes_2


  def weat_3(self, protocol_type):
      """
      WEAT 3 - terms representing protestantism and judaism + sets of pleasant/unplesant words
      """
      if protocol_type == 'BT':
        targets_1 = PROTESTANT_BRD
        targets_2 = JEWISH_BRD

      elif protocol_type == 'RT':
        targets_1 = PROTESTANT_RT
        targets_2 = JEWISH_RT

      attributes_1 = PLEASANT 
      attributes_2 = UNPLEASANT
      return targets_1, targets_2, attributes_1, attributes_2


  def weat_4(self, protocol_type):
      """
      WEAT 4 - terms representing catholicism and judaism + sets of pleasant/unplesant words
      """
      if protocol_type == 'BT':
        targets_1 = CATHOLIC_BRD
        targets_2 = JEWISH_BRD

      elif protocol_type == 'RT':
        targets_1 = CATHOLIC_RT
        targets_2 = JEWISH_RT

      attributes_1 = PLEASANT 
      attributes_2 = UNPLEASANT
      return targets_1, targets_2, attributes_1, attributes_2

    # again african american vs european american names, but with different attributes
  def weat_5(self, protocol_type):
      # excluded as in the original paper: Jay, Kristen, (here only excluded in the glove experiments)

      if protocol_type == 'BT':
        targets_1 = CHRISTIAN_BRD
        targets_2 = JEWISH_BRD

      elif protocol_type == 'RT':
        targets_1 = CHRISTIAN_RT
        targets_2 = JEWISH_RT

      attributes_1 = ECONOMIC_PRO
      attributes_2 = ECONOMIC_CON
      return targets_1, targets_2, attributes_1, attributes_2


  def weat_6(self, protocol_type):
      """
      WEAT 6- terms representing patriotism and non-patriotism+ sets of pleasant/unplesant words
      """
      if protocol_type == 'BT':
        targets_1 = CHRISTIAN_BRD
        targets_2 = JEWISH_BRD
        attributes_1 = VOLKSTREU_BRD
        attributes_2 = VOLKSUNTREU_BRD

      elif protocol_type == 'RT':
        targets_1 = CHRISTIAN_RT
        targets_2 = JEWISH_RT
        attributes_1 = VOLKSTREU_RT
        attributes_2 = VOLKSUNTREU_RT
      return targets_1, targets_2, attributes_1, attributes_2


  def weat_7(self, protocol_type):
      # excluded as in the original paper: Jay, Kristen, (here only excluded in the glove experiments)

      if protocol_type == 'BT':
        targets_1 = CHRISTIAN_BRD
        targets_2 = JEWISH_BRD

      elif protocol_type == 'RT':
        targets_1 = CHRISTIAN_RT
        targets_2 = JEWISH_RT

      attributes_1 = CONSPIRATORIAL_PRO
      attributes_2 = CONSPIRATORIAL_CON
      return targets_1, targets_2, attributes_1, attributes_2

  def similarity_precomputed_sims(self, w1, w2, type="cosine"):
      return self.similarities[w1, w2]


  def word_association_with_attribute_precomputed_sims(self, w, A, B):
    return np.mean([self.similarity_precomputed_sims(w, a) for a in A]) - np.mean([self.similarity_precomputed_sims(w, b) for b in B])


  def differential_association_precomputed_sims(self, T1, T2, A1, A2):
    return np.sum([self.word_association_with_attribute_precomputed_sims(t1, A1, A2) for t1 in T1]) \
           - np.sum([self.word_association_with_attribute_precomputed_sims(t2, A1, A2) for t2 in T2])


  def weat_effect_size_precomputed_sims(self, T1, T2, A1, A2):
    return (
             np.mean([self.word_association_with_attribute_precomputed_sims(t1, A1, A2) for t1 in T1]) -
             np.mean([self.word_association_with_attribute_precomputed_sims(t2, A1, A2) for t2 in T2])
           ) / np.std([self.word_association_with_attribute_precomputed_sims(w, A1, A2) for w in T1 + T2])


  def _random_permutation(self, iterable, r=None):
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))

  def weat_p_value_precomputed_sims(self, T1, T2, A1, A2, sample):
    logging.info("Calculating p value ... ")
    size_of_permutation = min(len(T1), len(T2))
    T1_T2 = T1 + T2
    observed_test_stats_over_permutations = []
    total_possible_permutations = math.factorial(len(T1_T2)) / math.factorial(size_of_permutation) / math.factorial((len(T1_T2)-size_of_permutation))
    logging.info("Number of possible permutations: %d", total_possible_permutations)
    if not sample or sample >= total_possible_permutations:
      permutations = combinations(T1_T2, size_of_permutation)
    else:
      logging.info("Computing randomly first %d permutations", sample)
      permutations = set()
      while len(permutations) < sample:
        permutations.add(tuple(sorted(self._random_permutation(T1_T2, size_of_permutation))))

    for Xi in permutations:
      Yi = filterfalse(lambda w: w in Xi, T1_T2)
      observed_test_stats_over_permutations.append(self.differential_association_precomputed_sims(Xi, Yi, A1, A2))
      if len(observed_test_stats_over_permutations) % 100000 == 0:
        logging.info("Iteration %s finished", str(len(observed_test_stats_over_permutations)))
    unperturbed = self.differential_association_precomputed_sims(T1, T2, A1, A2)
    is_over = np.array([o > unperturbed for o in observed_test_stats_over_permutations])
    return is_over.sum() / is_over.size


  def weat_stats_precomputed_sims(self, T1, T2, A1, A2, sample_p=None):
    test_statistic = self.differential_association_precomputed_sims(T1, T2, A1, A2)
    effect_size = self.weat_effect_size_precomputed_sims(T1, T2, A1, A2)
    p = self.weat_p_value_precomputed_sims(T1, T2, A1, A2, sample=sample_p)
    return test_statistic, effect_size, p


  def _create_vocab(self):
      """
      >>> weat = XWEAT(None); weat._create_vocab()
      :return: all
      """
      all = []
      for i in range(1, 8):
        t1, t2, a1, a2 = getattr(self, "weat_" + str(i))()
        all = all + t1 + t2 + a1 + a2
      all = set(all)
      return all

  def _output_vocab(self, path="./data/vocab_en.txt"):
      """
      >>> weat = XWEAT(None); weat._output_vocab()
      """
      vocab = self._create_vocab()
      with codecs.open(path, "w", "utf8") as f:
        for w in vocab:
              f.write(w)
              f.write("\n")
        f.close()

  def run_test_precomputed_sims(self, target_1, target_2, attributes_1, attributes_2, sample_p=None, similarity_type="cosine"):
      """Run the WEAT test for differential association between two
      sets of target words and two sets of attributes.
      RETURNS:
          (d, e, p). A tuple of floats, where d is the WEAT Test statistic,
          e is the effect size, and p is the one-sided p-value measuring the
          (un)likeliness of the null hypothesis (which is that there is no
          difference in association between the two target word sets and
          the attributes).
          If e is large and p small, then differences in the model between
          the attribute word sets match differences between the targets.
      """
      vocab = target_1 + target_2 + attributes_1 + attributes_2
      self._build_vocab_dict(vocab)
      T1 = self.convert_by_vocab(target_1)
      T2 = self.convert_by_vocab(target_2)
      A1 = self.convert_by_vocab(attributes_1)
      A2 = self.convert_by_vocab(attributes_2)
      while len(T1) < len(T2):
        logging.info("Popped T2 %d", T2[-1])
        T2.pop(-1)
      while len(T2) < len(T1):
        logging.info("Popped T1 %d", T1[-1])
        T1.pop(-1)
      while len(A1) < len(A2):
        logging.info("Popped A2 %d", A2[-1])
        A2.pop(-1)
      while len(A2) < len(A1):
        logging.info("Popped A1 %d", A1[-1])
        A1.pop(-1)
      assert len(T1)==len(T2)
      assert len(A1) == len(A2)
      self._build_embedding_matrix()
      self._init_similarities(similarity_type)

      print("To compute WEAT test")
      return self.weat_stats_precomputed_sims(T1, T2, A1, A2, sample_p)


def main():
  def boolean_string(s):
    if s not in {'False', 'True', 'false', 'true'}:
      raise ValueError('Not a valid boolean string')
    return s == 'True' or s == 'true'
  parser = argparse.ArgumentParser(description="Running XWEAT")
  parser.add_argument("--test_number", type=int, help="Number of the weat test to run", required=False)
  parser.add_argument("--protocol_type", type=str, help="Whether to run test for Reichstagsprotokolle (RT) or Bundestagsprotokolle (BRD)", required=True)
  parser.add_argument("--permutation_number", type=int, default=None,
                      help="Number of permutations (otherwise all will be run)", required=False)
  parser.add_argument("--output_file", type=str, default=None, help="File to store the results)", required=False)
  parser.add_argument("--lower", type=boolean_string, default=False, help="Whether to lower the vocab", required=True)
  parser.add_argument("--embedding_vocab", type=str, help="Vocab of the embeddings")
  parser.add_argument("--embedding_vectors", type=str, help="Vectors of the embeddings")
  parser.add_argument("--is_vec_format", type=boolean_string, default=False, help="Whether embeddings are in vec format")
  parser.add_argument("--embeddings", type=str, help="Vectors and vocab of the embeddings")
  args = parser.parse_args()

  start = time.time()
  logging.basicConfig(level=logging.INFO)
  logging.info("XWEAT started")
  weat = XWEAT()

  if args.test_number == 1:
    targets_1, targets_2, attributes_1, attributes_2 = weat.weat_1(args.protocol_type)
  elif args.test_number == 2:
    targets_1, targets_2, attributes_1, attributes_2 = weat.weat_2(args.protocol_type)
  elif args.test_number == 3:
    targets_1, targets_2, attributes_1, attributes_2 = weat.weat_3(args.protocol_type)
  elif args.test_number == 4:
    targets_1, targets_2, attributes_1, attributes_2 = weat.weat_4(args.protocol_type)
  elif args.test_number == 5:
    targets_1, targets_2, attributes_1, attributes_2 = weat.weat_5(args.protocol_type)
  elif args.test_number == 6:
    targets_1, targets_2, attributes_1, attributes_2 = weat.weat_6(args.protocol_type)
  elif args.test_number == 7:
    targets_1, targets_2, attributes_1, attributes_2 = weat.weat_7(args.protocol_type)
   
  else:
    raise ValueError("Only WEAT 1 to 7 are supported")

  if args.lower:
    targets_1 = [t.lower() for t in targets_1]
    targets_2 = [t.lower() for t in targets_2]
    attributes_1 = [a.lower() for a in attributes_1]
    attributes_2 = [a.lower() for a in attributes_2]

  elif args.is_vec_format:
    logging.info("Embeddings are in vec format")
    embd_dict = load_embeddings(args.embeddings)
  else:
    embd_dict = load_embedding_dict(vocab_path=args.embedding_vocab, vector_path=args.embedding_vectors, glove=False)
  weat.set_embd_dict(embd_dict)

  logging.info("Running test")
  result = weat.run_test_precomputed_sims(targets_1, targets_2, attributes_1, attributes_2, args.permutation_number)
  logging.info(result)
  with codecs.open(args.output_file, "w", "utf8") as f:
    f.write("Config: ")
    f.write(str(args.test_number) + " and ")
    f.write(str(args.lower) + " and ")
    f.write(str(args.permutation_number) + "\n")
    f.write("Result: ")
    f.write(str(result))
    f.write("\n")
    end = time.time()
    duration_in_hours = ((end - start) / 60) / 60
    f.write(str(duration_in_hours))
    f.close()

if __name__ == "__main__":
  main()
