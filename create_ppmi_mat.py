from collections import Counter
import numpy as np
import os
import codecs
from scipy import sparse
from text_preprocessing import remove_umlauts
from bias_specifications import antisemitic_streams
import argparse
import logging
import json
import itertools
from utils import create_attribute_sets, create_target_sets, convert_attribute_set, CreateCorpus
from nltk.corpus import stopwords
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

def get_unigrams(corpus, min_count=10, filter_stopwords=False):
    if filter_stopwords:
      german_stop_words = stopwords.words('german')
      german_stop_words = remove_umlauts(german_stop_words)
      german_stop_words.append('0')

    unigram_counts = Counter()
    logging.info(f'Get unigrams')
    for ii, sent in enumerate(corpus):
        if ii % 200000 == 0:
            logging.info(f'finished {ii/len(corpus):.2%} of corpus')
        for token in sent:
          if filter_stopwords and token not in german_stop_words:
              unigram_counts[token] += 1
          else:
              unigram_counts[token] += 1
    unigram_counts = {k:v for k,v in unigram_counts.items() if v >= min_count}                
    return unigram_counts

def create_index(unigram_counts, kind, full=False, top_attribute='sentiment'):
    """Create tok2word and word2tok indices

    :param unigram_counts: unigram counts derived from a corpus
    :param kind: kind of protocols instance - either RT or BRD
    :param full: Whether to compute the full matrix or mini-matrix restricted to attribute and term sets
    :param top_attribute: which attribute to be ordered on top of the matrix
    """

    # Initialize attribute and target sets of the corpus
    attributes = create_attribute_sets(unigram_counts, kind)
    targets = create_target_sets(unigram_counts, kind)
    top_words = attributes[f'{top_attribute}_pro'] + attributes[f'{top_attribute}_con']
    tok2indx = {tok: indx for indx, tok in enumerate(top_words)}
    j = len(top_words)

    if full:
      for tok in unigram_counts.keys():
          if tok not in top_words:
            tok2indx[tok] = j
            j += 1
    else:
      # Only create index including attribute and target sets
      atts = list(dict.fromkeys(list(itertools.chain.from_iterable(attributes.values()))))
      targets = list(dict.fromkeys(list(itertools.chain.from_iterable(targets.values()))))
      matrix_terms = atts + targets
      for tok in matrix_terms:
          if tok not in top_words:
              tok2indx[tok] = j
              j += 1
    indx2tok = {indx: tok for tok,indx in tok2indx.items()}
    logging.info(f'vocabulary size: {len(tok2indx)}')
    return tok2indx, indx2tok

def get_coo_counts(corpus: list, tok2indx: dict, window_size=2):
    """Retrieve co-occurence counts

    Retrieve co-occurence counts within given window of given size by looping trough each target word in a sentence and examining n words behind and in front of the focus word

    :param corpus: list of sentences composing the corpus
    :tok2indx: word2tok index of the corpus
    :window size: int to define the size of the context
    """

    back_window = window_size
    front_window = window_size
    coo_counts = Counter()
    logging.info(f'Get co-occurrence counts')
    for ix, sent in enumerate(corpus):
        tokens = [tok2indx[tok] for tok in sent if tok in tok2indx]
        for ii_word, word in enumerate(tokens):
            ii_context_min = max(0, ii_word - back_window)
            ii_context_max = min(len(tokens) - 1, ii_word + front_window)
            ii_contexts = [
                ii for ii in range(ii_context_min, ii_context_max + 1) 
                if ii != ii_word]
            for ii_context in ii_contexts:
                skipgram = (tokens[ii_word], tokens[ii_context])
                coo_counts[skipgram] += 1    
        if ix % 200000 == 0:
            logging.info(f'finished {ix/len(corpus):.2%} of corpus')

    logging.info('done')
    logging.info(f'number of co-occurring word pairs: {len(coo_counts)}')
    return coo_counts

def create_coo_mat(coo_counts: dict):
    """Create co-occurrence matrix

    :param coo_counts: co-occurrence counts of the corpus
    """

    row_indxs = []
    col_indxs = []
    values = []
    ii = 0
    logging.info(f'Create co-occurence matrix')
    for (tok1, tok2), sg_count in coo_counts.items():
        ii += 1
        if ii % 1000000 == 0:
            logging.info(f'finished {ii/len(coo_counts):.2%} of skipgrams')    
        row_indxs.append(tok1)
        col_indxs.append(tok2)
        values.append(sg_count)
    wwcnt_mat = sparse.csr_matrix((values, (row_indxs, col_indxs)))
    return wwcnt_mat

def create_ppmi_mat(coo_ma, coo_counts, smooth=0, neg=1, normalize=False):
    """Create PMMI matrix
    
    :param coo_mat: co-occurrence matrix of the corpus
    :param coo_counts of the corpus
    :param smooth: smoothing parameter for add-k smoothing 
    :param neg: number of negative samples for computing shifted PPMI
    :param normalize: whether PPMI matrix should be normalized
    """

    # Sanity check
    num_skipgrams = coo_mat.sum()
    assert(sum(coo_counts.values())==num_skipgrams)

    prob_norm = coo_mat.sum() + (coo_mat.shape[0] * coo_mat.shape[1]) * smooth

    # For creating sparce matrices
    row_indxs = []
    col_indxs = []
    ppmi_values = []   # positive pointwise mutial information
    sppmi_values = []  # smoothed positive pointwise mutual information


    sum_over_words = np.array(coo_mat.sum(axis=0)).flatten() + smooth
    sum_over_contexts = np.array(coo_mat.sum(axis=1)).flatten() + smooth

    # Shifted PPMI - neg=1 will not cause shifting
    neg = np.log(neg)

    # context-distribution smoothing acc. to Levy et al. (2014)
    alpha = 0.75
    sum_over_words_alpha = sum_over_words**alpha
    nca_denom = np.sum(sum_over_words_alpha)

    ii = 0
    logging.info(f'Create PPMI matrix')
    for (tok_word, tok_context), sg_count in coo_counts.items():
        ii += 1
        if ii % 1000000 == 0:
            logging.info(f'finished {ii/len(coo_counts):.2%} of skipgrams')

        nwc = sg_count + smooth
        Pwc = nwc / prob_norm
        nw = sum_over_contexts[tok_word]
        Pw = nw / prob_norm
        nc = sum_over_words[tok_context]
        Pc = nc / prob_norm
        # Give rare words higher probability
        nca = sum_over_words_alpha[tok_context]
        Pca = nca / nca_denom

        pmi = np.log(Pwc/(Pw*Pc)) - neg   
        ppmi = max(pmi, 0)
        spmi = np.log(Pwc/(Pw*Pca))
        sppmi = max(spmi, 0)

        row_indxs.append(tok_word)
        col_indxs.append(tok_context)
        ppmi_values.append(ppmi)
        sppmi_values.append(sppmi)

    ppmi_mat = sparse.csr_matrix((ppmi_values, (row_indxs, col_indxs)))
    sppmi_mat = sparse.csr_matrix((sppmi_values, (row_indxs, col_indxs)))    
    return ppmi_mat, sppmi_mat

def main():
  parser = argparse.ArgumentParser(description="Compute PPMI matrix")
  parser.add_argument("--protocols", type=str, help="Path to protocols", required=True)
  parser.add_argument("--protocol_type", nargs='?', choices = ['RT', 'BRD'], help="Whether to run test for Reichstagsprotokolle (RT) or Bundestagsprotokolle (BRD)",
 required=True)
  # parser.add_argument("--top_attribute", type=str, help='Which attribute set to be used for subsequent label propagation - either sentiment, patriotism, economic or conspiratorial')
  parser.add_argument("--min_count", type=int, help="Minimum number of occurences of a word to be included in PPMI matrix", required=True)
  parser.add_argument("--full", action='store_true', help="Compute full PPMI matrix - in default mode only PPMI mat containing target and attribute sets of all bias specifications is computed")
  parser.add_argument("--window_size", type=int, help="Window size to use for creating CO and PPMI matrices", required=True)
  parser.add_argument("--normalize", action='store_true', help="Whether to normalize PPMI matrix")
  parser.add_argument("--smooth", type=int, help="Smoothing parameter for add-k smoothing", default=0)
  parser.add_argument("--neg", type=int, help="Number of negative samples to use for computing shifted PPMI", default=1)
  parser.add_argument("--output_file", type=str, help='Output file to store matrix')

  args = parser.parse_args()
  sentences = list(CreateCorpus(args.protocols))
  unigrams = get_unigrams(sentences, min_count=args.min_count)
  
  if args.full:
    tok2indx, indx2tok = create_index(unigrams, kind= args.protocol_type, full=args.full)
  else:
    tok2indx, indx2tok = create_index(unigrams, kind= args.protocol_type)
  skipgrams = get_coo_counts(sentences, tok2indx, args.window_size)
  coo_mat = create_coo_mat(skipgrams)

  ppmi_mat, sppmi_mat = create_ppmi_mat(coo_mat, skipgrams,args.smooth, args.neg, args.normalize)
  # Save matrices
  if not os.path.exists('matrices'):
    os.makedirs('matrices')
    
  sparse.save_npz(f'matrices/ppmi_{args.output_file}.npz', ppmi_mat, compressed=True)
  sparse.save_npz(f'matrices/sppmi_{args.output_file}.npz', sppmi_mat, compressed=True)

  # Save tok2indx dictionaries
  if not os.path.exists('ppmi_vocab'):
    os.makedirs('ppmi_vocab')
  with codecs.open(f'ppmi_vocab/{args.output_file}.json',"w", encoding='utf-8') as f:
      f.write(json.dumps(tok2indx))


if __name__ == "__main__":
  main()
