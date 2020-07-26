from collections import Counter
import numpy as np
import os
import codecs
from scipy import sparse
import argparse
import logging
import json
from utils import create_attribute_sets, create_target_sets
from nltk.corpus import stopwords
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

def umlauts(text):
    """
    Replace umlauts for a given text
    
    :param word: text as string
    :return: manipulated text as str
    """
    
    tempVar = word # local variable
    
    # Using str.replace() 
    
    tempVar = tempVar.replace('ä', 'ae')
    tempVar = tempVar.replace('ö', 'oe')
    tempVar = tempVar.replace('ü', 'ue')
    tempVar = tempVar.replace('Ä', 'Ae')
    tempVar = tempVar.replace('Ö', 'Oe')
    tempVar = tempVar.replace('Ü', 'Ue')
    tempVar = tempVar.replace('ß', 'ss')
    
    return tempVar

def get_unigrams(corpus, min_count=10, filter_stopwords=False):
    if filter_stopwords:
      german_stop_words = stopwords.words('german')
      german_stop_words_to_use = [umlauts(word) for word in german_stop_words]   # List to hold words after conversion

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

# def get_indices(unigram_counts, attibutes_1, attributes_2):
#     all_words = targets_1 + targets_2
#     tok2indx = {tok: indx for indx, tok in enumerate(all_words)}
#     j = len(all_words)
#     for tok in unigram_counts.keys():
#         if tok not in all_words:
#             tok2indx[tok] = j
#             j += 1
#     indx2tok = {indx: tok for tok,indx in tok2indx.items()}
#     logging.info(f'vocabulary size: {len(tok2indx)}')
#     return tok2indx, indx2tok

def create_index(unigram_counts, kind, full=False, top_dimension='sentiment'):
    attributes =  create_attribute_sets(unigram_counts, kind)
    targets = create_target_sets(unigram_counts, kind)
    top_1, top_2 = convert_attribute_set(top_dimension)
    top_words = attributes[top_1] + attributes[top_2]
    tok2indx = {tok: indx for indx, tok in enumerate(top_words)}
    j = len(all_words)

    if full:
      for tok in unigram_counts.keys():
          if tok not in top_words:
            tok2indx[tok] = j
            j += 1
    else:
      # Only create index including attribute and target sets
      atts = list(dict.fromkeys(list(itertools.chain.from_iterable(attributes.values()))))
      targets = list(dict.fromkeys(list(itertools.chain.from_iterable(targets.values()))))
      words_to_use = atts + targets
      for tok in words_to_use:
          if tok not in top_words:
              tok2indx[tok] = j
              j += 1
    indx2tok = {indx: tok for tok,indx in tok2indx.items()}
    logging.info(f'vocabulary size: {len(tok2indx)}')
    return tok2indx, indx2tok

def get_skipgram_counts(corpus, tok2indx, window_size=2):
    back_window = 2
    front_window = 2
    skipgram_counts = Counter()
    logging.info(f'Get skipgram counts')
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
                skipgram_counts[skipgram] += 1    
        if ix % 200000 == 0:
            logging.info(f'finished {ix/len(corpus):.2%} of corpus')

    logging.info('done')
    logging.info(f'number of skipgrams: {len(skipgram_counts)}')
    return skipgram_counts

def create_coo_mat(skipgram_counts):
    row_indxs = []
    col_indxs = []
    values = []
    ii = 0
    logging.info(f'Create co-occurence matrix')
    for (tok1, tok2), sg_count in skipgram_counts.items():
        ii += 1
        if ii % 1000000 == 0:
            logging.info(f'finished {ii/len(skipgram_counts):.2%} of skipgrams')    
        row_indxs.append(tok1)
        col_indxs.append(tok2)
        values.append(sg_count)
    wwcnt_mat = sparse.csr_matrix((values, (row_indxs, col_indxs)))
    return wwcnt_mat

def create_ppmi_mat(coo_mat, skipgram_counts, smooth=0, neg=1, normalize=False):
    """
    Create PMMI matrix
    
    :coo_mat: co-occurence matrix
    :skipgram_counts
    :smooth: smoothing parameter for add-k smoothing 
    :neg: number of negative samples for computing shifted PPMI
    
    """
    num_skipgrams = coo_mat.sum()
    assert(sum(skipgram_counts.values())==num_skipgrams)

    prob_norm = coo_mat.sum() + (coo_mat.shape[0] * coo_mat.shape[1]) * smooth

    # for creating sparce matrices
    row_indxs = []
    col_indxs = []
    ppmi_values = []   # positive pointwise mutial information
    sppmi_values = []  # smoothed positive pointwise mutual information

    sum_over_words = np.array(coo_mat.sum(axis=0)).flatten() + smooth
    sum_over_contexts = np.array(coo_mat.sum(axis=1)).flatten() + smooth

    # Shifted PPMI 
    neg = np.log(neg)

    # context-distribution smoothing
    alpha = 0.75
    sum_over_words_alpha = sum_over_words**alpha
    nca_denom = np.sum(sum_over_words_alpha)

    ii = 0
    logging.info(f'Create PPMI matrix')
    for (tok_word, tok_context), sg_count in skipgram_counts.items():
        ii += 1
        if ii % 1000000 == 0:
            logging.info(f'finished {ii/len(skipgram_counts):.2%} of skipgrams')

        nwc = sg_count + smooth
        Pwc = nwc / prob_norm
        nw = sum_over_contexts[tok_word]
        Pw = nw / prob_norm
        nc = sum_over_words[tok_context]
        Pc = nc / prob_norm
        # Give rare words higher probability
        nca = sum_over_words_alpha[tok_context]
        Pca = nca / nca_denom

        pmi = np.log(Pwc/(Pw*Pc))   
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
  parser.add_argument("--corpus", type=str, help="Corpus path", required=True)
  parser.add_argument("--protocol_type", nargs='?', choices = ['RT', 'BRD'], help="Whether to run test for Reichstagsprotokolle (RT) or Bundestagsprotokolle (BRD)",
 required=True)
  # parser.add_argument("--top_dimension", type=str, help='Which attribute set to be used for subsequent label propagation - either sentiment, patriotism, economic or conspiratorial')
  parser.add_argument("--min_count", type=int, help="Minimum number of occurences of a word to be included in PPMI matrix", required=True)
  parser.add_argument("--full", action='store_true', help="Compute full PPMI matrix - in default mode only PPMI mat containing target and attribute sets is computed")
  parser.add_argument("--window_size", type=int, help="Window size to use for creating CO and PPMI matrices", required=True)
  parser.add_argument("--normalize", action='store_true', help="Whether to normalize PPMI matrix")
  parser.add_argument("--smooth", type=int, help="Smoothing parameter for add-k smoothing", default=0)
  parser.add_argument("--neg", type=int, help="Number of negative samples to use for computing shifted PPMI", default=1)
  parser.add_argument("--output_file", type=str, help='Output file to store matrix')

  args = parser.parse_args()
  sentences = list(CreateCorpus(args.corpus))
  unigrams = get_unigrams(sentences, min_count=args.min_count)
  # attributes = create_attribute_sets(unigrams, kind=args.protocol_type)
  # att_1, att_2 = convert_attribute_set(args.attribute_specifications)
  if args.full:
    tok2indx, indx2tok = create_index(unigrams, kind= args.protocol_type, full=args.full)
  else:
    tok2indx, indx2tok = create_index(unigrams, kind= args.protocol_type)
  skipgrams = get_skipgram_counts(sentences, tok2indx, args.window_size)
  coo_mat = create_coo_mat(skipgrams)

  ppmi_mat, sppmi_mat = create_ppmi_mat(coo_mat, skipgrams,args.smooth, args.neg, args.normalize)
  # Save matrices
  if not os.path.exists('matrices'):
    os.makedirs('matrices')
    
  sparse.save_npz(f'matrices/ppmi_{args.output_file}.npz', ppmi_mat, compressed=True)
  sparse.save_npz(f'matrices/sppmi_{args.output_file}.npz', sppmi_mat, compressed=True)

  # Save token-2-index dictionaries
  if not os.path.exists('tok2indx'):
    os.makedirs('tok2indx')
  with codecs.open(f'tok2indx/{args.output_file}.json',"w", encoding='utf-8') as f:
      f.write(json.dumps(tok2indx))


if __name__ == "__main__":
  main()
