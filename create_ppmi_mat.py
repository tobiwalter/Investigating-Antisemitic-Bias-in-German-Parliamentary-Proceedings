from collections import Counter
import numpy as np
from scipy import sparse
import argparse
import logging
import json
from representations.utils import CreateCorpus
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

def get_unigrams(corpus, min_count=5):
    unigram_counts = Counter()
    logging.info(f'Get unigrams')
    for ii, sent in enumerate(corpus):
        if ii % 200000 == 0:
            logging.info(f'finished {ii/len(corpus):.2%} of corpus')
        for token in sent:
            unigram_counts[token] += 1
    unigram_counts = {k:v for k,v in unigram_counts.items() if v >= min_count}                
    return unigram_counts

def get_indices(unigram_counts, targets_1, targets_2):
    all_words = targets_1 + targets_2
    tok2indx = {tok: indx for indx, tok in enumerate(all_words)}
    j = len(all_words)
    for tok in unigram_counts.keys():
        if tok not in all_words:
            tok2indx[tok] = j
            j += 1
    indx2tok = {indx: tok for tok,indx in tok2indx.items()}
    logging.info(f'vocabulary size: {len(tok2indx)}')
    return tok2indx, indx2tok

def get_skipgram_counts(corpus, tok2indx, indx2tok):
    skipgram_counts = Counter()

    for ix, sent in enumerate(corpus):
        tokens = [tok2indx[tok] for tok in sent if tok in tok2indx]
        logging.info(f'Get skipgram counts')
        for ii_word, word in enumerate(tokens):
            ii_contexts = [
                ii for ii in range(len(tokens)) 
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

def create_ppmi_mat(coo_mat, skipgram_counts):  
    num_skipgrams = coo_mat.sum()
    assert(sum(skipgram_counts.values())==num_skipgrams)

    # for creating sparce matrices
    row_indxs = []
    col_indxs = []
    ppmi_values = []   # positive pointwise mutial information
    sppmi_values = []  # smoothed positive pointwise mutual information

    sum_over_words = np.array(coo_mat.sum(axis=0)).flatten()
    sum_over_contexts = np.array(coo_mat.sum(axis=1)).flatten()

    # smoothing
    alpha = 0.75
    sum_over_words_alpha = sum_over_words**alpha
    nca_denom = np.sum(sum_over_words_alpha)

    ii = 0
    logging.info(f'Create PPMI matrix')
    for (tok_word, tok_context), sg_count in skipgram_counts.items():
        ii += 1
        if ii % 1000000 == 0:
            logging.info(f'finished {ii/len(skipgram_counts):.2%} of skipgrams')

        nwc = sg_count
        Pwc = nwc / num_skipgrams
        nw = sum_over_contexts[tok_word]
        Pw = nw / num_skipgrams
        nc = sum_over_words[tok_context]
        Pc = nc / num_skipgrams
        # Give rare words higher probability
        nca = sum_over_words_alpha[tok_context]
        Pca = nca / nca_denom

        pmi = np.log2(Pwc/(Pw*Pc))   
        ppmi = max(pmi, 0)
        spmi = np.log2(Pwc/(Pw*Pca))
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
  parser.add_argument("--protocol_type", type=str, help="Run tests for Reichstagsprotokolle or Bundestagsprotokolle?", required=True)
  parser.add_argument("--attribute_specifications", type=str, help='Which attribute set to be used for subsequent label propagation - either sentiment, patriotism, economic or conspiratorial')
  parser.add_argument("--output_file", type=str, help='Output file to store matrix')

  args = parser.parse_args()
  sentences = list(CreateCorpus(args.corpus))
  unigrams = get_unigrams_sorted(sentences)
  attributes = create_attribute_sets(unigrams, kind=args.protocol_type)
  att_1, att_2 = convert_attribute_set(args.attribute_specifications)

  tok2indx, indx2tok = get_indices(unigrams, attributes[att_1], attributes[att_2])
  skipgrams = get_skipgram_counts(sentences, tok2indx, indx2tok)
  coo_mat = create_coo_mat(skipgrams)
  ppmi_mat, sppmi_mat = create_ppmi_mat(coo_mat, skipgrams)

  # Save matrices
  if not os.path.exists('matrices'):
    os.makedirs('matrices')
    
  sparse.save_npz(f'matrices/ppmi_{args.corpus}.npz', ppmi_mat, compressed=True)
  sparse.save_npz(f'matrices/sppmi_{args.corpus}.npz', sppmi_mat, compressed=True)

  # Save token-2-index dictionaries
  if not os.path.exists('tok2indx'):
    os.makedirs('tok2indx')
  with codecs.open(f'tok2indx/{args.corpus}.json',"w", encoding='utf-8') as f:
      f.write(json.dumps(tok2indx))


if __name__ == "__main__":
  main()
