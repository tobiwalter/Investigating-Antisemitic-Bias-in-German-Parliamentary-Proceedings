from utils import create_target_sets, create_attribute_sets, inverse, convert_attribute_set
from scipy import sparse, stats
import os
import numpy as np
import codecs
import argparse
import logging
import time
import json
logging.basicConfig(level=logging.INFO)

M = 1e-06
class LabelPropagation:
  """
  Performs harmonic function label propagation on a graph

  Attributes
  ----------
  ppmi (scipy.sparse.csr.csr_matrix): PPMI matrix/graph
  index (dict): word2tok index for graph
  attributes (dict): attribute sets of the graph 
  """

  def __init__(self, ppmi_mat, tok2indx, protocol_type):
    self.ppmi = ppmi_mat
    self.index = tok2indx
    self.attributes = create_attribute_sets(self.index, kind=protocol_type)
    self.labels = None
    self.scores = None

  @classmethod
  def load(cls, matrix_path, indx_path, **kwargs):
    ppmi_mat = sparse.load_npz(matrix_path)
    logging.info(f'Shape of PPMI matrix:{ppmi_mat.shape}')
    tok2indx = json.load(open(indx_path, 'r', encoding='utf-8'))
    assert len(tok2indx) == ppmi_mat.shape[0]
    return cls(ppmi_mat, tok2indx, **kwargs)

  def reindex(self, semantic_domain, random=False):
    """
    Reindex the PPMI matrix to put the attributes on top that shall be used as labelled nodes to propagate from

    param semantic_domain: Which semantic domain to propagate from
    random: Whether to use random indices for reindexing (for testing purposes)
    """

    pos_att, neg_att = f'{semantic_domain}_pro', f'{semantic_domain}_con'

    if random:
      pos_indices_to_swap = np.random.randint(0,len(self.index),len(self.attributes[pos_att]))
      neg_indices_to_swap = np.random.randint(0,len(self.index),len(self.attributes[neg_att]))
    else:
      pos_indices_to_swap = [self.index[word] for word in self.attributes[pos_att]]
      neg_indices_to_swap = [self.index[word] for word in self.attributes[neg_att]]

    matrix = self.ppmi.toarray()
    i2t = {v:k for k,v in self.index.items()}
    # Switch matrix rows
    for indx_old, indx_new in enumerate(pos_indices_to_swap):
        matrix[[indx_old, indx_new]] = matrix[[indx_new, indx_old]]
        matrix[:,[indx_old, indx_new]] = matrix[:,[indx_new, indx_old]]
        i2t[indx_old], i2t[indx_new] = i2t[indx_new], i2t[indx_old]
    # Switch matrix columns
    for indx_old, indx_new in enumerate(neg_indices_to_swap):
        matrix[[indx_old+len(pos_indices_to_swap), indx_new]] = matrix[[indx_new, indx_old+len(pos_indices_to_swap)]]
        matrix[:,[indx_old+len(pos_indices_to_swap), indx_new]] = matrix[:,[indx_new, indx_old+len(pos_indices_to_swap)]]
        # Switch index
        i2t[indx_old+len(pos_indices_to_swap)], i2t[indx_new] = i2t[indx_new], i2t[indx_old+len(pos_indices_to_swap)]
    
    # Adapt index
    self.index = {v:k for k,v in i2t.items()}
    print([(f'{word}:{self.index[word]}') for word in self.attributes[pos_att]])
    print([(f'{word}:{self.index[word]}') for word in self.attributes[neg_att]])
    self.ppmi = sparse.csr_matrix(matrix)

  def create_labels(self, targets_1, targets_2):
    """Create pre-assigned y-labels for labelled nodes"""
    labels = []
    for word in targets_1:
        labels.append(1)
    for word in targets_2:
        labels.append(0)
    labels = np.array([labels, list(reversed(labels))]).T
    self.labels = labels

  def propagate(self):
    """Compute harmonic function scores based on Laplacian matrix operations"""

    L = sparse.csgraph.laplacian(self.ppmi, normed=True).toarray()
    # Sub-matrices L_ul and L_uu
    L_ul = L[len(self.labels):, :len(self.labels)]
    L_uu = L[len(self.labels):, len(self.labels):]
    
    # Compute scores f_u (add a little bit if noise to L_uu to avoid LinAlgError when computing inverse)
    try:
      self.scores = fu = np.multiply(-1.0, np.linalg.inv(L_uu)).dot(L_ul).dot(self.labels)
    except:
      logging.info(f'Add little amount of noise ({M}) to enable computation of inverse')
      self.scores = fu = np.multiply(-1.0, np.linalg.inv(L_uu + np.eye(L_uu.shape[0])*M)).dot(L_ul).dot(self.labels)

  def save_scores(self, path, semantic_domain):
    """Save harmonic function scores"""

    if not os.path.exists('fu_scores'):
      os.makedirs('fu_scores')
    np.save(f'fu_scores/{path}_{semantic_domain}.npy', self.scores)

  def get_bias_term_indices(self, targets):
     return {k: [self.index[word] - len(self.labels) for word in v] for k,v in targets.items()}

  def get_bias_term_scores(self, bias_term_indices):
    """Retrieve the scores belonging to target terms"""

    self.scores = self.scores.T[0]
    return {k: self.scores[v] for k,v in bias_term_indices.items()}

  # def output_stats(self):
  #   with codecs.open('')

  # def score_df(targets, scores):
  #   return pd.DataFrame(index = np.concatenate([v for v in targets.values()]),
  #                       data = np.concatenate([v for v in scores.values()]),
  #                       columns = ['position_score'])

  def t_test(self, targets_1, targets_2):
    """Compute t-test between the harmonic function scores of two target sets"""

    t2, p2 = stats.ttest_ind(self.scores[targets_1], self.scores[targets_2], equal_var=True)
    logging.info("t = " + str(t2))
    logging.info("p = " + str(p2))
    return (t2,p2)

  def load_scores(self, path):
    self.scores = np.load(path)

def main():

  parser = argparse.ArgumentParser(description="Propagate labels based on PPMI matrix")
  parser.add_argument("--ppmi", type=str, help="Path to PPMI matrix to be used for label propagation", required=True)
  parser.add_argument("--index", type=str, help="Path to token-2-index dictionary to be used for label propagation", required=True)
  parser.add_argument("--protocol_type", "-pt", nargs='?', choices = ['RT', 'BRD'], help="Whether to run test for Reichstagsprotokolle (RT) or Bundestagsprotokolle (BRD)", required=True)
  parser.add_argument("--semantic_domain", "-sem", type=str, help='Which attribute set to be used for label propagation - either sentiment, patriotism, economic or conspiratorial')
  parser.add_argument("--random", action='store_true')
  parser.add_argument("--output_file", type=str, help='Path to output file for label propagation scores of bias term indices')

  args = parser.parse_args()
  lp = LabelPropagation.load(args.ppmi, args.index, protocol_type=args.protocol_type)
  att_1, att_2 = f'{args.semantic_domain}_pro', f'{args.semantic_domain}_con'
  lp.create_labels(lp.attributes[att_1], lp.attributes[att_2])
  print(lp.labels)

  if args.semantic_domain != 'sentiment' or args.random:
    logging.info('Reindex matrix')
    lp.reindex(args.semantic_domain, random=args.random)
  targets = create_target_sets(lp.index, kind=args.protocol_type)
  bias_term_indices = lp.get_bias_term_indices(targets)
  print(bias_term_indices)
  
  start = time.time()
  if args.random:
    logging.info(f'Start label propagation for random attribute sets')
  else:
    logging.info(f'Start label propagation for attributes {att_1} and {att_2}')
  lp.propagate()
  #lp.load_scores('fu_scores/kaiserreich_1.npy')
  lp.save_scores(args.output_file, args.semantic_domain)
  elapsed = time.time()
  logging.info(f'Label propagation finished. Took {(elapsed - start) / 60} min.')
  bias_term_scores = lp.get_bias_term_scores(bias_term_indices)

  with codecs.open(f'fu_scores/{args.output_file}_{args.semantic_domain}.txt', "w", "utf8") as f:
    for k,v in bias_term_scores.items():
      f.write(f'Mean score {k}: {v.mean()}\n')
      f.write(f'Median score {k}: {np.percentile(v, 50)}\n')
      f.write(f'Std {k}: {v.std()}\n\n')

    f.write('T-tests:\n')
    for t1,t2 in [('christian', 'jewish'), ('protestant', 'catholic'),
    ('protestant', 'jewish'), ('catholic', 'jewish')]:
      logging.info(f't-test for {t1} and {t2}:\n')
      t,p = lp.t_test(bias_term_indices[t1], bias_term_indices[t2])
      f.write(f't-test for {t1} and {t2}:\n')
      f.write(f'test statistic: {t}, ')
      f.write(f'p-value: {p}\n')
    f.close()

if __name__ == "__main__":
  main()



