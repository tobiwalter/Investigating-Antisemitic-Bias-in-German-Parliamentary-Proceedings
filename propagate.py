from representations.utils import create_target_sets, create_attribute_sets, inverse, convert_attribute_set
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
  def __init__(self, ppmi_mat, tok2indx):
    self.ppmi = ppmi_mat
    self.index = tok2indx
    self.labels = None
    self.scores = None
  @classmethod
  def load(cls, matrix_path, indx_path):
    ppmi_mat = sparse.load_npz(matrix_path)
    indx2tok = json.load(open(indx_path, 'r', encoding='utf-8'))
    return cls(ppmi_mat, indx2tok)

  def create_labels(self, targets_1, targets_2):
    labels = []
    for word in targets_1:
        labels.append(1)
    for word in targets_2:
        labels.append(0)
    labels = np.array(labels)
    self.labels = labels

  def propagate(self):
    # Compute normalized laplacian 
    L = sparse.csgraph.laplacian(self.ppmi, normed=True).toarray()
    
    # Sub-matrices L_ul and L_uu
    L_ul = L[len(self.labels):, :len(self.labels)]
    L_uu = L[len(self.labels):, len(self.labels):]
    
    # Compute scores f_u (add a little bit if noise to L_uu to avoid LinAlgError when computing inverse)
    self.scores = (-1* inverse(L_uu + np.eye(L_uu.shape[0])*M)).dot(L_ul).dot(self.labels)
    # np.save(f'f_u_scores/{label}', f_u)

  def save_scores(self, path):
    if not os.path.exists('fu_scores'):
      os.makedirs('fu_scores')
    np.save(f'fu_scores/{path}', self.scores)

  def get_bias_term_indices(self, targets):
     return {k: [self.index[word] - len(self.labels) for word in v] for k,v in targets.items()}

  def get_bias_term_scores(self, bias_term_indices):
    return {k: self.scores[v] for k,v in bias_term_indices.items()}

  # def output_stats(self):
  #   with codecs.open('')

  # def score_df(targets, scores):
  #   return pd.DataFrame(index = np.concatenate([v for v in targets.values()]),
  #                       data = np.concatenate([v for v in scores.values()]),
  #                       columns = ['position_score'])

  def t_test(self, targets_1, targets_2):
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
  parser.add_argument("--protocol_type", type=str, help="Run tests for Reichstagsprotokolle or Bundestagsprotokolle?", required=True)
  parser.add_argument("--attribute_specifications", type=str, help='Which attribute set to be used for label propagation - either sentiment, patriotism, economic or conspiratorial')
  parser.add_argument("--output_file", type=str, help='Path to output file for label propagation scores of bias term indices')

  args = parser.parse_args()
  attributes = create_attribute_sets(args.index, kind=args.protocol_type)
  att_1, att_2 = convert_attribute_set(args.attribute_specifications)
  
  lp = LabelPropagation.load(args.ppmi, args.index)
  lp.create_labels(attributes[att_1], attributes[att_2])

  start = time.time()
  logging.info(f'Start label propagation at {start}')
 # lp.propagate()
  lp.load_scores('fu_scores/kaiserreich_1.npy')
 # lp.save_scores(args.output_file)
  elapsed = time.time()
  logging.info(f'Label propagation finished. Took {(elapsed - start) / 60} min.')
  targets = create_target_sets(lp.index, kind=args.protocol_type)
  bias_term_indices = lp.get_bias_term_indices(targets)
  bias_term_scores = lp.get_bias_term_scores(bias_term_indices)

  with codecs.open(f'{args.output_file}.txt', "w", "utf8") as f:
    for k,v in bias_term_scores.items():
      f.write(f'Mean score {k}: {v.mean()}')
      f.write(f'Median score {k}: {np.percentile(v, 50)}')
      f.write(f'Std {k}: {v.std()}\n')

    f.write('T-tests:\n')
    for t1,t2 in [('christian', 'jewish'), ('protestant', 'catholic'),
    ('protestant', 'jewish'), ('catholic', 'jewish')]:
      logging.info(f't-test for {t1} and {t2}:\n')
      t,p = lp.t_test(bias_term_indices[t1], bias_term_indices[t2])
      logging.info(t,p)
      f.write(f't-test for {t1} and {t2}:\n')
      f.write(f'test statistic: {t}, ')
      f.write(f'p-value: {p}\n')


if __name__ == "__main__":
  main()



