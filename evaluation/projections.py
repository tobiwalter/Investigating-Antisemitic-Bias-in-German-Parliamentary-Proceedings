import sys
sys.path.append('./..')
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from plot_utils import set_size
from collections import OrderedDict
from bias_specifications import antisemitic_streams
from utils import load_embedding_dict
from scipy import stats
import argparse
import logging 

# Plotting parameters
TEX_FONTS = {
    # Use LaTeX to write all text
    "text.usetex": False,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}
# Figure height and width
WIDTH = 360
FIG_DIM = set_size(WIDTH)
plt.rcParams.update(TEX_FONTS)

# Word pairs that establish bias subspace
TARGETS_CHRISTIAN = ['christ', 'christlich','christentum']
TARGETS_JEWISH = ['jude', 'juedisch', 'judentum']  

class SubspaceProjections:

  def __init__(self):
      self.embd_dict = None
      self.vocab = None
      self.embedding_matrix = None

  def set_embd_dict(self, embd_dict):
      self.embd_dict = embd_dict

  def convert_by_vocab(self, items, numbers=True):
      """Converts a sequence of [tokens|ids] using the vocab."""
      if numbers:
        output = [self.vocab[item] for item in items if item in self.vocab]
      else:
        output = [item for item in items if item in self.vocab]
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
      self.embedding_matrix = self.mat_normalize(self.embedding_matrix)

      self.embd_dict = None

  def mat_normalize(self,mat, norm_order=2, axis=1):
    return mat / np.transpose([np.linalg.norm(mat, norm_order, axis)])

  def bias_axes(self, semantic_domain, protocol_type):
      """
      WEAT 1 - target terms representing Judaism and Christianity
      """
      targets_1 = TARGETS_CHRISTIAN
      targets_2 = TARGETS_JEWISH
      attributes_1, attributes_2 = antisemitic_streams(semantic_domain, protocol_type)
      return targets_1, targets_2, attributes_1, attributes_2

  def cosine(self, a, b):
    # norm_a = self.mat_normalize(a,axis=0)
    # norm_b = self.mat_normalize(b,axis=0)
    cos = np.dot(a, np.transpose(b))
    # cos = np.dot(norm_a, np.transpose(norm_b))
    return cos

  def doPCA(self, targets_1, targets_2, plot=False):
      matrix = []
      T1 = self.convert_by_vocab(targets_1)
      T2 = self.convert_by_vocab(targets_2)
      for a,b in zip(T1, T2):
          center = (self.embedding_matrix[a] + self.embedding_matrix[b]) /2
          matrix.append(self.embedding_matrix[a] - center)
          matrix.append(self.embedding_matrix[b] - center)
      matrix = np.array(matrix)
      pca = PCA(n_components = round(len(T1)))
      pca.fit(matrix)
      if plot:
        plt.bar(np.arange(pca.n_components_), pca.explained_variance_ratio_)
        plt.show()
      
      race_direction_pca = pca.components_[0]
      return race_direction_pca

  def get_projections(self, attribute, race_direction, slice):
      """Compute subspace projections of attribute terms onto the bias direction"""
      A1 = self.convert_by_vocab(attribute, numbers=False)
      projections = {}
      for word in A1:
          projection = self.cosine(self.embedding_matrix[self.vocab[word]], race_direction) 
          if slice in ('kaiserreich_1', 'weimar'):
              projections[word] = projection * (-1)
              logging.info(f'Projection for {word}: {projection}.')
          else:
              projections[word] = projection
              logging.info(f'Projection for {word}: {projection}.')
      return projections


  def plot_onto_bias_direction(self, projections_1, projections_2, style='ggplot', output_file=''):

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(FIG_DIM))
        # Plot values on the x-axis
        ax.plot(list(projections_1.values()), np.zeros(len(projections_1)), 'co', ms=2.5)
        ax.plot(list(projections_2.values()), np.zeros(len(projections_2)), 'mo', ms=2.5)
        ax.legend(['positive', 'negative'], loc= 'lower right')

        ax.set_xlim(min(min(projections_2.values())-0.05,-0.3),
             max(max(projections_1.values())+0.05,0.3))
        
        # Only annotate every second term
        for i in range(0, len(projections_1),2):
            k,v = list(projections_1.items())[i][0], list(projections_1.items())[i][1]
            ax.annotate(k, xy=(v,0), xytext= (v,0.025), rotation=90, ma='left', fontsize='xx-small', fontstretch='ultra-condensed',
                        arrowprops=dict(facecolor='blue', shrink=0.02, alpha=0.15,width=1, headwidth=2))
                    
        for i in range(0, len(projections_2), 2):
            k,v = list(projections_2.items())[i][0], list(projections_2.items())[i][1]                   
            ax.annotate(k, xy=(v,0), xytext= (v,-0.05), rotation=90, ma='left',  fontsize='xx-small', fontstretch='ultra-condensed',
                        arrowprops= dict(facecolor='blue', shrink=0.02, alpha=0.15,width=1, headwidth=2))
            
        fig.set_size_inches(FIG_DIM[0], (FIG_DIM[0]/2.3))
        ax.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        plt.tight_layout()
        if len(output_file) > 0:
          plt.savefig(f'plots/{output_file}.pdf',dpi=300)
        plt.show()     


  def t_test(self, projections_1, projections_2):
    """Compute t-test between the mean RIPA of two opposing semantic domains"""

    t2, p2 = stats.ttest_ind(projections_1, projections_2, equal_var=False)
    logging.info("t = " + str(t2))
    logging.info("p = " + str(p2))
    return (t2,p2)

def main():
    def boolean_string(s):
      if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
      return s == 'True' or s == 'true'
    parser = argparse.ArgumentParser(description="Compute subspace projections onto ethno-religious bias subspace - plot and/or compute t-test between them")
    parser.add_argument("--protocol_type", nargs='?', choices = ['RT', 'BRD'], help="Whether to run test for Reichstagsprotokolle (RT) or Bundestagsprotokolle (BRD)",required=True)
    parser.add_argument("--sem_domain", nargs='?', choices= ['sentiment', 'patriotism', 'economic', 'conspiratorial', 'racist', 'religious', 'ethic'], help='Which semantic domain to test in WEAT', required=True)
    parser.add_argument("--output_file", type=str, default=None, help="File to store the results)", required=False)
    parser.add_argument("--embedding_vocab", type=str, help="Vocab of the self.embedding_matrix")
    parser.add_argument("--embedding_vectors", type=str, help="Vectors of the self.embedding_matrix")
    parser.add_argument("--slice", type=str, help="The slice to plot")
    parser.add_argument("--plot_projections", type=boolean_string, default=True, help="Whether to plot subspace projections", required=True)
    parser.add_argument("--plot_pca", type=boolean_string, default=False, help="Whether to plot subspace projections", required=True)
    parser.add_argument("--t_test", type=boolean_string, default=True, help="Whether to compute t-test for a semantic domain ", required=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info('Started')
    ripa = SubspaceProjections()

    targets_1, targets_2, attributes_1, attributes_2 = ripa.bias_axes(args.sem_domain, args.protocol_type)

    embd_dict = load_embedding_dict(vocab_path=args.embedding_vocab, vector_path=args.embedding_vectors, glove=False)
    ripa.set_embd_dict(embd_dict)
    vocab = targets_1 + targets_2 + attributes_1 + attributes_2
    ripa._build_vocab_dict(vocab)
    ripa._build_embedding_matrix()
    # Don't forget to normalize vectors!!!

    race_direction_pca = ripa.doPCA(targets_1,targets_2, plot=args.plot_pca)
    
    projections_pro= ripa.get_projections(attributes_1, race_direction_pca, args.slice)
    projections_con= ripa.get_projections(attributes_2, race_direction_pca, args.slice)

    if args.plot_projections:
      with plt.style.context('ggplot'):
        logging.info(f'Plot projections for semantic sphere {args.sem_domain}')
        ripa.plot_onto_bias_direction(projections_pro, projections_con, output_file=args.output_file)
    if args.t_test:
      if not os.path.exists('t_test'):
        os.makedirs('t_test')
      logging.info(f'Conduct t-test for semantic sphere {args.sem_domain}')
      t,p = ripa.t_test(list(projections_pro.values()), list(projections_con.values()))
      with codecs.open(f't_test/{args.slice}_{args.semantic_domain}.txt', "w", "utf8") as f:
        f.write(f'test statistic: {t}, ')
        f.write(f'p-value: {p}\n')
        f.close()

if __name__ == "__main__":
  main()






