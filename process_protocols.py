# -*- coding: utf-8 -*-
import sys
import os
from gensim.utils import save_as_line_sentence
from text_preprocessing import *
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

lemmatizer = GermanLemmatizer()
logging.info('Lemmatizer loaded.')

tpath = os.path.abspath(os.path.join(ROOT_DIR, "data"))
os.chdir(tpath)

class ProcessProtocols(object):
    """
    Apply text preprocessing pipeline on a folder of protocols
    """

    def __init__(self, dirname, brd=False):
        """
        param dirname: name of directory to process
        param brd: whether to process an instance of BRD protocols
        """
        self.dirname = dirname
        self.brd = brd
        if not os.path.exists(f'{dirname}_processed'):  
            os.makedirs(f'{dirname}_processed')
    def process_and_save(self):
        logging.info(f'Start processing of files from folder {dirname}')
        if self.brd:
            logging.info(f'Processing protocols from BRD period')
        else:
            logging.info(f'Processing protocols from Reichstag period')
        i = 0
        files_total = len(os.listdir(self.dirname))
        logging.info(f'{files_total} files were found.')
        border = round(files_total / 10)
        for num in range(1,files_total+1):
            if not os.path.isfile(os.path.join(f'{self.dirname}_processed' , f'{num}_sents.txt')):
                try:
                    text = codecs.open(os.path.join(self.dirname, f'{num}_sents.txt'),'r', encoding='utf-8').readlines()
                    if self.kind == 'BRD':
                        text = remove_punctuation(text)
                        text = remove_double_spaces(text)
                        text = extract_protocol(text)
                    text = remove_linebreaks(text)
                    if self.kind == 'BRD':
                        text = remove_noisy_digits(text)
                        text = remove_dash_and_minus_signs(text)
                    text = replace_digits(text)
                    text = remove_double_spaces(text)
                    text = reduce_numerical_sequences(text)
                    text = filter_lines(text)
                    text = [removeGermanChainWords(line) for line in text]
                    text = [remove_hyphens(line) for line in text]
                    text = [lemmatizer.lemmatize(line) for line in text]
                    text = [lowercase(line) for line in text]
                    text = [removeUmlauts(line) for line in text]
                    text = [harmonizeSpelling(line) for line in text]
                    save_as_line_sentence(text, f'{self.dirname}_processed/{num}_sents.txt')
                    i += 1
                    if i % border == 0:
                      logging.info('Processing {:03.1f} percent finished'.format(int((i/files_total) * 100)))

                except FileNotFoundError:
                    print(f'File {num} was not found.')

if __name__ == "__main__":
  try:
    dirname = sys.argv[1]
    kind = sys.argv[2]
  except IndexError as e: 
    print(e)
  
  ProcessProtocols(dirname, kind).process_and_save()













