# -*- coding: utf-8 -*-
import sys
import os 
import glob
from gensim.utils import save_as_line_sentence
from text_preprocessing import *
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
lemmatizer = GermanLemmatizer()
# Initialise spelling correction instance 
logging.info('Lemmatizer loaded.')

tpath = os.path.abspath(os.path.join(ROOT_DIR, "data"))
os.chdir(tpath)

class ProcessProtocols(object):
    def __init__(self, dirname):
        self.dirname = dirname
        if not os.path.exists(f'{dirname}_processed'):
            os.makedirs(f'{dirname}_processed')
    def process_and_save(self):
        logging.info('Start processing of files.')
        i = 0
        files_total = len(os.listdir(self.dirname))
        logging.info(f'{files_total} files were found.')
        border = round(files_total / 10)
        for num in range(1,files_total+1):
            if not os.path.isfile(os.path.join(f'{self.dirname}_processed' , f'{num}_sents.txt')):
                try:
                    text = open(os.path.join(self.dirname, f'{num}_sents.txt'),'r', encoding='utf-8').readlines()
                    text = remove_punctuation(text)
                    text = remove_double_spaces(text)
                    text = extract_protocol(text)
                    text = remove_linebreaks(text)
                    text = remove_noisy_digits(text)
                    text = replace_digits(text)
                    text = remove_double_spaces(text)
                    text = reduce_numerical_sequences(text)
                    text = remove_dash_and_minus_signs(text)
                    text = filter_lines(text)
                    text = [removeGermanChainWords(line) for line in text]
                    text = [remove_hyphens_pre_and_appending(line) for line in text]
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
  except IndexError as e: 
    print(e)

ProcessProtocols(dirname).process_and_save()













