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

logging.info('Lemmatizer loaded.')

tpath = os.path.abspath(os.path.join(ROOT_DIR, "data"))
os.chdir(tpath)

class ProcessProtocols(object):
    def __init__(self, input)
        self.input = input

    def process_and_save(self):
        logging.info('Start processing of file.')
        try:
             text = open(os.path.join(tpath, self.input),'r', encoding='utf-8').readlines()
             text = remove_punctuation(text)
             text = remove_double_spaces(text)
             text = remove_noisy_digits(text)
             text = replace_digits(text)
             text = remove_double_spaces(text)
             text = reduce_numerical_sequences(text)
             text = remove_dash_and_minus_signs(text)
             text = filter_lines(text)
             text = [removeGermanChainWords(line) for line in text]
             logging.info('Chainword splitting finished')
             text = [remove_hyphens_pre_and_appending(line) for line in text]
             text = [lemmatizer.lemmatize(line) for line in text]
             logging.info('Lemmatizing finished')
             text = [lowercase(line) for line in text]
             text = [removeUmlauts(line) for line in text]
             text = [harmonizeSpelling(line) for line in text]
             if self.input.endswith('.txt'):
                save_as_line_sentence(text, f'{self.input[:-4]}_processed.txt')
             else:
                save_as_line_sentence(text, f'{self.input}_processed.txt')
             logging.info('Processing finished')

        except FileNotFoundError:
            print(f'File was not found.')

if __name__ == "__main__":
  try:
    filename = sys.argv[1]
  except IndexError as e: 
    print(e)

ProcessProtocols(filename).process_and_save()


