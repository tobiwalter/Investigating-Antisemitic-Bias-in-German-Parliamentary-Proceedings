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
spell_checker = GermanSpellChecker('dictionaries/de_full.txt')
logging.info('Lemmatizer and spell checker loaded.')

# Add and delete certain dictionary entries 
with open('dictionaries/manual_additions.txt', 'r', encoding='utf-8') as f:
    words_to_add = [word.strip() for word in f.readlines()]
with open('dictionaries/manual_deletions.txt', 'r', encoding='utf-8') as f:
    words_to_delete = [word.strip() for word in f.readlines()]
    
spell_checker.add_entries(words_to_add)
spell_checker.delete_entries(words_to_delete)

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
        for file in os.listdir(self.dirname):
            if not os.path.isfile(os.path.join(f'{self.dirname}_processed' , file)):
                 try:
                     text = open(os.path.join(self.dirname, file),'r', encoding='utf-8').readlines()
                     text = remove_punctuation(text)
                     text = remove_double_spaces(text)
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
                     text = [spell_checker.correct(line) for line in text]

                # with open(os.path.join(f'{self.dirname}_processed' , f'{num}_sents.txt'), 'w', encoding='utf-8') as out_file:
                #     for line in text:
                #         out_file.write(line)
                #         out_file.write('\n')
                     save_as_line_sentence(text, f'{self.dirname}_processed/{file}')
                     i += 1
                     logging.info('Processing {:03.1f} percent finished'.format(int((i/files_total) * 100)))
    
                 except FileNotFoundError:
                    print(f'File was not found.')

if __name__ == "__main__":
  try:
    dirname = sys.argv[1]
  except IndexError as e: 
    print(e)

ProcessProtocols(dirname).process_and_save()


