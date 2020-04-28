# -*- coding: utf-8 -*-
import sys
import os 
from gensim.utils import save_as_line_sentence
from text_preprocessing import *
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
# from text_preprocessing import lemmatizer_plus

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.abspath(os.path.join(ROOT_DIR,  'CharSplit')))
# import char_split
lemmatizer = GermanLemmatizer()
# Initialise spelling correction instance 
print(ROOT_DIR)
spell_checker = GermanSpellChecker('dictionaries/de_full.txt')
logging.info('Lemmatizer and spell checker loaded.')

tpath = os.path.abspath(os.path.join(ROOT_DIR, "data"))
os.chdir(tpath)

# Add and delete certain dictionary entries 
# TO-DO: load words from txt-list
words_to_add = 'xxen, §, volkswirtschaftlich, anteilscheine, er_es, oberkirchenrat, lohnklasse, privathand, hinterbliebenenversicherung, \
invalidenzahl, privatangestellten, invalidengesetz, tanach, zionistisch, oberkirchenrat, lutherisch, evangelisch-lutherisch, jesuitenorden, \
, reichsfeind, undeutsch, antinational, antideutsch, überfremdung, deutschnationalen, viehfutter, inlandsgetreide, volksvermögen, \
wirtschaftsinteressen, weingesetz, königstreu, landesausschuffes, beschneidung, landesfinanzamt, württembergisch, friedensvertrag, \
reichsbank, reichspostverwaltung, friedensvertrag, regierungsbezirks, industriematerial, inlandswein'.split(', ')
words_to_delete = 'roth, sabbat, volksthums, volksthum'.split(', ')

spell_checker.add_entries(words_to_add)
spell_checker.delete_entries(words_to_delete)

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
            if not os.path.exists(os.join(self.dirname, f'{num}_sents.txt')):
                try:
                    text = open(os.path.join(self.dirname, f'{num}_sents.txt'), encoding='utf-8').readlines()
                    text = remove_punctuation(text)
                    text = remove_double_spaces(text)
                    text = extract_protocol(text)
                    with open('f{self.dirname}_processed/{num}_sents.txt', 'w') as out_file:
                        for line in text:
                            f.write(line)
                            f.write('\n')

                    # save_as_line_sentence(text, f'{self.dirname}_processed/{num}_sents.txt')
                    i += 1
                    if i % border == 0:
                      logging.info('Processing {:03.1f} percent finished'.format(int((i/(files_total)) * 100)))

                except FileNotFoundError:
                    print(f'File {num} was not found.')

if __name__ == "__main__":
  try:
    dirname = sys.argv[1]
  except IndexError as e: 
    print(e)

ProcessProtocols(dirname).process_and_save()













