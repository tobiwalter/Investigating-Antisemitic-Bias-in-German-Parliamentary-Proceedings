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

tpath = os.path.abspath(os.path.join(ROOT_DIR, "data"))
# sys.path.append(tpath)
os.chdir(tpath)

# Initialise lemmatizer
lemmatizer = GermanLemmatizer()
# Initialise spelling correction instance 
spell_checker = GermanSpellChecker('dictionaries/de_full.txt')

logging.info('Lemmatizer and spell checker loaded.')

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
        i = 0
        files_total = len(os.listdir(self.dirname))
        border = round(files_total / 10, 2)
        logging.info('Start processing of files.')
        for file in os.listdir(self.dirname):
            file_name = os.path.splitext(os.path.basename(file))[0]
            text = open(os.path.join(self.dirname, file), encoding='utf-8').readlines()
            text = extract_protocol(text)
            text = remove_linebreaks(text)
            text = remove_punctuation(text)
            text = remove_double_spaces(text)
            text = remove_noisy_digits(text)
            text = replace_digits(text)
            text = reduce_numerical_sequences(text)
            text = remove_dash_and_minus_signs(text)
            text = remove_double_spaces(text)
            text = remove_empty_lines(text)
            text = [removeGermanChainWords(line) for line in text]
            text = [expandCompoundToken(line) for line in text]
            text = [lemmatizer.lemmatize(line) for line in text]
            text = [spell_checker.correct(line) for line in text]
            text = [[tok.lower() for tok in line] for line in text]
  #             text = itertools.chain.from_iterable(text)
            save_as_line_sentence(text, f'{self.dirname}_processed/{file_name}.txt')
            i += 1
            if i % border == 0:
              logging.info('Processing {:03.1f} percent finished'.format(float((i/(files_total)) * 100)))


if __name__ == "__main__":
  try:
    dirname = sys.argv[1]
  except IndexError as e: 
    print(e)

ProcessProtocols(dirname).process_and_save()













