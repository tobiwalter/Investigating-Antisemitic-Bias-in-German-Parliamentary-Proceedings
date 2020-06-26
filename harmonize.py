# -*- coding: utf-8 -*-
import sys
import os 
import glob
from gensim.utils import save_as_line_sentence
from text_preprocessing import *
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

tpath = os.path.abspath(os.path.join(ROOT_DIR, "data"))
os.chdir(tpath)


def harmonize(dirname):
    if not os.path.exists(f'{dirname}_harmonized'):
        os.makedirs(f'{dirname}_harmonized')
    logging.info('Start processing of files.')
    i = 0
    files_total = len(os.listdir(dirname))
    logging.info(f'{files_total} files were found.')
    border = round(files_total / 10)
    for num in range(1,files_total+1):
        try:
            text = open(os.path.join(dirname, f'{num}_sents.txt'),'r', encoding='utf-8').readlines()
            text = [line.split() for line in text]
            text = [harmonizeSpelling(line) for line in text]
            save_as_line_sentence(text, f'{dirname}_harmonized/{num}_sents.txt')
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

harmonize(dirname)
