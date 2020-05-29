# -*- coding: utf-8 -*-
import glob
import re
import os
from datetime import datetime
import locale
from representations.utils import save_corpus
import dateparser

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
tpath = os.path.abspath(os.path.join(ROOT_DIR, "data"))
os.chdir(tpath)

# Process the protocols and create corpus
#locale.setlocale(locale.LC_ALL, 'de_DE')
FIRST_DATES = {
    1: '22 Oktober 1889',
    2: '3 Dezember 1895',
    3: '6 Februar 1919',
    4: '21 März 1933'
}

PERIODS = {
    1: {'start' : datetime(1867,2,24), 'end' : datetime(1895,5,24)},
    2: {'start' : datetime(1895,12,3), 'end' : datetime(1918,10,26)},
    3: {'start' : datetime(1919,2,6), 'end' : datetime(1932,12,9)},
    4: {'start' : datetime(1933,3,21), 'end' : datetime(1942,4,26)},    
}

def sort_corpus(number):
    '''Checking if I can extract the dates of the sessions in order to use them as sorting parameter'''
    corpus_and_dates = []
    
    # We know the first date by manually looking into the corpus
    last_date = dateparser.parse(FIRST_DATES.get(number))
    period_start = PERIODS.get(number).get('start')
    period_end = PERIODS.get(number).get('end')
    
    # Retrieve files in chronological order they were created
#     for file in sorted(glob.glob(os.path.join(os.path.basename(top_dir), 'doc_*.txt')),key=os.path.getmtime):
    for file in sorted(glob.glob('./protocols_{}/doc_*.txt'.format(number)), key=os.path.getmtime):
        text = open(file, 'r', encoding='utf-8').readlines()
        text = [line.strip() for line in text]
        match = re.search(r'\d+ Sitzung (?:am )?(?:Montag |Dienstag |Mittwoch |Donnerstag |Freitag |Sonnabend |Sonntag )?(?:den )?(\d+ (?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember) \d{4})', " ".join(text))
        if match:
            date = match.groups()[0]
            date = dateparser.parse(date)
            # date = datetime.strptime(date,'%d %B %Y')
            # out of period range due to OCR error
            if date is not None:
                if date < period_start or date > period_end:
                    date = last_date
            else:
            # if there is a value error (e.g. as an OCR error -> 38 Januar instead of 28 Januar), impute with last date
                date = last_date
                
            # Store the last valid date to impute instances for which no date was found
            last_date = date
        else:
            date = last_date
            
        corpus_and_dates.append((date,text))
    
    # Return corpus sorted by date
    sorted_corpus = sorted(corpus_and_dates, key=lambda x: x[0])
    # Return only list 
    sorted_corpus = [sublist[1] for sublist in sorted_corpus]
    return sorted_corpus

def main():

	# Sort all original Reichstag corpora and create balanced slices in terms of number of protocols per slice
	corpus_1 = sort_corpus(1)
	corpus_2 = sort_corpus(2)
	corpus_3 = sort_corpus(3)
	corpus_4 = sort_corpus(4)

	full_corpus = corpus_1 + corpus_2 + corpus_3 + corpus_4

	slice_border = round(len(full_corpus) / 5)
	print(slice_border)

	slice_1 = full_corpus[:slice_border]
	slice_2 = full_corpus[slice_border:(2*slice_border)]
	slice_3 = full_corpus[(2*slice_border):(3*slice_border)]
	slice_4 = full_corpus[(3*slice_border):(4*slice_border)]
	slice_5 = full_corpus[(4*slice_border):]

	save_corpus(slice_1, 'rt_slice_1')
	save_corpus(slice_2, 'rt_slice_2')
	save_corpus(slice_3, 'rt_slice_3')
	save_corpus(slice_4, 'rt_slice_4')
	save_corpus(slice_5, 'rt_slice_5')

if __name__ == "__main__":
	main()

