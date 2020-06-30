# -*- coding: utf-8 -*-
import glob
import re
import os
from datetime import datetime
import dateparser
from representations.utils import save_corpus

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
tpath = os.path.abspath(os.path.join(ROOT_DIR, "data"))
os.chdir(tpath)

# Process the protocols and create corpus

# These are the dates of the first protocol in each of the 4 original "YYYY.corr.seg" Reichstag documents - as oen can see, the first document is not in the right order
START_DATES = {
    1895: '22 Oktober 1889',
    1918: '3 Dezember 1895',
    1933: '6 Februar 1919',
    1942: '21 März 1933'
}

# These are the actual periods for each of the 4 original "YYYY.corr.seg" Reichstag documents with start and end dates 

PERIODS = {
    1895: {'start' : datetime(1867,2,24), 'end' : datetime(1895,5,24)},
    1918: {'start' : datetime(1895,12,3), 'end' : datetime(1918,10,26)},
    1933: {'start' : datetime(1919,2,6), 'end' : datetime(1932,12,9)},
    1942: {'start' : datetime(1933,3,21), 'end' : datetime(1942,4,26)},    
}

def sort_corpus(year):
    """As at least the first document is not in the right order, let's create a function that sorts each corpus

    param year: which of the 4 original corpora shall be sorted
    """

    # Initialize a list that stores each protocol with its corresponding date 
    corpus_and_dates = []
    
    # The variable last_date keeps track of the last valid date encountered by the function in order to keep a chronological ordering - it is initialized with the start date in each corpus  
    last_date = dateparser.parse(START_DATES.get(year))
    period_start = PERIODS.get(year).get('start')
    period_end = PERIODS.get(year).get('end')
    
    # Retrieve files in chronological order of creation
    for file in sorted(glob.glob(f'./protocols_{year}/doc_*.txt'), key=os.path.getmtime):
        text = codecs.open(file, 'r', encoding='utf-8').readlines()
        text = [line.strip() for line in text]
        # For each protocol, check if a date can be extracted
        date_match = re.search(r'\d+ Sitzung (?:am )?(?:Montag |Dienstag |Mittwoch |Donnerstag |Freitag |Sonnabend |Sonntag )?(?:den )?(\d+ (?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember) \d{4})', " ".join(text))

        # If match found, assign it to variable 
        if date_match:
            date = date_match.groups()[0]
            date = dateparser.parse(date)
            # date = datetime.strptime(date,'%d %B %Y')

            if date is not None:
                # if date out of period range due to OCR error, impute date with last valid date
                if date < period_start or date > period_end:
                    date = last_date
            else:
            # if date is None(e.g. due to an OCR error -> 38 Januar instead of 28 Januar), impute date with last valid date
                date = last_date
                
            # Store the last valid date to impute instances for which no or a faulty date was found
            last_date = date

        # No date match at all, impute with last valid date
        else:
            date = last_date
        
        # Store protocol in the list along with date assigned to it     
        corpus_and_dates.append((date,text))
    
    # Sort the corpus by date 
    sorted_corpus = sorted(corpus_and_dates, key=lambda x: x[0])
    # Discard the stored dates, return only protocols
    sorted_corpus = [sublist[1] for sublist in sorted_corpus]
    return sorted_corpus

def main():

	# Sort all original Reichstag corpora and create balanced slices in terms of year of protocols per slice
	corpus_1 = sort_corpus(1)
	corpus_2 = sort_corpus(2)
	corpus_3 = sort_corpus(3)
	corpus_4 = sort_corpus(4)

	full_corpus = corpus_1 + corpus_2 + corpus_3 + corpus_4

    # TO-DO: implement condition that creates either balanced or historical slices 

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

