import sys
import os 
from text_preprocessing import remove_punctuation, remove_double_spaces, remove_noisy_digits, remove_dash_and_minus_signs, extract_meeting_protocols_reichstag
import logging
import codecs

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
tpath = os.path.abspath(os.path.join(ROOT_DIR, "data"))
os.chdir(tpath)

text = codecs.open(os.path.join(tpath, sys.argv[1]),'r', encoding='utf-8').readlines()
# Do some minor text cleaning before extraction of the protocols 
text = remove_punctuation(text)
text = remove_double_spaces(text)
text = remove_noisy_digits(text)
text = remove_dash_and_minus_signs(text)

if __name__ == "__main__":
	extract_meeting_protocols_reichstag(text,sys.argv[1][:4])
		
