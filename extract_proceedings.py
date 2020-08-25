import sys
import os 
import codecs
from text_preprocessing import remove_punctuation, remove_double_spaces, remove_noisy_digits, remove_dash_and_minus_signs, reichstag_patterns, extract_meeting_protocols_reichstag
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
tpath = os.path.abspath(os.path.join(ROOT_DIR, "data"))
os.chdir(tpath)

if __name__ == "__main__":
	text = codecs.open(os.path.join(tpath, sys.argv[1]),'r', encoding='utf-8').readlines()
	# Do some minor text cleaning before protocols are extracted  
	text = remove_punctuation(text)
	text = remove_double_spaces(text)
	text = remove_noisy_digits(text)
	text = remove_dash_and_minus_signs(text)


	# Extract protocols and save them in ./data folder
	patterns = reichstag_patterns()
	extract_meeting_protocols_reichstag(text, *patterns, sys.argv[1][:4])
		
