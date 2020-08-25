# -*- coding: utf-8 -*-
import sys
import os
import re
import spacy
import codecs
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.abspath(os.path.join(ROOT_DIR,  'CharSplit')))
import char_split

sys.path.append(os.path.abspath(os.path.join(ROOT_DIR, 'germalemmaplusplus')))
from germalemmaplusplus.lemmatizer import GermaLemmaPlusPlus, Token, Sentence

# Text processing functions

def remove_linebreaks(doc: List) -> List:
    return [re.sub(r'[\n\t]', '', line).strip() for line in doc]

def remove_punctuation(doc: List) -> List:
    pattern = re.compile('[%s]' % re.escape('!"#$&\'()*+,./:;<=>?@®©[\\]^_`{|}~„“«»'))
    return [re.sub(pattern,'', line).strip() for line in doc]

def remove_double_spaces(doc: List) -> List:
    return [re.sub('\s\s+',' ', line).strip() for line in doc]

def remove_noisy_digits(doc: List) -> List:
    """ Remove digits that are appending, prepending or in the middle of some words, e.g. because of faulty OCR """

    temp = [re.sub(r'\b\d(?P<quote>[A-Za-zßäöüÄÖÜ]+)\b', '\g<quote>', line) for line in doc]
    temp = [re.sub(r'\b(?P<quote>[A-Za-zßäöüÄÖÜ]+)\d\b', '\g<quote>', line) for line in temp] 
    temp = [re.sub(r'\b(?P<quote>[A-Za-zßäöüÄÖÜ]+)\d(?P<name>[A-Za-zßäöüÄÖÜ]*)\b', '\g<quote> \g<name>', line) for line in temp]
    return temp

def replace_digits(doc: List) -> List:
    return [re.sub(r'\d+',' 0 ' ,line).strip() for line in doc]

def reduce_numerical_sequences(doc: List) -> List:
    """If there are sequences of digits occuring, reduce to a single placeholder."""

    return [re.sub(r'((0)\s?){2,}', '\\2 ', line).strip() for line in doc]

def filter_doc(doc: List) -> List:
    """Filter all lines that are empty or only a single character long."""

    return [line for line in doc if len(line) >1]

def lowercase(sentence: str) -> str:
    return [tok.lower() for tok in sentence]

def remove_dash_and_minus_signs(doc: List) -> List:
    """Remove lone standing dashes/hyphens at beginning, middle and end of lines."""

    temp = [re.sub(r'\s(-|—|–|\s)+\s', ' ', line) for line in doc]
    temp = [re.sub(r'^(-|—|–|\s)+\s', '', line) for line in temp]
    temp = [re.sub(r'\b\s(-|—|–|\s)+$', '', line) for line in temp]
    return temp

def char_splitting(i: int, groups: List, chainword="und") -> str:
    """Helper function for remove_german_chainwords."""
    
    word1 = groups[0].replace(" ","")
    word2 = groups[1].replace(" ","")

    if len(groups) >= 4:
        word3 = str(groups[2]).replace(" ","")
    
    if i == 0:
        splitted = char_split.split_compound(word3)[0][-1].lower()
        return " {}{} {} {}{} {} {}".format(word1,splitted,chainword,word2,splitted,chainword,word3)
    if i == 1:
        splitted = char_split.split_compound(word2)[0][-1].lower()
        return " {}{} {} {}".format(word1,splitted,chainword,word2)
    if i == 2:
        # if both gendered versions are named, use the male version for compound splitting because it usually yields better results 
        # resiults
        if word1[-5:] == 'innen':
            splitted = (char_split.split_compound(word1[:-5])[0][-2])
        else:
            splitted = (char_split.split_compound(word1)[0][-2])
        return " {} {} {}{}".format(word1, chainword, splitted, word2.lower())


def remove_german_chainwords(sentence: str) -> str:
    """Split German chain words into their constituent terms."""

    regex = []

    # Split up a combination of three hyphenated chain words, e.g.: Bildungs-, Sozial- und Innenpolitik: Bildungspolitik und Sozialpolitik und Innenpolitik 
    regex.append(r"\s([A-ZÄÖÜ][a-zäöüß]+)[\s]?-[\s]?([A-ZÄÖÜ][a-zäöüß]+)-[\s]?(?:und|oder|als auch|sowie|wie|bzw|&|,)+[\s]([A-ZÄÖÜ][a-zäöüß]+)")

    # Hyphenated chain words that build two words but we have to append the second part of the second word to the first word, e.g: Ein- und Ausfuhr -> Einfuhr und Ausfuhr 
    regex.append(r"\s([A-ZÄÖÜ][a-zäöüß]+)[' ']?-[' ']?(?:und|oder|als auch|wie|sowie|bzw)+[' ']([A-ZÄÖÜ][a-zäöüß]+)")            

    # In the less common case of hyphenated chain words with the hyphen at the second word, add the first part of the first word to the end of the second word, e.g.: Reichsausgaben und -Einnahme -> Reichsausgaben und Reichseinnahmen 

    regex.append(r"\s([A-ZÄÖÜ][a-zäöüß]+)[' ']-?[' ']?(?:und|oder|als auch|wie|sowie|bzw)+[' ']-[' ']?([A-Za-zäöüßÄÖÜß]+)")
    
    # Search for each type of compound word and split them up 
    m = re.search(regex[0],sentence)
    if m:
        findings = m.groups()
        for c in zip(findings[::3], findings[1::3], findings[2::3], range(0,len(findings),3)):
            sentence = sentence.replace(sentence[m.start(c[-1]):m.end(c[-1])], char_splitting(0,c))
        return sentence
    
    m = re.search(regex[1],sentence)
    if m:
        findings = m.groups()
        for c in zip(findings[::2], findings[1::2], range(0,len(findings),2)):
            sentence = sentence.replace(sentence[m.start(c[-1]):m.end(c[-1])], char_splitting(1,c))
        return sentence
    
    m = re.search(regex[2],sentence)
    if m:
        findings = m.groups()
        for c in zip(findings[::2], findings[1::2], range(0,len(findings),2)):
            sentence = sentence.replace(sentence[m.start(c[-1]):m.end(c[-1])], char_splitting(2,c))
        return sentence
    
    return sentence

def remove_hyphens(sentence: str, split_chars="-|—|–") -> str:
    """Remove hyphens that are still prepending or appending words, After splitting up chain words -> They are either noise or a chain word was not successfully split"""

    new_text = sentence
    for t in sentence.split():
        parts = []
        for p in re.split(split_chars, t) :  # for each part p in compound token t
            if not p: continue  # skip empty part
            else:               # add p as separate token
                parts.append(p)
        if len(parts)>0:
            new_text = new_text.replace(t, "-".join(parts))
    return new_text

def remove_umlauts(sentence: str) -> str:
    text_out = []
    for tok in sentence:
        res = tok.replace('ä', 'ae')
        res = res.replace('ö', 'oe')
        res = res.replace('ü', 'ue')
        res = res.replace('Ä', 'Ae')
        res = res.replace('Ö', 'Oe')
        res = res.replace('Ü', 'Ue')
        text_out.append(res)
    return text_out


def harmonizeSpelling(line: List, spelling_dict: Dict) -> List:
    """Harmonize all words in the dictionary to uniform spelling"""
    text_out = [re.sub(tok,spelling_dict[tok],tok) if tok in spelling_dict else tok for tok in line]
    return text_out

class GermanLemmatizer:
    def __init__(self):
        self.lm = spacy.load('de_core_news_sm', disable=['parser', 'ner'])
        self.lemmatizer = GermaLemmaPlusPlus()

    def lemmatize(self, doc: List) -> List:
        doc = self.lm(doc)
        tokens = [(tok.text, tok.tag_) for tok in doc]
        sent = Sentence()
        for tokstr, pos in tokens:
            sent.add(Token(tokstr,pos))
            
        return [token.LEMMA for token in self.lemmatizer.lemmatize(sent)]


def bundestag_patterns() -> Tuple:
    start_patterns_bundestag = '|'.join(
                ['Ich eröffne die \d+ (\n )?Sitzung',
                 'Die \d+ (\n )?Sitzung des (Deutschen)? (Bundestages|Bundestags) ist eröffnet',
                 'Ich erkläre die \d+ (\n )?Sitzung des (Deutschen )?(Bundestages|Bundestags) für eröffnet',     
                 'Die (\n )?Sitzung (\n )?ist (\n )?eröffnet',
                 'Ich (\n )?eröffne (\n )?die (\n )?Sitzung',
                 'Beginn:? \d+ Uhr']
                        )
    end_patterns_bundestag = '|'.join(
                ['(Schluß|Schluss) der Sitzung \d+',
                 'Die Sitzung ist geschlossen',
                 'Ich schließe die (\n )?Sitzung'
                    ])
    bundestag_start = re.compile(f"({start_patterns_bundestag})", re.IGNORECASE)
    bundestag_end = re.compile(f"({end_patterns_bundestag})", re.IGNORECASE)

    return bundestag_start, bundestag_end

def extract_protocol_bundestag(doc: List, start_patterns: List, end_patterns: List) -> List:

    """Extract only the parts of a session protocol belonging to the actual debate without appendices, by searching for phrases used to open/close a session. If none of the patterns is found, return the unaltered text."""

    # Join the lines with new-line character to a running text, as some of the opening patterns are scattered over two lines due to faulty sentence tokenizing (e.g. "Ich eröffne die 32. Sitzung" was split after the full-stop)

    temp = ' \n '.join(doc) 

    # Search for start and end pattern
    sitzung_start = start_patterns.search(temp)
    sitzung_end = end_patterns.search(temp)

    if sitzung_start:
       # If both patterns found, return only text between start and end indices of the matching objects
       if sitzung_end:
           temp = temp[sitzung_start.start():sitzung_end.end()]
        # If only one of start/end patterns is found, discard all text before/after the found pattern.
       else: 
           temp = temp[sitzung_start.start():]
    elif sitzung_end:
       temp = temp[:sitzung_end.end()]

    # Split string by new-line character to transform protocol back to line format 
    text_out = temp.split(' \n ')
    return text_out


def reichstag_patterns() -> Tuple:
    start_patterns_reichstag = '|'.join(
               ['(Z|I)ch eröffne die (\d+ |erste )?(\n )?Sitzung',
                'Di(e|s) (\d+ |erste )?(\n )?Sitzung (\n )?ist (\n )?erö(f|s)(f|s)net',
                'Die Sitzung-ist eröffnet',
                'eröffne ich (hiermit )?die Sitzung',
                'Ich eröffne die \d+$',
                '(Z|I)ch erkläre die (\d+ |erste )?Sitzung für eröffnet']
                       )

    restart_patterns_reichstag = '|'.join(
               ['Die (\n )?Sitzung (\n )?ist wieder (\n )?eröffnet',

                'Ich eröffne die Sitzung wieder',
                'Ich eröffne die Sitzung von neuem',
                'Ich eröffne die Sitzung noch einmal'
               ])
                       
    end_patterns_reichstag = '|'.join(
               ['(Schluß|Schluss|Sckluß) der Sitzung (um )?\d+ Uhr',
                'Die Sitzung ist geschlossen',
                'Die Sitzung ist geschloffen',
                'Ich schließe die (\n )?Sitzung'
                   ])

    return start_patterns_reichstag, end_patterns_reichstag, restart_patterns_reichstag

def extract_meeting_protocols_reichstag(doc, start_patterns, end_patterns, restart_patterns, year):
    """
    param doc: document to process
    param number: which of the 4 original documents to process
    """
    PATH_TO_STORE = 'protocols'

    i = 0
    sitzung = False
    
    # Store in folder protocols_YEAR, e.g. protocols_1895
    if not os.path.exists(f'{PATH_TO_STORE}_{year}'):
        os.makedirs(f'{PATH_TO_STORE}_{year}')

    temp_doc = None

    reichstag_restart = re.compile(f"({restart_patterns})", re.IGNORECASE)
    reichstag_end = re.compile(f"({end_patterns})", re.IGNORECASE)

    if year == '1942':                 
        reichstag_start = re.compile(r'Die Sitzung wird um \d+ Uhr(?:\s\d+ Minute)?n?(?:\sabends)? durch den Präsidenten eröffnet',
                                  re.IGNORECASE)
    else:
        reichstag_start = re.compile(f"({start_patterns})", re.IGNORECASE)

    # Check in each line if a session was started, ended, interrupted or restarted
    for line in doc:
        sitzung_restart = reichstag_restart.search(line)
        sitzung_start = reichstag_start.search(line)
        sitzung_end = reichstag_end.search(line)
        sitzung_abgebrochen = re.search(r'Die Sitzung wird um \d+ Uhr (?:\d+ Minute)?n? abgebrochen', line,
                               re.IGNORECASE)
        
        if sitzung_restart:
            sitzung = True
            temp_doc.write(line)
            temp_doc.write('\n')
            continue
        if sitzung_start:
            i += 1
            if i % 100 == 0:
                logging.info(f'{i} documents extracted')
            # Set the bool 'sitzung' to True, so that each line is written to the document as long as a session is not ended or interrupted 
            sitzung = True
            # Creat a new document when new session was found
            temp_doc = codecs.open(f'{PATH_TO_STORE}_{year}/doc_{i}.txt', 'w',  encoding='utf-8')
            temp_doc.write(line)
            temp_doc.write('\n')
            continue
        if sitzung_end or sitzung_abgebrochen:
            if temp_doc is not None:
                temp_doc.write(line)
                temp_doc.write('\n')
                # Set the bool 'sitzung' to False, so that lines are not written to the document as long as another starting phrase was found
                sitzung = False
                continue

        if sitzung:
            if len(line) > 1:
                temp_doc.write(line)   
                temp_doc.write('\n')
    
    logging.info('Screening done!')
    logging.info(f'{i} documents extracted in total.')
    temp_doc.close()
