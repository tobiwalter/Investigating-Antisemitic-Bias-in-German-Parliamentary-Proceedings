# -*- coding: utf-8 -*-
import sys
import os
import re

import spacy
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.abspath(os.path.join(ROOT_DIR,  'CharSplit')))
import char_split

sys.path.append(os.path.abspath(os.path.join(ROOT_DIR,  'germalemmaplusplus')))
from germalemmaplusplus.lemmatizer import GermaLemmaPlusPlus, Token, Sentence
from symspellpy import SymSpell,Verbosity


def remove_linebreaks(doc):
    return [re.sub(r'[\n\t]', '', line).strip() for line in doc]

def remove_punctuation(doc):    
    pattern = re.compile('[%s]' % re.escape('x!"#$&\'()*+,./:;<=>?@®©[\\]^_`{|}~„“«»'))
    return [re.sub(pattern,'', line).strip() for line in doc]

def remove_double_spaces(doc):
    return [re.sub('\s\s+',' ', line).strip() for line in doc]

def remove_noisy_digits(doc):
    """
    Remove digit 1 appending or prepending to some words in the text due to faulty OCR
    """
    temp = [re.sub(r'\b1(?P<quote>[A-Za-z]+)\b', '\g<quote>', line) for line in doc]
    temp = [re.sub(r'\b(?P<quote>[A-Za-z]+)1\b', '\g<quote>', line) for line in temp] 
    temp = [re.sub(r'\b(?P<quote>[A-Za-zßäöü]+)1(?P<name>[A-Za-zßäöü]*)\b', '\g<quote> \g<name>', line) for line in temp]
    return temp

def replace_digits(doc):
    return [re.sub(r'\d+',' 0 ' ,line).strip() for line in doc]

def reduce_numerical_sequences(doc): 
    """
    If there are sequences of digits appearing, reduce to a single placeholder
    """
    return [re.sub(r'((0)\s?){2,}', '\\2 ', line).strip() for line in doc]

def filter_lines(doc):
    """
    Filter all lines that are empty or only a single char long
    """
    return [line for line in doc if len(line) >1]

def lowercase(line):
    return [tok.lower() for tok in line]

def remove_dash_and_minus_signs(doc):
    """
    remove lone standing dashes/hyphens and minus signs at beginning, middle and end of line
    """
    temp = [re.sub(r'\s(-|—|–|\s)+\s', ' ', line) for line in doc]
    temp = [re.sub(r'^(-|—|–|\s)+\s', '', line) for line in temp]
    temp = [re.sub(r'\b\s(-|—|–|\s)+$', '', line) for line in temp]
    return temp

def charSplitting(i,groups,chainword="und"):
    
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


def removeGermanChainWords(text):
    regex = []
    # splitting up a combination of three hyphenated chain words, e.g.: Bildungs-, Sozial- und Innenpolitik: Bildungspolitik und Sozialpolitik und Innenpolitik 
    regex.append(r"\s([A-ZÄÖÜ][a-zäöüß]+)[\s]?-[\s]?([A-ZÄÖÜ][a-zäöüß]+)-[\s]?(?:und|oder|als auch|sowie|wie|bzw|&|,)+[\s]([A-ZÄÖÜ][a-zäöüß]+)")

    # hyphenated chain words building two words but we have to append the second part of the second word to the first word, e.g: Ein- und Ausfuhr -> Einfuhr und Ausfuhr 
    regex.append(r"\s([A-ZÄÖÜ][a-zäöüß]+)[' ']?-[' ']?(?:und|oder|als auch|wie|sowie|bzw)+[' ']([A-ZÄÖÜ][a-zäöüß]+)")            
    # in the less common case of hyphenated chain words with the hyphen at the second word, add the first part of first word to end of second, e.g.: Reichsausgaben und -Einnahme -> Reichsausgaben und Reichseinnahmen 
    regex.append(r"\s([A-ZÄÖÜ][a-zäöüß]+)[' ']-?[' ']?(?:und|oder|als auch|wie|sowie|bzw)+[' ']-[' ']?([A-Za-zäöüßÄÖÜß]+)")
    
    sentence = text
    m = re.search(regex[0],sentence)
    if m:
        findings = m.groups()
        for c in zip(findings[::3], findings[1::3], findings[2::3], range(0,len(findings),3)):
            sentence = sentence.replace(sentence[m.start(c[-1]):m.end(c[-1])], charSplitting(0,c))
        return sentence
    
    m = re.search(regex[1],sentence)
    if m:
        findings = m.groups()
        for c in zip(findings[::2], findings[1::2], range(0,len(findings),2)):
            sentence = sentence.replace(sentence[m.start(c[-1]):m.end(c[-1])], charSplitting(1,c))
        return sentence
    
    m = re.search(regex[2],sentence)
    if m:
        findings = m.groups()
        for c in zip(findings[::2], findings[1::2], range(0,len(findings),2)):
            sentence = sentence.replace(sentence[m.start(c[-1]):m.end(c[-1])], charSplitting(2,c))
        return sentence
    
    return sentence

def remove_hyphens_pre_and_appending(text, split_chars="-"):
    '''Remove prepending and appending hyphens from words -> They are either noise or the chain word could not be split'''
    new_text = text
    for t in text.split():
        parts = []
        for p in t.split(split_chars):  # for each part p in compound token t
            if not p: continue  # skip empty part
            else:               # add p as separate token
                parts.append(p)
        if len(parts)>0:
            new_text = new_text.replace(t, "-".join(parts))
    return new_text


start_patterns = '|'.join(
                ['Ich eröffne die \d+ (\n )?Sitzung',
                 'Die \d+ (\n )?Sitzung des (Deutschen)? (Bundestages|Bundestags) ist eröffnet',
                 'Ich erkläre die \d+ (\n )?Sitzung des (Deutschen )?(Bundestages|Bundestags) für eröffnet',     
                 'Die (\n )?Sitzung (\n )?ist (\n )?eröffnet',
                 'Ich (\n )?eröffne (\n )?die (\n )?Sitzung',
                 'Beginn:? \d+ Uhr']
                        )
end_patterns = '|'.join(
                ['(Schluß|Schluss) der Sitzung \d+',
                 'Die Sitzung ist geschlossen',
                 'Ich schließe die (\n )?Sitzung'
                    ])

start_pattern = re.compile(f"({start_patterns})", re.IGNORECASE)
end_pattern = re.compile(f"({end_patterns})", re.IGNORECASE)
def extract_protocol(doc):

   temp = ' \n '.join(doc) 

   # Search for start and end pattern
   sitzung_start = start_pattern.search(temp)
   sitzung_end = end_pattern.search(temp)
   if sitzung_start:
       # If both patterns found, return only text between start and end of the matching objects
       if sitzung_end:
           temp = temp[sitzung_start.start():sitzung_end.end()]
   # If only start or end pattern found, use the found pattern as border for start/end
       else: 
           temp = temp[sitzung_start.start():]
   elif sitzung_end:
       temp = temp[:sitzung_end.end()]
   # If none found, return unaltered text

   # Split string by new-line character to transform protocol back to line format 
   text_out = temp.split(' \n ')
   return text_out

def removeUmlauts(text):
    text_out = []
    for tok in text:
        res = tok.replace('ä', 'ae')
        res = res.replace('ö', 'oe')
        res = res.replace('ü', 'ue')
        res = res.replace('Ä', 'Ae')
        res = res.replace('Ö', 'Oe')
        res = res.replace('Ü', 'Ue')
        text_out.append(res)
    return text_out

spelling_dict = open(os.path.join(ROOT_DIR, 'dictionaries/harmonize_dict.txt'), 'r').readlines()
spelling_dict = {line.split()[0] : line.split()[1] for line in spelling_dict}

# def harmonizeSpelling(text):
#     text_out = [re.sub(k,v,tok) for k,v in spelling_dict.items() for tok in text]
#     return text_out

def harmonizeSpelling(text):
    text_out = [re.sub(tok,spelling_dict[tok],tok) if tok in spelling_dict else tok for tok in text]
    
    return text_out
class GermanLemmatizer:
    def __init__(self):
        self.lm = spacy.load('de_core_news_sm', disable=['parser', 'ner'])
        self.lemmatizer = GermaLemmaPlusPlus()

    def lemmatize(self, doc):
        doc = self.lm(doc)
        tokens = [(tok.text, tok.tag_) for tok in doc]
        sent = Sentence()
        for tokstr, pos in tokens:
            sent.add(Token(tokstr,pos))
            
        return [token.LEMMA for token in self.lemmatizer.lemmatize(sent)]

class GermanSpellChecker:
    def __init__(self, dictionary_path, count_treshold=10, max_edit_distance=1):
        self.spell_checker = SymSpell(count_threshold=count_treshold, max_dictionary_edit_distance=max_edit_distance)
        loaded = self.load_dictionary(dictionary_path)
        if loaded:
            print('Dictionary loaded!')
        else:
            print('Dictionary not loaded!')

    def load_dictionary(self, dictionary_path):
        return self.spell_checker.load_dictionary(dictionary_path, 0, 1, separator = ' ', encoding='utf-8')


    def correct(self, doc, skip_token=r'\d+'):
        ''' Looks up top suggestion for each token in the document and return it'''
        text_out = [self.spell_checker.lookup(tok, Verbosity.TOP, include_unknown=True,ignore_token=skip_token)[0].term
                    if len(tok) >=4 else tok for tok in doc]
        return text_out

    def add_entries(self,list, count=10):
        for word in list:
            self.spell_checker.create_dictionary_entry(word,count)

    def delete_entries(self,list):
        for word in list:
            self.spell_checker.delete_dictionary_entry(word)
