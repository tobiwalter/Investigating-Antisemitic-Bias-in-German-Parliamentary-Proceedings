import re
import json
import pickle
from pathlib import Path

DATA_FOLDER = Path('./obj')
VOCAB_FOLDER = Path('./data/vocab')

# sets of pleasant/unpleasant words

PLEASANT = 'streicheln, Freiheit, Gesundheit, Liebe, Frieden, Freude, Freund, Himmel, loyal, Vergnügen, Diamant, sanft, ehrlich, \
glücklich, Regenbogen, Diplom, Geschenk, Ehre, Wunder, Sonnenaufgang, Familie, Lachen, Paradies, Ferien'.lower().split(', ') 

UNPLEASANT = 'Mißbrauch, Absturz, Schmutz, Mord, Krankheit, Tod, Trauer, vergiften, stinken, Angriff, Katastrophe, Haß, \
verschmutzen, Tragödie, Scheidung, Gefängnis, Armut, häßlich, Krebs, töten, faul, erbrechen, Qual'.lower().split(', ') 

# sets patriotic/non-patriotic words 

VOLKSTREU = 'patriotisch, vaterlandsliebe, volksbewußtsein, volksgeist, germanische, deutschnational, nationalbewußtsein, \
vaterländisch, reichstreu, nationalgesinnt, nationalstolz, deutschnational, königstreu'.split(', ')

VOLKSUNTREU = 'nichtdeutsch, fremdländisch, fremd, undeutsch, vaterlandslos, reichsfeind, landesverräter, reichsfeindlich, \
unpatriotisch, antideutsch, deutschfeindlich, umstürzler'.split(', ')   

OUTSIDER_WORDS = 'unaufrichtig, komisch, boshaft, unberechenbar, primitiv, beängstigend, hinterlistig, energisch, trügerisch, \
neidisch, gierig, abscheulich, verachtenswert, brutal, ungeheuer, berechnend, grausam, gemein, intolerant, aggressiv'\
.lower().split(', ') 

JEWISH_STEREOTYPES_NOUNS = 'Gier, Wucherer, Drückeberger, Kriegsgewinnler, Macht, Einfluß, Kriegstreiber, Lügner, \
Weltherrschaft, Kommunismus, Kapitalismus, Liberalismus, Außenseiter'.lower().split(', ')

JEWISH_STEREOTYPES_CHARACTER = 'egoistisch, fremd, dekadent, haßerfüllt, habgierig, geldgierig, penetrant, hinterlistig, \
intellektuell, pervers, hinterhältig, betrügerisch, gebeugt, bucklig'.split(', ')

JEWISH_STEREOTYPES_POLITICAL = 'liberalistisch, modern, materialistisch, liberal, undeutsch, unpatriotisch, säkular, \
sozialistisch, links, bolschewistisch'.split(', ')

JEWISH_OCCUPATIONS = 'Pfandleiher, Geldleiher, Kaufmann, Händler, Bankier, Finanzier, Steuereintreiber, Zöllner, \
Trödelhändler'.lower().split(', ') 

def charSplitting(i,groups,chainword="und"):
    word1 = groups[0].replace(" ","")
    word2 = groups[1].replace(" ","")
    if len(groups) >= 4:
        word3 = str(groups[2]).replace(" ","")
    if len(groups) >= 5:
        word4 = str(groups[3]).replace(" ","")
    if i == 0:
        #print("{}{}".format(groups[0],groups[1]),groups[1])
        return "{}{} {} {}".format(word1,word2.lower(),chainword,word2)
    if i == 1:
        if len(word4)>1:
            splitted = char_split.split_compound(word4)[0][-1].lower()
            return "{}{} {} {}{} {} {}".format(word1,word3.split("-")[1],chainword,word2,word3.split("-")[1],chainword,word3)
        else:
            splitted = char_split.split_compound(word3)[0][-1].lower()
            return "{}{} {} {}{} {} {}".format(word1,splitted,chainword,word2,splitted,chainword,word3)
    if i == 2:
        if len(word3)>1:
            splitted = char_split.split_compound(word3)[0][-1].lower()
            return "{}{} {} {}".format(word1,word2.split("-")[1],chainword,word2)
        else:
            splitted = char_split.split_compound(word2)[0][-1].lower()
            return "{}{} {} {}".format(word1,splitted,chainword,word2)

def removeGermanChainWords(text):
    regex = []
    # brackets with following word: usually belonging together in german: (Wirtschafts-)Informatik, building two words
    regex.append("['(']{1}([A-Za-z0-9_äÄöÖüÜß]+).[')'](.?\w+)")
    # list of combined words beloning together (3)
    regex.append("([A-Za-z0-9_äÄöÖüÜß]+)-[,][' ']?([A-Za-z0-9_äÄöÖüÜß]+)-[' ']?[und|oder|sowie|&|,]+[' ']([A-Za-z0-9_äÄöÖüÜß]+-?([A-Za-z0-9_äÄöÖüÜß]+))")
    # brackets with following word: usually belonging together in german: lv- oder kvbestandsfuehrungssystem, 
    # building two words but we have to append the second part of the second word to the first word
    regex.append("([A-Za-z0-9_äÄöÖüÜß]+)-[' ']?[und|oder|sowie|&]+[' ']([A-Za-z0-9_äÄöÖüÜß]+-?([A-Za-z0-9_äÄöÖüÜß]+))")
    # Wirtschafts-/Informatik
    regex.append("([A-Za-z0-9_äÄöÖüÜß]+)-['']?['/','&',',']['']?([A-Za-z0-9_äÄöÖüÜß]+)")
    
    sentence = text
    m = re.search(regex[0],sentence)
    if m:
        findings = m.groups()
        for c in zip(findings[::2], findings[1::2], range(0,len(findings),2)):
            sentence = sentence.replace(sentence[m.start(c[-1]):m.end(c[-1])], charSplitting(0,c))


    m = re.search(regex[1],sentence)
    if m:
        findings = m.groups()
        for c in zip(findings[::4], findings[1::4], findings[2::4], findings[3::4], range(0,len(findings),4)):
            sentence = sentence.replace(sentence[m.start(c[-1]):m.end(c[-1])], charSplitting(1,c))

    m = re.search(regex[2],sentence)
    if m:
        findings = m.groups()
        for c in zip(findings[::3], findings[1::3], findings[2::3], range(0,len(findings),3)):
            sentence = sentence.replace(sentence[m.start(c[-1]):m.end(c[-1])], charSplitting(2,c))

    return sentence

def glplusplus_lemmatizing(texts):
    nlp = de_core_news_sm.load(disable=['parser', 'ner'])
    doc = nlp(texts)
    tokens = [(tok.text, tok.tag_) for tok in doc]
    sent = Sentence()
    for tokstr, pos in tokens:
        sent.add(Token(tokstr,pos))
        
    return [token.LEMMA for token in lemmatizer_plus.lemmatize(sent)]

def save_corpus(corpus, filepath):
    with open(str(DATA_FOLDER / filepath) + '.pkl', 'wb') as f:
        pickle.dump(corpus, f, pickle.HIGHEST_PROTOCOL)

def save_vocab(model, filepath):
    words = sorted([w for w in model.wv.vocab], key=lambda w: model.wv.vocab.get(w).index)
    index = {w: i for i, w in enumerate(words)}
    json_repr = json.dumps(index)
    with open(str(VOCAB_FOLDER / filepath) + '.json',"w", encoding='utf-8') as f:    
        f.write(json_repr)
        f.close()



def load_corpus(filepath):
    with open(str(DATA_FOLDER / filepath), 'rb') as f:
        corpus = pickle.load(f)
        return corpus

def assign_label(word, attributes):
    if word in jewish_words:
        return 'jewish'
    elif word in christian_words:
        return 'christian'
    else: return attributes

def filter_target_set(target_set, embeddings):
    '''Filter out all target words that did not reach min_count and hence are not in the embeddings'''
    return [word for word in target_set if word in embeddings]

def create_attribute_sets(embeddings):
    attribute_sets = {
        'pleasant' : filter_target_set(PLEASANT, embeddings),
        'unpleasant' : filter_target_set(UNPLEASANT, embeddings),
        'outsider_words' : filter_target_set(OUTSIDER_WORDS, embeddings), 
        'jewish_occupations' : filter_target_set(JEWISH_OCCUPATIONS, embeddings),
        'jewish_nouns' : filter_target_set(JEWISH_STEREOTYPES_NOUNS, embeddings),
        'jewish_character' : filter_target_set(JEWISH_STEREOTYPES_CHARACTER, embeddings),
        'jewish_political' : filter_target_set(JEWISH_STEREOTYPES_POLITICAL, embeddings),
        'volkstreu' : filter_target_set(VOLKSTREU, embeddings),
        'volksuntreu' : filter_target_set(VOLKSUNTREU, embeddings)
         }
    
    return attribute_sets


def create_target_sets(word_vectors): 
    target_sets = {
    'jewish' : filter_target_set('rabbi, synagoge, koscher, sabbat, orthodox, judentum, jude, jüdisch, mose, talmud, israel, abraham, zionistisch'.split(', '), word_vectors),
    
    'christian' : filter_target_set('taufe, katholizismus, christentum, evangelisch, evangelium, jesus, christ, christlich, katholisch, kirche, pfarrer, ostern, bibel'.split(', '), word_vectors),
    
    'catholic' : filter_target_set('katholizismus, katholisch, katholik, papst, römisch-katholisch, jesuiten, jesuitenorden, ultramontanismus, ultramontanen, zentrumspartei'.split(', '), word_vectors),
    
    'protestant' : filter_target_set('protestant, protestantisch, protestantismus, evangelisch, evangelium, landeskirche, oberkirchenrat, lutherisch, evangelisch-lutherisch, reformiert'.split(', '), word_vectors)
                }
    # Join them together to form bias words
    return target_sets

def load_embeddings(embeddings_path):
    """
    >>> load_embeddings("/work/anlausch/glove_twitter/glove.twitter.27B.200d.txt")
    :param path:
    :return:
    """
    embbedding_dict = {}
        #Load Google's pre-trained Word2Vec model.
    if os.name != 'nt':
        model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    else:
        model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
    return model

def load_embedding_dict(vocab_path="", vector_path="", embeddings_path="", glove=False, postspec=False):
  """
  >>> _load_embedding_dict()
  :param vocab_path:
  :param vector_path:
  :return: embd_dict
  """
  if embeddings_path != "":
    embd_dict = utils.load_embeddings(embeddings_path)
    return embd_dict
  else:
    embd_dict = {}
    vocab = json.load(open(vocab_path, 'r'))
    vectors = np.load(vector_path)
    for term, index in vocab.items():
      embd_dict[term] = vectors[index]
    assert len(embd_dict) == len(vocab)
    return embd_dict



