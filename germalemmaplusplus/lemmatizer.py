import sys
import copy

############################

class Token:

    def __init__(self, FORM, XPOS):
        self.FORM = FORM
        self.XPOS = XPOS

    def __str__(self):
        return self.FORM

############################

class Sentence:

    def __init__(self):
        self.tokens = list()

    #######################

    def add(self, token):
        self.tokens.append(token)

    #######################

    def __iter__(self):
        return iter(self.tokens)

    #######################

    def __str__(self):
        return " ".join(str(tok) for tok in self.tokens)
                
############################

class System:
    pass

############################

class GermaLemma(System):

    def __init__(self):
        import sys
        import germalemma
        lemmadict = {"pickle":sys.path[0]+"/lemmata.pickle"}
        self.processor = germalemma.GermaLemma(**lemmadict)

    ################################

    def lemmatize(self, sentence):
        for tok in sentence:
            try:
                tok.LEMMA = self.processor.find_lemma(tok.FORM, tok.XPOS)
            except ValueError:
                if tok.XPOS in ["NN", "NE"]:
                    tok.LEMMA = tok.FORM
                else:
                    tok.LEMMA = tok.FORM.lower()
        return sentence


############################

class GermaLemmaPlusPlus(System):

    def __init__(self):
        self.spacy = Spacy()
        self.germalemma = GermaLemma()

    ################################

    def lemmatize(self, sentence):
        
        spacysent = copy.deepcopy(sentence)
        germalemmasent = copy.deepcopy(sentence)
        
        spacysent = self.spacy.lemmatize(spacysent)
        germalemmasent = self.germalemma.lemmatize(germalemmasent)
        
        for tok, spacytok, germalemmatok in zip(sentence, spacysent, germalemmasent):
            
            tokstrlower = tok.FORM.lower()

            if tok.XPOS in ["NN", "NE", "ADJA", "ADJD", "ADV"] or tok.XPOS.startswith("V"):
                tok.LEMMA = germalemmatok.LEMMA
            
            #For pronouns use custom mapping
            #or if surface form is not captured, take spaCy's lemma
            elif tok.XPOS in ["PPER"]:
                if tokstrlower in ["ich", "mich", "mir", "meiner"]:
                    tok.LEMMA = "ich"
                elif tokstrlower in ["du", "dich", "dir", "deiner"]:
                    tok.LEMMA = "du"
                elif tokstrlower in ["er", "ihn"]:
                    tok.LEMMA = "er"
                elif tokstrlower in ["sie", "ihnen", "ihrer"]:
                    tok.LEMMA = "sie"
                elif tokstrlower in ["es"]:
                    tok.LEMMA = "es"
                elif tokstrlower in ["ihm", "seiner"]:
                    tok.LEMMA = "er_es"
                elif tokstrlower in ["ihr"]:
                    tok.LEMMA = "sie_ihr"
                elif tokstrlower in ["wir", "uns", "unser"]:
                    tok.LEMMA = "wir"
                elif tokstrlower in ["euch", "euer"]:
                    tok.LEMMA = "ihr"
                else:
                    tok.LEMMA = spacytok.LEMMA
                
            elif tok.XPOS in ["PRF"]:   
                if tokstrlower in ["mich", "mir"]:
                    tok.LEMMA = "ich"
                elif tokstrlower in ["dich", "dir"]:
                    tok.LEMMA = "du"
                elif tokstrlower in ["sich"]:
                    tok.LEMMA = "sich"
                elif tokstrlower in ["uns"]:
                    tok.LEMMA = "wir"
                elif tokstrlower in ["euch"]:
                    tok.LEMMA = "ihr"
                else:
                    tok.LEMMA = spacytok.LEMMA
                
            elif tok.XPOS in ["PPOSS", "PPOSAT"]:
                if tokstrlower.startswith("mein"):
                    tok.LEMMA = "mein"
                elif tokstrlower.startswith("dein"):
                    tok.LEMMA = "dein"
                elif tokstrlower.startswith("sein"):
                    tok.LEMMA = "sein"
                elif tokstrlower.startswith("unser"):
                    tok.LEMMA = "uns"
                elif tokstrlower.startswith("eu"):
                    tok.LEMMA = "euer"
                elif tokstrlower.startswith("ihr"):
                    tok.LEMMA = "ihr"
                else:
                    tok.LEMMA = spacytok.LEMMA
                
            #Otherwise use spaCy's lemma
            else: 
                tok.LEMMA = spacytok.LEMMA
                    
                #For indefinite ART set lemma to "ein"
                if tok.XPOS == "ART" and spacytok.LEMMA.startswith("ein"):
                    tok.LEMMA = "ein"

        return sentence

############################

class Spacy(System):

    def __init__(self):
        import spacy
        # nlp = spacy.load('de', disable=['ner', 'parser'])
        import de_core_news_sm
# !python -m spacy download de_core_news_sm
        nlp = de_core_news_sm.load(disable=['parser', 'ner'])
        self.processor = nlp

    ################################

    def lemmatize(self, sentence):
        
        from spacy.tokens import Doc

        #Create a doc object by passing the tokenized sentence to spacy's tokenizer.
        toks = [tok.FORM.lower() for tok in sentence]
        doc = Doc(self.processor.vocab, words=toks)

        #Re-tag the text to improve the lemmatization of spaCy
        self.processor.tagger(doc)

        for i, tok in enumerate(sentence):
            tok.LEMMA = doc[i].lemma_
        return sentence



if __name__ == "__main__":
    sent = Sentence()
    for tokstr, pos in [("Sie", "PPER"), ("ist", "VAFIN"), ("eine", "ART"), ("gute", "ADJA"), ("Lehrerin", "NN"), (".", "$.")]:
        sent.add(Token(tokstr, pos))
    
    print(sent)

    s = Spacy()
    gl = GermaLemma()
    glplusplus = GermaLemmaPlusPlus()

    print("spaCy:", [tok.LEMMA for tok in s.lemmatize(sent)])
    print("GermaLemma:", [tok.LEMMA for tok in gl.lemmatize(sent)])
    print("GermaLemma++:", [tok.LEMMA for tok in glplusplus.lemmatize(sent)])

