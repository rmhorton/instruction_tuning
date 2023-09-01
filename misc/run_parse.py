# run_parse
# JMA 31 Aug 2023
import re, os, sys, pprint
import spacy as sp   

def run_parse(the_sentence):
    phrase = english_language(the_sentence)
    print(phrase)
    # print([(i, i.label_) for i in phrase.ents]) We dont need the entities. 
    the_parse = []
    found_root = False
    for k, token in enumerate(phrase):
        # 'Stop' words are the closed classes e.g. pronouns, of words.  Only a small finite number of words make up the class.
        # Stop words  plus the auxilaries and verb give us all the gramatical structure we need.

        if token.dep_ =='ROOT':
            the_parse.append((token.lemma_, token.tag_, sp.explain(token.tag_), token.pos_, token.dep_, k))
            # One word after after the main verb is needed for inversions.  
            found_root = True
        elif token.is_stop or found_root:
            # print(token.text, token.lemma_, end = '\t')
            # print(token.is_stop, token.tag_, token.pos_, token.dep_)
            the_parse.append((token.lemma_, token.tag_, sp.explain(token.tag_), token.pos_, token.dep_,k))
            # The root is the main verb in the sentence
            if found_root:
                break

    return the_parse

def get_lemma(token_parse):
    return token_parse[0]

def get_tag(token_parse):
    return token_parse[1]

def get_pos(token_parse):
    return token_parse[3] 

def get_dep(token_parse):
    return token_parse[4] 

english_language = sp.load('en_core_web_trf')
while True:
    s = input('> ')  #sys.stdin.readline('> ')
    print(f's {s}')
    if not s:
        break
    p = run_parse(s)
    pprint.pprint(p)
    print('lemma:\t', [get_lemma(z) for z in p])
    print('tag:\t', [get_tag(z) for z in p])
    print('POS:\t', [get_pos(z) for z in p])
    print('dep\t', [get_dep(z) for z in p])