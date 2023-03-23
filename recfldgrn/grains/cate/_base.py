import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def process_input_str(input_str, mapping_num = False):
    # replace all the punctuation to space.
    # for i in string.punctuation:
    #     input_str = input_str.replace(i, ' ')
    # make the word tokenization. 
    input_str = word_tokenize(input_str)
    
    output_str = []
    for word in input_str:
        word_n = word.lower()
        if False: #by default, change the word format to the normal one.
            word_n = lemmatizer.lemmatize(word_n)
            word_n = lemmatizer.lemmatize(word_n, 'v')
        if has_numbers(word_n) == True and mapping_num == True:
            word_n = 'num'
        # if word_n == 'walking':
        #     word_n = 'walk'
        # if word_n[-2:] == 'ly':
        #     word_n = word_n.replace('ly', '')
        output_str.append(word_n)
    
    return output_str