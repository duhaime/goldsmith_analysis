from __future__ import division
from collections import Counter
from math import log, exp
from statistics import stdev
import codecs

def find_mean_word_length(input_string):
    return sum(len(x) for x in input_string.split()) / len(input_string.split())
    
def find_sentence_length(sentence):
    return len(sentence.split())
    
def count_syllables(word):
    vowels = ['a', 'e', 'i', 'o', 'u']

    verbose = False
    on_vowel = False
    in_diphthong = False
    minsyl = 0
    maxsyl = 0
    lastchar = None

    word = word.lower()
    for c in word:
        is_vowel = c in vowels
        
        if on_vowel == None:
            on_vowel = is_vowel
            
        # y is a special case
        if c == 'y':
            is_vowel = not on_vowel

        if is_vowel:
            #if verbose: print c, "is a vowel"
            if not on_vowel:
                # We weren't on a vowel before. Seeing a new vowel bumps the syllable count.
                minsyl += 1
                maxsyl += 1
            elif on_vowel and not in_diphthong and c != lastchar:
                # We were already in a vowel. Don't increment anything except the max count, and only do that once per diphthong.
                in_diphthong = True
                maxsyl += 1

        on_vowel = is_vowel
        lastchar = c

    # Some special cases:
    if word[-1] == 'e':
        minsyl -= 1
        
    # if it ended with a consonant followed by y, count that as a syllable.
    if word[-1] == 'y' and not on_vowel:
        maxsyl += 1

    #some quick analysis suggested that minsyl is usually more accurate than maxsyl, so we'll return only the former
    return minsyl #, maxsyl
    
def find_average_syllable_length(string):
    split_string = string.split()
    syllable_sum = sum([count_syllables(word) for word in split_string])
    average_syllable_length = syllable_sum / len(split_string)
    return average_syllable_length
    
def find_type_token_ratio(string):
    split_string = string.split()
    return len(set(split_string)) / len(split_string)
    
def find_relative_word_frequencies(string):
    split_string = string.split()
    counts = Counter(split_string)
    print counts
    #counts is a dict, so we just make a call to the key value and divide by the length of the string to get a length normalized value
    length_normalized_counts = [(counts[x]/len(split_string)) for x in counts]
    return length_normalized_counts
    
def generate_corpus_word_frequency_dict(giza_vocab_file):
    corpus_word_freqs_dict = {}
    with codecs.open(giza_vocab_file) as giza_vocab:
        giza_vocab = giza_vocab.read().replace("\r","").split("\n")
        total_words = len(giza_vocab)
        for row in giza_vocab:
            total_words += 1
            split_row = row.split(" ")
            if len(split_row) > 1:
                token = split_row[1]
                count = split_row[2]
                corpus_word_freqs_dict[token] = count / total_words
            
    return corpus_word_freqs_dict, total_words
        
def find_frequencies_of_words_in_corpus(string, corpus_freqs_dict, total_words_in_corpus):
    split_string = string.split()
    freq_list = []
    for word in split_string:
        try:
            normalized_freq = corpus_freqs_dict[word]
        except:
            normalized_freq = 1 / total_words_in_corpus
            
        freq_list.append(normalized_freq)
        return freq_list
        
def calculate_normalized_product_of_freq_list(freq_list):
    #because one sometimes runs into an underflow problem when multiplying probabilities, we convert to log space, add, then return exp value
    log_sum = sum([log(x) for x in freq_list])
    
    #now it is also important to return a length-normalized value, so multiply the log_sum (that is, raise it to power) by 1 / length_of_freq_list
    length_normalized_probability = log_sum * (1/len(freq_list))

    #run exp to return value to probability space 0:1    
    return exp(length_normalized_probability)

def calculate_standard_deviation(list_object):
    return stdev(list_object)
     
#these lines make our functions callable from outside this script   
if __name__ == '__main__':
    pass