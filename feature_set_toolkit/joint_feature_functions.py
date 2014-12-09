'''This script will aggregate all joint features, or features that require simultaneous analysis on French and English strings. The majority of
these features will consist of the following form. For each combination of n words in English and n words in French (where n in a parameter from
3 to 10 (e.g.), we calculate the pi_sigma_probability of that window pair. So when n = 3, we find all combinations of 3 words from E and compare
each of those to all windows from F. During each comparison, we find the pi_sigma prob given those windows. We then return the max probability
for the three gram combinations.

Next we run a parameter sweep of consecutive n grams, from 2:10. For each, we again find the max of each run and return it.

Then we run a parameter sweep of consecutive n grams from 3:10 skipping parameter sweep from 1:6 and return the max for each sweep.'''

from nltk.tokenize import WordPunctTokenizer
from itertools import islice
import pi_sigma
import regex as re
import codecs

def strip_non_ascii(string):
        return "".join(x for x in string if ord(x) < 128)

def load_stopwords():
    with codecs.open("feature_set_toolkit/resources/underwood_stopwords.txt","r","utf-8") as stopwords_in:
        return set(stopwords_in.read().split())
        
stopwords = load_stopwords()
tokenizer = WordPunctTokenizer()

def clean_string(dirty_string):
        
        #tokenize
        cleaner_string = u" ".join(tokenizer.tokenize(dirty_string))
    
        #strip punctuation
        cleaner_string = re.sub(ur"\p{P}+", "", cleaner_string)
    
        #lower
        cleaner_string = cleaner_string.lower()
        
        #remove stopwords
        cleaner_string = u" ".join(x for x in cleaner_string.split() if x not in stopwords)
        
        return cleaner_string

def calculate_pi_sigma(english_string, french_string):
    return pi_sigma.pi_sigma_probability(english_string, french_string)

def create_windows(iterable, length):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(iterable)
    result = tuple(islice(it, length))
    if len(result) == length:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result
    
def find_n_gram_trans_prob(full_english, full_french):
    
    max_prob_array = []
    
    clean_english = clean_string(full_english)
    clean_french  = clean_string(full_french)
    
    #This function reads in a full english passage and full french passage, and finds the max probability for each n gram unit from 2:6
    for window_extension in xrange(5):
        window_length = 2 + window_extension

        #create windows retuns a generator object, so coerce to a list and iterate over both
        english_windows = list( create_windows( clean_english.split(), window_length ) )
        french_windows  = list( create_windows( clean_french.split(), window_length ) )
        
        prob_array_for_current_window_length = []
        
        for english_window in english_windows:
            for french_window in french_windows:
            
                #the windows currently consist of list objects, so join to string for analysis
                english_window_joined = " ".join(english_window)
                french_window_joined  = " ".join(french_window) 
            
                window_trans_prob = calculate_pi_sigma( english_window_joined, french_window_joined )
                
                #print strip_non_ascii(english_window_joined), strip_non_ascii(french_window_joined), window_trans_prob
                
                prob_array_for_current_window_length.append( window_trans_prob )
                
        #now here we run into a problem if the input sentence is too short for our desired window length
        max_value_for_current_window_length = max( prob_array_for_current_window_length )
        
        max_prob_array.append( max_value_for_current_window_length )
        
    return max_prob_array
    
def run_pi_sigma(full_english, full_french):
    clean_english = clean_string( full_english )
    clean_french  = clean_string( full_french )
    
    return calculate_pi_sigma( clean_english, clean_french )