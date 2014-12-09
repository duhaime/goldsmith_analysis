from __future__ import division
from math import log
from numpy import exp
from nltk.tokenize import WordPunctTokenizer
import codecs
import regex as re
import itertools as it

#######################
# Define some globals #
#######################

#specify tokenizer
tokenizer = WordPunctTokenizer()

#specify stopwords
with codecs.open("resources\\underwood_stopwords.txt","r","utf-8") as stopwords_in:
    stopwords = set(stopwords_in.read().split())
    
###############################################
# Quick hack to print utf-8 to ascii terminal #
###############################################

def strip_non_ascii(string):
    return "".join(x for x in string if ord(x) < 128)
    
########################################################
# Create function that splits text into rolling window #
########################################################

def create_windows(iterable, size):
    shiftedStarts = [it.islice(iterable, s, None) for s in xrange(size)]
    return it.izip(*shiftedStarts)
    
##############################################
# Create function with which to clean string #
##############################################

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
    
######################################
# Read probabilities into dictionary #
######################################

def create_probability_dictionary():
    '''read in the output from Giza and generate P(e|f) for all e and all f'''
    french_english_translation_probability_dict = {}
    
    #read in ...dduhaime.actual.ti.final -- column 0 contains English word, column 1 contains French word, column 2 contains prob t(e|f)
    with codecs.open("resources/single_article_encyclopedie_probability_table.final","r","utf-8") as giza_in:
        for row in giza_in:
            split_row = row.replace("\r","").replace("\n","").split(" ")
            
            if "NULL" not in split_row:
                if split_row[1] in french_english_translation_probability_dict:
                    french_english_translation_probability_dict[ split_row[1] ][ split_row[0] ] = split_row[2]
                    
                else:
                    french_english_translation_probability_dict[ split_row[1] ] = {}
                    french_english_translation_probability_dict[ split_row[1] ][ split_row[0] ] = split_row[2]    
                
    return french_english_translation_probability_dict

####################################################### 
# Create dictionary with counts for each english word #
#######################################################
        
def create_counts_dictionary():
    '''read in a file that contains token [space] counts_of_token_in_giza_training_data. We'll use this data to generate c(e) and c(f) count values for all English and French words for Laplacian smoothing'''
    counts_of_english_words_dict = {}
    
    with codecs.open("resources/toklow.en.vcb","r","utf-8") as counts_in:
        counts_in = counts_in.read().replace("\r","").split("\n")
        for row in counts_in:
            split_row = row.split()
            if len(split_row) > 1:
                word              = split_row[1]
                word_count        = split_row[2]    
                counts_of_english_words_dict[word] = word_count
            
    return counts_of_english_words_dict

##########################################################
# Create function to calculate probability_of_trans(e|f) #
##########################################################
    
def find_unsmoothed_translation_probability( french_word, english_word ):
    '''here we simply find the naive probability of translation and return that value'''
    return float( french_english_translation_probability_dict[french_word][english_word] )
    
def find_laplace_smoothed_translation_probability( french_word, english_word ):    
    '''this function reads in a French word and an English word and then returns the Laplace smoothed probability
    that the latter is a translation of the former'''
    
    #first, get the unsmothed probability of translation
    try:
        unsmoothed_translation_probability = float( french_english_translation_probability_dict[french_word][english_word] )
    except KeyError:
        unsmoothed_translation_probability = 0
        
    #then, identify total number of types in the french dictionary (this comes from the number of rows in french_giza_training_data.vcb)
    number_types_french_training_data = 1403
    
    #next, identify total number of times the english word appears in the english training data
    try:
        count_of_english_word = float( counts_of_english_words_dict[ english_word ] )
    except KeyError:
        count_of_english_word = 0
        
    #now the smoothed translation probability is p(f|e) = c(e,f) + 1 / c(e) + types_in_french ; and c(e,f) = trans(f|e) * c(e)  ||| c = count, trans() = translation_probability
    laplace_smoothed_probability = ( (unsmoothed_translation_probability * count_of_english_word) + 1 ) / ( count_of_english_word + number_types_french_training_data)
    
    return float( laplace_smoothed_probability )
    
##################################
# Calculate pi sigma probability #
##################################

def pi_sigma(french_string, english_string):
    '''this function reads in multi-word chunks from en and fr and calculates the probability that the former is trans of latter'''
    
    pi_prob = 0
    
    for french_word in french_string.split():
        
        #reset the sum_prob for each French word
        sigma_prob = 0
            
        #with that French word in hand, loop over the English words and sum up the probability(english|french) for all of those English words
        for english_word in english_string.split():
            
            try:
                probability_of_trans = find_unsmoothed_translation_probability( french_word, english_word )
            except:
                probability_of_trans = find_laplace_smoothed_translation_probability( french_word, english_word )
                
            sigma_prob = sigma_prob + probability_of_trans    
                
        #the first time you sum the probability of each English word for the current French word, pi_prob will be 0; thereafter it will be > 0. Check to see the current value, and set pi_prob = sigma_prob if the former is currently 0
        if sigma_prob != 0:
        
            #if sigma != 0, you have to length normalize sigma by dividing by the length of the english string
            sigma_prob = sigma_prob / len(english_string.split())
            
            if pi_prob != 0:
            
                #if pi_prob != 0, then you've already log scaled the initial pi_prob, so log scale the current probabilities and sum them
                pi_prob = pi_prob + log( sigma_prob )
                
            else:
                pi_prob = log( sigma_prob )
                
    #now, to raise the whole pi_prob to 1/length_of_french_input, we simply multiply our logspace(pi_prob) value times 1/len(french_string)
    pi_prob = pi_prob * ( 1 / len(french_string.split()) )
    
    #finally, return pi_prob into probability(0:1) space by running the exponential function
    final_pi_prob = exp( pi_prob )
        
    #return pi_prob in_probability_space
    return final_pi_prob
    
#############################
# Define Inputs and Outputs #
#############################

counts_of_english_words_dict                = create_counts_dictionary()
french_english_translation_probability_dict = create_probability_dictionary()

with codecs.open("pi_sigma_probabilities.txt","w","utf-8") as out:
    with codecs.open("resources/goldsmith_training_duhaime.csv","r","utf-8") as infile:
        infile = infile.read()
        
        #split infile into its constitutive rows (each row contains a pair of sentences that either does or does not contain a translation)         
        infile_rows = infile.split("\n")
        
        row_number = 0
        
        for row in infile_rows:
            row_number += 1
            
            row_cells = row.split("\t")
            if len(row_cells) > 1:
                
                french_sentence = row_cells[0]
                english_sentence = row_cells[1]
                ground_truth = row_cells[3]
                
                clean_french = clean_string( french_sentence )
                clean_english = clean_string( english_sentence )
                
                for french_window in create_windows(clean_french.split(), 20):
                    for english_window in create_windows(clean_english.split(), 20):
                    
                        french_window_string = u" ".join(x for x in french_window)
                        english_window_string = u" ".join(x for x in english_window)
                        
                        translation_probability = pi_sigma(french_window_string, english_window_string)
                    
                        out.write( french_window_string + "\t"
                        + english_window_string + "\t"
                        + unicode( translation_probability ) + "\t"
                        + unicode( ground_truth ) + "\t"
                        + unicode( row_number ) + "\n" )
                        