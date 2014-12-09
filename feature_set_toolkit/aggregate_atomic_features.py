import atomic_feature_functions
import pi_sigma
import codecs

def calculate_atomic_vectors(input_string):
    '''This function takes as input a string and returns as output an ordered list of features pertaining to that string.'''
    
    feature_list = []
       
    #################################
    # Call Atomic Feature Functions #
    #################################
    
    feature_list.append( atomic_feature_functions.find_mean_word_length(input_string) )
    feature_list.append( atomic_feature_functions.find_sentence_length(input_string) )
    feature_list.append( atomic_feature_functions.find_average_syllable_length(input_string) )
    feature_list.append( atomic_feature_functions.find_type_token_ratio(input_string) )
   
    return feature_list