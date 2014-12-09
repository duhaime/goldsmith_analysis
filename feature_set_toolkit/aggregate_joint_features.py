'''This script will aggregate all joint features, or features that require simultaneous analysis on French and English strings. These features
are defined in joint_feature_functions.py'''

import joint_feature_functions

def calculate_joint_vectors(english_string, french_string):
    joint_feature_list = []
    
    sliding_n_gram_values = joint_feature_functions.find_n_gram_trans_prob( english_string, french_string )
    aggregate_pi_probability = joint_feature_functions.run_pi_sigma( english_string, french_string )
    
    for value in sliding_n_gram_values:
        joint_feature_list.append( value )
        
    joint_feature_list.append( aggregate_pi_probability )
    
    return joint_feature_list    