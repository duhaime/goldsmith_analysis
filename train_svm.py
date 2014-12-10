import sys, codecs, pickle
sys.path.insert(0, 'feature_set_toolkit')
import aggregate_atomic_features
import aggregate_joint_features
import regex as re
from sklearn import svm
from nltk.tokenize import WordPunctTokenizer

'''This script takes as input a three column training file, with col_one = English string, col_two = French string, and col_three 0/1 value
that indicates whether English is a plagiarism of French or not. It uses aggregate_features.py to extract features, and feeds the extracted
features to an SVM via sci-kit learn. It then returns the trained classifier, which may be called further in the pipeline.'''

############################ 
# Define Cleaning Supplies #
############################

def load_stopwords():
    with codecs.open("feature_set_toolkit/resources/underwood_stopwords.txt","r","utf-8") as stopwords_in:
        return set(stopwords_in.read().split())
        
stopwords = load_stopwords()
tokenizer = WordPunctTokenizer()

#######################
# Parse Training Data #
#######################

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

#############
# Train SVM #
#############        
        
train_svm = 0
deploy_svm = 1

#we check here whether to train or not because it's a time consuming procedure, and if the classifier is already trained we can load it via pickle
if train_svm == 1:
        
    with codecs.open("goldsmith_training_subset.csv","r","utf-8") as f:
        f = f.read().split("\n")
        
        #list of feature lists will consist of an array of sublists, each of which contains an array that describes a row in our training data
        list_of_feature_lists = []
        list_of_groundtruth_values = []
        
        for row in f[0:47]:
            try:
            
                split_row = row.split("\t")
                if len(split_row) > 1:
                    french      = split_row[0]
                    english     = split_row[1]
                    groundtruth = split_row[3]
                    
                    #because one of our joint_feature functions halts if it takes as input a string of len < 6 sans stopwords, we should check here and continue if this is the case
                    if len( clean_string(french)) < 6:
                        print "skipped string"
                        continue
                    if len( clean_string(english)) < 6:
                        print "skipped string"
                        continue
                    
                    ##########################
                    # Gather Atomic Features #
                    ##########################
                        
                    english_features = aggregate_atomic_features.calculate_atomic_vectors(english)
                    french_features  = aggregate_atomic_features.calculate_atomic_vectors(french)
                    
                    #########################
                    # Gather Joint Features #
                    #########################
                    
                    joint_features = aggregate_joint_features.calculate_joint_vectors(english, french)
                    
                    #####################################
                    # Combine Atomic and Joint Features #
                    #####################################
                    
                    print english_features, french_features, joint_features
                    
                    combined_features = []
                    
                    for value in english_features:
                        combined_features.append(value)
                        
                    for value in french_features:
                        combined_features.append(value)
                        
                    for value in joint_features:
                        combined_features.append(value)
                    
                    ##########################################
                    # Dump Current Lists into Training Lists #
                    ##########################################
                    
                    list_of_feature_lists.append( combined_features )
                    
                    print "lofl:", list_of_feature_lists
                    
                    list_of_groundtruth_values.append( int(groundtruth) )
            
            except Exception as exc:
                print exc, "Couldn't process the following line:", "".join(x for x in english if ord(x) < 128)   
                
        ############# 
        # Train SVM #
        #############
        
        print list_of_feature_lists, list_of_groundtruth_values
        
        clf = svm.SVC()
        clf.fit( list_of_feature_lists , list_of_groundtruth_values )
    
    ##################################
    # Save / Load Trained Classifier #
    ##################################
    
    #to save
    pickle.dump( clf, open( "pickled_svm.p", "wb" ) )
    
if deploy_svm == 1:    
    
    #to load
    clf = pickle.load( open( "pickled_svm.p", "rb" ) )
    
    ###########################
    # Test Trained Classifier #
    ###########################
    
    with codecs.open("goldsmith_training_subset.csv","r","utf-8") as f:
        f = f.read().split("\n")
    
        for row in f[0:50]:
            try:
                split_row = row.split("\t")
                if len(split_row) > 1:
                    french      = split_row[0]
                    english     = split_row[1]
                    groundtruth = split_row[3] 
                    
                    english_features = aggregate_atomic_features.calculate_atomic_vectors(english)
                    french_features  = aggregate_atomic_features.calculate_atomic_vectors(french)
                    
                    #########################
                    # Gather Joint Features #
                    #########################
                    
                    joint_features = aggregate_joint_features.calculate_joint_vectors(english, french)
                    
                    #####################################
                    # Combine Atomic and Joint Features #
                    #####################################
                    
                    combined_features = []
                    for value in english_features:
                        combined_features.append(value)
                        
                    for value in french_features:
                        combined_features.append(value)
                        
                    for value in joint_features:
                        combined_features.append(value)
                    
                    print clf.predict([ combined_features ]), "\t", "\t".join([str(i) for i in combined_features])
                    
            except Exception as exc:
                print exc
