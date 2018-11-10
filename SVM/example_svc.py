# Alex Choy
# sample code that would take an input 
# of a digit (64 pixels to 1 vector) and
# return a classification result using SVM

# import modules
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
import argparse

preproc_object = './normalizer.pkl'
svc_object = './svclassifier.pkl'

def argumentparser():
    parser = argparse.ArgumentParser(description= \
        'This function runs an SVC model that classifies \
         handwritten digits of 64 pixels. This is for \
         sample tutorial/portfolio purposes')
    parser.add_argument('--data','d',help='Input 64 pixels of \
        handwritten digits')
    return parser

def main():
    '''
    Simple function that runs simple digit classifier
    '''
    
    # get inputs from argument parser
    parser = argumentparser()
    args = parser.parse_args()
    
    X = args.data
    
    # load pkl files for preprocessing and model
    load_preprocessor = pickle.load(open(preproc_object,'rb'))
    load_svc = pickle.load(open(svc_object,'rb'))
    
    X_norm = load_preprocessor.transform(X)
    output = load_svc.predict(X_norm)
    
    return output

if __name__ == '__main__':
    main()