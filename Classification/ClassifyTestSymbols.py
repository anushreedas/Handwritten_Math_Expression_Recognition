'''
This program classifies the given inkml files using the already built
SVM and Randomn Forest classifier and saves the results in a csv file

@author: Anushree D
@author: Nishi P
@author: Omkar S
'''

import pickle5 as pickle
from ExtractFeatures import *
import pandas as pd
from LoadInkmlFiles import *
import sys


def classify_samples(test_data,model_pickle,output_file):
    '''
    Classifies the given data samples using the already built
    SVM and Randomn Forest classifier and saves the results in a csv file
    :param test_data:   Dataset for classification
    :param model_pickle: path to the stored classifier model
    :param output_file: path to csv which stores the results
    :return: None
    '''
    # extract features from the samples
    X_test, y_test = feature_extraction(test_data)

    # load the classifier model
    model = open(model_pickle, 'rb')
    classifier = pickle.load(model)
    # classify dataset using the classifier model
    y_pred = classifier.predict(X_test)

    # get the UI for each row
    UIs = []
    for UI in test_data:
        UIs.append(UI)

    # write classification results to csv
    df = pd.DataFrame({'UI': UIs, 'y_pred': y_pred})
    df.to_csv(output_file, index=False, header=False)


if __name__ =='__main__':
    # check command parameters
    if len(sys.argv) <=2:
        print("Please enter parameters: [svm/rf] [path to directory]")
        exit(0)
    # can be svm or rf
    classifier_type = sys.argv[1]
    # path to classifier model
    model_pickle = classifier_type+'.pkl'
    # path to save results
    output_file = 'train_result_'+classifier_type+'.csv'
    # path to directory where the inkml files are
    data_dir = sys.argv[2]

    # load datasets

    # data_pickle = 'training_data.pkl'
    # with open(data_pickle, 'rb') as pickle_file:
    #     test_data = pickle.load(pickle_file)

    test_data = load_inkml_files(data_dir,test =True)

    # classify the samples
    classify_samples(test_data,model_pickle,output_file)
