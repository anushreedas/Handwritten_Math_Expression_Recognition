"""
Wrapper program for the entire project
This program runs Segmenter program for each sample from dataset.

@author: Anushree D
@author: Nishi P
@author: Omkar S
"""

import os, sys
from tqdm import tqdm
import Segmenter
import pickle5 as pickle

if __name__ == '__main__':
    # Do not include '/' in file level of folder
    filepath = sys.argv[1]
    # Examples
    # filepath = 'data/bonus_inkml'
    # filepath = 'data/inkml'
    # filepath = 'data/inkml/extension'
    try:
        assert os.path.exists(filepath)
    except:
        print('Check path')
        sys.exit(0)

    filelist = []
    # get all inkml files from directory and sub-directories
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if os.path.splitext(file)[1] == '.inkml':
                filelist.append(os.path.join(root, file))

    model_pickle = 'rf_merge_final.pkl'
    # load the classifier model
    model = open(model_pickle, 'rb')
    merge_classifier = pickle.load(model)

    model_pickle = 'rf_recognize.pkl'
    # load the classifier model
    model = open(model_pickle, 'rb')

    symbol_classifier = pickle.load(model)

    # filelist = filelist[6400:6600]

    for path in tqdm(filelist):
        m = Segmenter.MathExpression(path, merge_classifier, symbol_classifier)
        m.write_lgfile('train_all')
        # m.write_lgfile('result')
        print(path)
