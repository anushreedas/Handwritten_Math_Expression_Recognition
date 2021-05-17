'''
This program builds a SVM and Randomn Forest classifier and trains it using the training dataset features

@author: Anushree D
@author: Nishi P
@author: Omkar S
'''
import pickle5 as pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import LoadInkml as LoadInkml
from los import Graph
import numpy as np
from tqdm import tqdm
import os
import SegmenterFeatureExtractor as FeatureExtractor


def create_model(X, y, model_filename, classifier_type):
    """
    Builds the classifier model

    :param X: Features
    :param y: Labels
    :param model_filename: the file where to save the model
    :param classifier_type: svm/rf
    :return:
    """
    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print('Training Classifier..')
    if classifier_type is 'svm':
        # creates SVM model and fits it on training samples
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X_train, y_train)
        model = clf
        # stores the classifier in pickle file
        with open(model_filename, 'wb') as model_file:
            pickle.dump(model, model_file)

        # predict for test samples
        y_pred = clf.predict(X_test)
        # predict for training samples
        y_pred_train = clf.predict(X_train)

    else:
        # creates Random Forest model and fits it on training samples
        rf = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=40, max_features=11)
        rf.fit(X_train, y_train)
        model = rf
        # stores the classifier in pickle file
        with open(model_filename, 'wb') as model_file:
            pickle.dump(model, model_file)

        # predict for test samples
        y_pred = rf.predict(X_test)
        # predict for training samples
        y_pred_train = rf.predict(X_train)

    # Print accuracy score
    print("Test Accuracy score:", metrics.accuracy_score(y_test, y_pred))
    print("Train Accuracy score:", metrics.accuracy_score(y_train, y_pred_train))


def get_train_data(filepath):
    """
    Get train data file
    :param filepath:input file path
    :return: None
    """
    filelist = []
    # get all inkml files from directory and sub-directories
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if os.path.splitext(file)[1] == '.inkml':
                filelist.append(os.path.join(root, file))

    X = []
    y = []

    for path in tqdm(filelist):
        # pre-processing
        data = LoadInkml.parse_inkml(path)
        data.traces()

        # Line of Sight
        los = Graph(data)

        directed_graph = los.get_directed_graph()

        sample_X, sample_y = FeatureExtractor.getAllFeatures(directed_graph, data)
        if sample_X:
            if len(X) == 0:
                X = sample_X
            else:
                # print(np.array(sample_X).shape)
                X = np.append(X, sample_X, axis=0)

            y.extend(sample_y)

    print(len(X), len(y))
    training_data = np.column_stack((X, y))
    # print(training_data.shape)
    with open('train_features.pkl', 'wb') as dataset_file:
        pickle.dump(training_data, dataset_file)
    print('Dataset stored at: train_features.pkl')

    return None


if __name__ == '__main__':
    filepath = 'data/inkml/extension'
    # get_train_data(filepath)

    # change to svm for random forest classifier
    classifier_type = 'rf'
    model_filename = classifier_type + '_merge_final.pkl'

    train_data_pkl = 'train_features.pkl'
    # X,y = GetTrainDataset.load_train_dataset()
    with open(train_data_pkl, 'rb') as pickle_file:
        dataset = pickle.load(pickle_file)

        X = dataset[:, :-1]
        y = dataset[:, -1]

    # create the classifier model
    create_model(X, y, model_filename, classifier_type)
