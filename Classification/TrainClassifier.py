'''
This program builds a SVM and Randomn Forest classifier and trains it using the training dataset features

@author: Anushree D
@author: Nishi P
@author: Omkar S
'''
import pickle5 as pickle
from ExtractFeatures import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def create_model(X,y,model_filename,classifier_type):
    """
    builds the classifier model

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
        rf = RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_split=3)
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
    print("Test Accuracy score:",metrics.accuracy_score(y_test, y_pred))
    print("Train Accuracy score:", metrics.accuracy_score(y_train, y_pred_train))

if __name__ =='__main__':
    # change to svm for random forest classifier
    classifier_type = 'rf'
    model_filename = classifier_type+'.pkl'
    # get features.pkl from LoadInkmlFiles.py
    features_filename = 'features.pkl'

    # Extract X and y from the dataset dictionary
    with open(features_filename, 'rb') as pickle_file:
        dataframe = pickle.load(pickle_file)
        X = dataframe[dataframe.columns[:-1]].to_numpy()
        y = dataframe[dataframe.columns[-1]].to_numpy()

    # create the classifier model
    create_model(X,y,model_filename,classifier_type)