import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def main():
    data = '/home/omkarsarde/PycharmProjects/Datasets/features.pkl'
    df = pd.read_pickle(data)
    X = df.iloc[:, 1:4096]
    y = df.iloc[:, 4096]
    fnames = df.iloc[:, -1]
    df = None
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=151, test_size=0.3)
    X, y = None, None
    # print('Sanitized Data')
    # clf = SVC(kernel='linear', decision_function_shape='ovr')
    # print('Declared SVM Classifier')
    # clf.fit(X_train, y_train)
    # print('Fit SVM Classifier')
    # svm_predictions = clf.predict(X_test)
    # print('Completed SVM Prediction')
    # # model accuracy for X_test
    # # accuracy = clf.score(X_test, y_test)
    # # print(accuracy, ' SVM ACCURACY ')
    #
    # acc = accuracy_score(y_test, svm_predictions)
    # print(acc, 'ACC score SVM')
    # # creating a confusion matrix
    # cm = confusion_matrix(y_test, svm_predictions)
    # print(cm, 'CM SVM')

    clf = RandomForestClassifier(criterion='gini', random_state=151, bootstrap=False, n_jobs=-1, n_estimators=425)
    print('Declared RF Classifier')
    clf.fit(X_train, y_train)
    print('Fit RF Classifier')
    rf_predictions = clf.predict(X_test)
    print('Completed RF Prediction')
    # model accuracy for X_test
    # accuracy = clf.score(X_test, y_test)
    # print(accuracy, ' RF ACCURACY ')

    acc = accuracy_score(y_test, rf_predictions)
    print(acc, 'ACC score RF')
    # creating a confusion matrix
    cm = confusion_matrix(y_test, rf_predictions)
    print(cm, 'CM RF')


if __name__ == '__main__':
    main()
