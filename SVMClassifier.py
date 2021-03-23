import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split






def main():
    data = '/home/omkarsarde/PycharmProjects/Datasets/features.pkl'
    df = pd.read_pickle(data)
    X = df.iloc[:,1:4096]
    y = df.iloc[:,4096]
    fnames = df.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=151, test_size=0.3)
    print('Sanitized Data')
    svm_model_linear = SVC(kernel='linear', C=1,cache_size=1000)
    print('Declared Classifier')
    svm_model_linear.fit(X_train, y_train)
    print('Fit Classifier')
    svm_predictions = svm_model_linear.predict(X_test)
    print('Completed Prediction')
    # model accuracy for X_test
    accuracy = svm_model_linear.score(X_test, y_test)
    print(accuracy,' ACCURACY ')

    acc = accuracy_score(y_test,svm_predictions)
    print(acc, 'ACC score')
    # creating a confusion matrix
    cm = confusion_matrix(y_test, svm_predictions)
    print(cm,'CM')

if __name__ == '__main__':
    main()