import pickle5 as pickle
from ExtractFeatures import *
from sklearn import metrics
import pandas as pd

with open('test_data.pkl', 'rb') as pickle_file:
    test_data = pickle.load(pickle_file)
    X_test, y_test = feature_extraction(test_data)
svm_pickle = open('svm.pkl', 'rb')
classifier = pickle.load(svm_pickle)
y_pred = classifier.predict(X_test)

UIs = []
for UI in test_data:
    UIs.append(UI)

df = pd.DataFrame({'UI':UIs,'y_pred':y_pred})
df.to_csv('resultsvm.csv',index=False,header=False)