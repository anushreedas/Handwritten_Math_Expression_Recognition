import pickle5 as pickle
from ExtractFeatures import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

print('Extracting Feature..')
# train_data = load_inkml_files('trainingSymbols')
# with open('training_data.pkl', 'rb') as pickle_file:
#     train_data = pickle.load(pickle_file)
#     X, y = feature_extraction(train_data)

# data = np.column_stack((X,y))
# print(data.shape)
# df = pd.Dataframe(data)
# df.to_pickle('features.pkl',index= False)

with open('features.pkl', 'rb') as pickle_file:
    dataframe = pickle.load(pickle_file)
    X = dataframe[dataframe.columns[:-1]].to_numpy()
    y = dataframe[dataframe.columns[-1]].to_numpy()
    # print(X.shape, y.shape)


print('Training Classifier..')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# clf.fit(X, y)
# model = clf
#
# with open('svm.pkl', 'wb') as model_file:
#     pickle.dump(model, model_file)


rf = RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_split=3)
rf.fit(X_train, y_train)
model = rf

with open('rf.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

y_pred = rf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))