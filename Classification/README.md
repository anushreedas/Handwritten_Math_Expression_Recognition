# Handwritten Math Symbol Classification

Python program to classify individual handwritten mathematical symbolsfrom the CROHME isolated symbol competition data set. 
For classification, we used 
1) a Support Vector Machine, and 
2) a decision tree-based model(e.g., random forest, gradient boosted trees) 
provided by the python scitkit-learn library: https://scikit-learn.org/stable

Do the following for training and test dataset:
1. Run LoadInkmlFiles.py to load .inkml files and store in pickle file
2. Run ExtractFeatures.py to extract features from dataset and store in pickle file

Run TrainClassifier.py to get rf or svm classifier for symbol classification.

To run the SVM classifier:
python3 ClassifyTestSymbols.py svm dir

To run the Random Forest classifier:
python3 ClassifyTestSymbols.py rf dir
