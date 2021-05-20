# Handwritten_Math_Symbol_Classification

Do the following for training and test dataset:
1. Run LoadInkmlFiles.py to load .inkml files and store in pickle file
2. Run ExtractFeatures.py to extract features from dataset and store in pickle file

Run TrainClassifier.py to get rf or svm classifier for symbol classification.

To run the SVM classifier:
python3 ClassifyTestSymbols.py svm dir

To run the Random Forest classifier:
python3 ClassifyTestSymbols.py rf dir
