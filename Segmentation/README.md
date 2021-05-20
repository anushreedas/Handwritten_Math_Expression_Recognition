Program that reads stroke data from a CROHME.inkml file, segments strokes into symbols, classifies the symbols, and then produces a label graph (.lg)file in Object-Relationship Format as output. The classifier used is the random forest classifier from https://github.com/anushreedas/Handwritten_Math_Expression_Recognition

Authors: Anushree D, Nishi P, Omkar S

Project 2 Math Symbol Segmentation and Classification

*****************Instructions*****************

Install requirements using:
    $ pip install -r requirements.txt

Run the DatasetSegmenter.py over the training / test data
    $ python3 DataSegmenter.py ./home/user/inputfolder

##Special Note: Do not include trailing / at the last level of the inputfolder
