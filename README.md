# Handwritten_Math_Symbol_Classification

## Classification:
Python program to classify individual handwritten mathematical symbolsfrom the CROHME isolated symbol competition data set. 
For classification, we used 
1) a Support Vector Machine, and 
2) a decision tree-based model(e.g., random forest, gradient boosted trees) 
provided by the python scitkit-learn library: https://scikit-learn.org/stable

## Segmentation
Program that reads stroke data from a CROHME.inkml file, segments strokes into symbols, classifies the symbols, and then produces a label graph (.lg)file inObject-Relationship Formatas output. 

## Parsing
A system that takes CROHME stroke data, and produces a Symbol Layout Tree (SLT) as output. Such a representation can be translated directly to MathML, LATEX, andother representations for the appearance of math expressions.
