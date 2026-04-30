# Handwritten Math Expression Recognition
End-to-end pipeline for recognizing online handwritten mathematical expressions from InkML stroke data.

This project implements segmentation, classification, and parsing inspired by:

> Hu, L., & Zanibbi, R. (2011). HMM-Based Recognition of Online Handwritten Mathematical Symbols Using Segmental K-Means Initialization and a Modified Pen-Up/Down Feature.

## Overview
This system takes handwritten math expressions as input (`.inkml`) and performs:
1. Segmentation – grouping strokes into symbols
2. Classification – identifying symbols
3. Parsing – determining spatial relationships

Final output: structured representation of the expression (`.lg` format)

## Pipeline
```
InkML → Preprocessing → Segmentation → Feature Extraction → Classification → Parsing → Output
```

## Repository Structure
```
Handwritten_Math_Expression_Recognition/

├── Classification/
│   ├── ClassifyTestSymbols.py
│   ├── ExtractFeatures.py
│   ├── LoadInkmlFiles.py
│   ├── TrainClassifier.py
│   ├── svm.pkl                  # Trained classifier
│   ├── project1.pdf             # Classification report
│   ├── requirements.txt
│   └── README.md
│
├── Segmentation/
│   ├── DatasetSegmenter.py
│   ├── Segmenter.py
│   ├── SegmenterFeatureExtractor.py
│   ├── ExtractFeatures.py
│   ├── LoadInkml.py
│   ├── TrainClassifier.py
│   ├── los.py                   # Line-of-Sight graph implementation
│   ├── rf_merge.pkl             # Trained merge classifier
│   ├── CSCI_737_Project_2.pdf   # Segmentation report
│   ├── requirements.txt
│   └── README.md
│
├── Parsing/
│   ├── p3.py                    # Parsing pipeline
│   ├── ExtractFeatures.py
│   ├── Summary.txt
│   ├── PRec_Team5_Project3.pdf  # Parsing report
│   └── README.md
│
└── README.md
```

## Component Details
### ✂️ Segmentation
* Implemented using Line-of-Sight (LoS) graphs (`los.py`)
* Extracts:
  * Geometric features (distance, overlap, offsets)
  * Shape context features
* Uses a Random Forest classifier (`rf_merge.pkl`) to decide whether strokes should be merged

Output: grouped strokes representing symbols

### 🔢 Classification
* Feature extraction (ExtractFeatures.py):
  * Normalized y-coordinate
  * Vicinity slope (α)
  * Curvature (β)
* Classifiers:
  * SVM (svm.pkl)
  * Random Forest (used during experiments)
* Scripts:
  * TrainClassifier.py → trains model
  * ClassifyTestSymbols.py → predicts symbols

Output: labeled symbols

### 🧩 Parsing
* Implemented in p3.py
* Steps:
  * Compute distance matrix between symbols
  * Build Minimum Spanning Tree (MST)
  * Assign relationships based on geometric overlap

Relationships include:

* Left / Right
* Above / Below
* Inside
* Superscript / Subscript

Output: .lg file describing structure

## How to Run
1. Segmentation
```
cd Segmentation

# Install dependencies
pip install -r requirements.txt

# Run segmentation on dataset
python3 DatasetSegmenter.py <path_to_input_folder>
```
2. Classification
```
cd Classification

# Install dependencies
pip install -r requirements.txt

# Load InkML Data
python3 LoadInkmlFiles.py

# Extract Features
python3 ExtractFeatures.py

# Train Classifier
python3 TrainClassifier.py

# Run Classification
# SVM Classifier
python3 ClassifyTestSymbols.py svm <path_to_test_directory>

# Random Forest Classifier
python3 ClassifyTestSymbols.py rf <path_to_test_directory>
```

3. Parsing
```
cd Parsing

python3 p3.py <path_to_inkml_directory>
```

## Results
### Segmentation + Classification
* F1 Score: 58.38% (segmentation)
* F1 Score: 33.72% (combined)
### Parsing
* F1 Score: ~40.25%

## Limitations
* Errors propagate across pipeline stages
* Segmentation quality strongly impacts final results
* Overlap-based parsing is sensitive to thresholds
* Multi-stroke symbols (+, x, =) are frequently misclassified

## Tech Stack
* Python
* NumPy, Pandas
* Scikit-learn
* XML parsing (InkML)

## Reports
* Classification → Classification/project1.pdf
* Segmentation → Segmentation/CSCI_737_Project_2.pdf
* Parsing → Parsing/PRec_Team5_Project3.pdf

## Future Work
* Replace feature engineering with deep learning models
* Improve segmentation using graph neural networks
* Learn parsing relationships instead of rule-based logic
* Build end-to-end trainable architecture

## Author

**Anushree Das**
[LinkedIn](https://linkedin.com/in/anushree-s-das) · [GitHub](https://github.com/anushreedas) · [Medium](https://medium.com/@anushree-das)
