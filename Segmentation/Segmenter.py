"""runs Segmenter program for each sample from dataset.

@author: Anushree D
@author: Nishi P
@author: Omkar S
"""

import SegmenterFeatureExtractor as SegmenterFeatureExtractor
import pickle5 as pickle
import LoadInkml as LoadInkml
from los import Graph
import ExtractFeatures
import numpy as np
import os, sys


class MathExpression:
    def __init__(self, filename, merge_classifier, symbol_classifier):
        self.filename = filename
        self.symbols = []
        self.merge_classifier = merge_classifier
        self.symbol_classifier = symbol_classifier

        data = LoadInkml.parse_inkml(filename)
        data.traces()

        # Line of Sight
        los = Graph(data)
        # for key in los.edges:
        #     print (key,los.edges[key])
        directed_graph = los.get_directed_graph()

        # print(directed_graph)
        self.get_segmentations(directed_graph, data)

    def get_segmentations(self, directed_graph, data):
        X, y = SegmenterFeatureExtractor.getAllFeatures(directed_graph, data)

        if len(X) == 0:
            return

        y_pred = self.merge_classifier.predict(X)

        segmentations = []
        for i in range(len(directed_graph)):
            if y_pred[i] == '*':
                if len(segmentations) == 0:
                    segmentations.append(directed_graph[i])
                else:
                    found = False
                    for seg in segmentations:
                        if found:
                            break
                        if directed_graph[i][0] in seg:
                            if directed_graph[i][1] in seg:
                                found = True
                            else:
                                seg.append(directed_graph[i][1])
                                found = True
                        else:
                            if directed_graph[i][1] in seg:
                                seg.append(directed_graph[i][0])
                                found = True
                    if not found:
                        segmentations.append(directed_graph[i])

        for i in range(len(data.strokeID)):
            found = False
            for seg in segmentations:
                if data.strokeID[i] in seg:
                    found = True
            if not found:
                segmentations.append([data.strokeID[i]])

        symbols = []
        for seg in segmentations:
            strokes = []
            for stroke_id in seg:
                strokes.append(data.coordinates[stroke_id])
            if len(strokes) > 0:
                # extract feature for cluster
                features = ExtractFeatures.generate_features(strokes)
                features = np.array(features).reshape(1, -1)
                # predict class label for cluster
                y_pred = self.symbol_classifier.predict(features)

                sym_class = str(y_pred[0])
                sym_id = sym_class + '_' + str(seg[0])
                symbols.append(Symbol(sym_id, sym_class, seg))
        self.symbols = symbols

    def write_lgfile(self, directory):
        """
        writes the output to .lg file
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, os.path.splitext(os.path.basename(self.filename))[0] + '.lg')
        with open(filepath, 'w') as f:
            for symbol in self.symbols:
                f.write('O, ' + str(symbol.symbol_id) + ', ' + symbol.symbol_class + ', 1.0')
                for stroke_id in symbol.stroke_list:
                    f.write(', ' + str(stroke_id))
                f.write('\n')


class Symbol:
    """
    Symbol Class
    """
    def __init__(self, sym_id, sym_class, stroke_l):
        self.symbol_id = sym_id
        self.symbol_class = sym_class
        self.stroke_list = stroke_l


if __name__ == '__main__':
    model_pickle = 'rf_merge_final.pkl'
    # load the binary classifier model
    model = open(model_pickle, 'rb')
    merge_classifier = pickle.load(model)

    model_pickle = 'rf_recognize.pkl'
    # load the recognition classifier model
    model = open(model_pickle, 'rb')
    symbol_classifier = pickle.load(model)

    path = sys.argv[1]
    try:
        assert os.path.exists(path)
    except:
        print('Check path')
        sys.exit(0)
    # path = '/Users/anushree/PycharmProjects/PRecProject2/data/inkml/MfrDB/MfrDB0003.inkml'

    m = MathExpression(path, merge_classifier, symbol_classifier)
    m.write_lgfile('output')
