"""
This program implements agglomerative clustering with geometric overlap parser system.

@author: Anushree Das
@author: Gautam Gadipudi
@author: Nishi Parameshwara
@author: Raj Bhensadadia
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import pickle5 as pickle
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance
from tqdm import tqdm
import ExtractFeatures
import sys

class K_Agglomerative_Model:
    """
    Stores clusters formed from agglomerative clustering gor a given k
    """
    def __init__(self,k,clusters):
        self.k = k
        self.clusters = clusters

    def get_info(self):
        """
        Returns list of list of coordinates of each stroke in each cluster
        """
        coods = []
        strokes = []
        for cluster in self.clusters:
            single_cluster_coods = []
            single_cluster_strokes = []
            for stroke in self.clusters[cluster]:
                single_cluster_coods.extend(stroke.coods)
                single_cluster_strokes.append(stroke)
            coods.append(single_cluster_coods)
            strokes.append(single_cluster_strokes)

        return coods, strokes

class Expression:
    def __init__(self, file):
        self.filename = file
        self.strokes = self.get_strokes()
        self.symbols = []
        self.relationships = []
        self.k_agglomerative_clusters = []

    def agglomerative(self):
        k_agglomerative_clusters = []

        for k in range(len(self.strokes),0,-1):
            clusters = {}
            # Initialize clusters
            for i in range(k):
                clusters[i] = []

            if k == len(self.strokes):
                # when k = number of strokes
                # each stroke will belong to a seperate cluster
                i = 0
                for stroke in self.strokes:
                    clusters[i].append(stroke)
                    i += 1
                k_agglomerative_clusters.append(K_Agglomerative_Model(k,clusters))
            else:

                coods, strokes = k_agglomerative_clusters[-1].get_info()

                # initialize distance matrix
                dist = [[np.Inf for _ in range(len(coods))] for _ in range(len(coods))]

                # get minimum distance between two clusters
                for i in range(len(coods)-1):
                    for j in range(i+1,len(coods)):
                        dist[i][j] = self.find_min(coods[i],coods[j])

                # find indices of clusters which are closest
                min_index = np.where(dist == np.amin(dist))
                closest_clusters = list(zip(min_index[0], min_index[1]))[0]

                index = 0
                # assign all old clusters which aren't needed to be merged to current clusters model
                for i in range(len(coods)):
                    if i not in closest_clusters:
                        clusters[index].extend(strokes[i])
                        index += 1

                # merge two clusters and add to current clusters model
                for i in closest_clusters:
                    for stroke in strokes[i]:
                        clusters[index].append(stroke)

                k_agglomerative_clusters.append(K_Agglomerative_Model(k, clusters))

        self.k_agglomerative_clusters = k_agglomerative_clusters

    def recognition(self, clustering_model):
        """
        Select the segmentation with the smallest number of symbols
        that produces the highest geometric mean over the class probabilities
        produced by the random forest classifier from Project 1
        """
        self.symbols = []
        if len(clustering_model) > 0:
            # load random forest classifier
            rf_model = open('rf.pkl', 'rb')
            classifier = pickle.load(rf_model)

            selected_k_index = 0
            highest_gmean = 0

            for i in range(len(clustering_model)):
                model = clustering_model[i]
                features = []

                # extract features for each cluster
                for cluster in model.clusters.keys():
                    strokes = []
                    for stroke in model.clusters[cluster]:
                        strokes.append(stroke.coods)
                    if len(strokes) > 0:
                        features.append(ExtractFeatures.generate_features(strokes))

                # calculate class probabilities for each cluster
                class_probabilities = classifier.predict_proba(features)
                # get geometric mean
                g_mean = self.geometric_mean(class_probabilities)

                # select k with highest geometric mean
                if g_mean > highest_gmean:
                    selected_k_index = i
                    highest_gmean = g_mean

            self.selected_k = selected_k_index

            # Store the clusters for strokes for the selected k
            # along with symbol class obtained from the classifier
            for cluster in clustering_model[selected_k_index].clusters.keys():
                strokes = []
                strokeid_list = []
                for stroke in clustering_model[selected_k_index].clusters[cluster]:
                    strokes.append(stroke.coods)
                    strokeid_list.append(stroke.id)
                if len(strokes) > 0:
                    # extract feature for cluster
                    features = ExtractFeatures.generate_features(strokes)
                    features = np.array(features).reshape(1, -1)
                    # predict class label for cluster
                    y_pred = classifier.predict(features)

                    sym_class = str(y_pred[0])
                    sym_id = sym_class + '_' + str(strokeid_list[0])
                    self.symbols.append(Symbol(sym_id, sym_class, strokeid_list))

    def get_strokes(self):
        try:
            tree = ET.parse(self.filename, ET.XMLParser(encoding='utf-8'))
        except Exception:
            try:
                tree = ET.parse(self.filename, ET.XMLParser(encoding='iso-8859-1'))
            except Exception:
                return []

        root = tree.getroot()
        doc_namespace = "{http://www.w3.org/2003/InkML}"

        # extract stroke coordinates
        strokes = [Stroke(trace_tag.get('id'),
                          [[round(float(axis_coord))
                            if float(axis_coord).is_integer()
                            else round(float(axis_coord) * 10000)
                            for axis_coord in coord[1:].split(' ')]
                           if coord.startswith(' ')
                           else [round(float(axis_coord))
                                 if float(axis_coord).is_integer()
                                 else round(float(axis_coord) * 10000)
                                 for axis_coord in coord.split(' ')]
                           for coord in (trace_tag.text).replace('\n', '').split(',')])
                   for trace_tag in root.findall(doc_namespace + 'trace')]

        return strokes

    def geometric_mean(self,arr):
        """
        Calculates the geometric mean over the class probabilities
        """
        n = len(arr)
        prod = 1
        for row in arr:
            prod = prod * max(row)
        g_mean = prod ** (1 / n)
        return g_mean

    def find_min(self,list1,list2):
        # Returns minimum distance between all coordinates of two clusters
        return min(distance.cdist(list1,list2).min(axis=1))

    def min_spanning_tree(self,clustering_model):

        coods, strokes = clustering_model[self.selected_k].get_info()

        # initialize distance matrix
        dist = [[np.Inf for _ in range(len(coods))] for _ in range(len(coods))]
        rel = [[np.Inf for _ in range(len(coods))] for _ in range(len(coods))]

        # get minimum distance between two clusters
        for i in range(len(coods) - 1):
            for j in range(i + 1, len(coods)):
                dist[i][j] = self.find_min(coods[i], coods[j])
                rel[i][j] = self.calculate_relationship(coods[i],coods[j])

        X = csr_matrix(dist)
        # build minimum spanning tree
        Tcsr = minimum_spanning_tree(X)
        span_tree = Tcsr.toarray()

        # add relationship between symbols based on spanning tree edges
        for i in range(len(self.symbols) - 1):
            for j in range(i + 1, len(self.symbols)):
                if span_tree[i][j] != 0:
                    r = rel[i][j]
                    # if relationship is Left, flip symbols and set relationship as Right
                    if r == 'Left':
                        self.relationships.append(Relationship(self.symbols[j].symbol_id,
                                                               self.symbols[i].symbol_id, 'Right'))
                    else:
                        self.relationships.append(Relationship(self.symbols[i].symbol_id,
                                                           self.symbols[j].symbol_id, r))

    def bounding_box(self,stroke_traces):
        x = stroke_traces[:, 0]
        y = stroke_traces[:, 1]
        bbox = {}
        bbox['xmin'] = x.min()
        bbox['ymin'] = y.min()
        bbox['xmax'] = x.max()
        bbox['ymax'] = y.max()

        return bbox

    def get_overlap_param(self, stroke1, stroke2):

        stroke1 = np.array(stroke1)
        stroke2 = np.array(stroke2)

        # bounding box area difference
        bbox_stroke1 = self.bounding_box(stroke1)
        bbox_stroke2 = self.bounding_box(stroke2)

        # overlapping area of bounding boxes
        x_left = max(bbox_stroke1['xmin'], bbox_stroke2['xmin'])

        # find if first stroke is on right or left
        if x_left == bbox_stroke1['xmin']:
            hdir = 'r'
        else:
            hdir = 'l'

        y_top = max(bbox_stroke1['ymin'], bbox_stroke2['ymin'])

        # find if first stroke is above second symbol or below
        if y_top == bbox_stroke1['ymin']:
            vdir = 't'
        else:
            vdir = 'b'

        x_right = min(bbox_stroke1['xmax'], bbox_stroke2['xmax'])
        y_bottom = min(bbox_stroke1['ymax'], bbox_stroke2['ymax'])


        horizontal_overlapping_val = max((x_right - x_left),0)
        vertical_overlapping_val = max((y_bottom - y_top),0)

        hflag = False

        if horizontal_overlapping_val > ((bbox_stroke1['xmax']-bbox_stroke1['xmin'])*(3/4)) or \
                    horizontal_overlapping_val > ((bbox_stroke2['xmax']-bbox_stroke2['xmin'])*(3/4)):
            hflag = True

        vflag = False
        sup = False
        sub = False

        if vertical_overlapping_val > 0:
            if bbox_stroke2['ymin'] > bbox_stroke1['ymin']+((bbox_stroke1['ymax']-bbox_stroke1['ymin'])*(3/4)):
                sup = True
            elif bbox_stroke2['ymax'] < bbox_stroke1['ymin']+((bbox_stroke1['ymax']-bbox_stroke1['ymin'])*(1/4)):
                sub = True
            if vertical_overlapping_val > ((bbox_stroke1['ymax']-bbox_stroke1['ymin'])*(3/4)) or \
                    vertical_overlapping_val > ((bbox_stroke2['ymax']-bbox_stroke2['ymin'])*(3/4)):
                vflag = True

        return hflag,vflag,hdir,vdir,sup,sub


    def calculate_relationship(self,traces_A, traces_B):

        hflag,vflag,hdir,vdir,sup,sub = self.get_overlap_param(traces_A,traces_B)

        if hflag and vflag:
            return 'Inside'

        if not hflag and vflag:
            if hdir == 'l':
                return 'Right'
            else:
                return 'Left'

        if hflag and not vflag:
            if vdir == 't':
                return 'Below'
            else:
                return 'Above'

        if sup:
            return 'Sup'
        if sub:
            return 'Sub'

        return '_'


    def ac_mst(self):
        # apply k agglomerative clustering
        if len(self.k_agglomerative_clusters) == 0:
            self.agglomerative()

        # find most suitable clusters
        self.recognition(self.k_agglomerative_clusters)

        if len(self.strokes) != 0:

            # add relationship between symbols based on min spanning tree edges
            self.min_spanning_tree(self.k_agglomerative_clusters)

        # write all output symbols created to .lg file
        self.write_lgfile('result')


    def write_lgfile(self, directory):
        # writes the output to .lg file
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, os.path.splitext(os.path.basename(self.filename))[0] + '.lg')
        with open(filepath, 'w') as f:
            for symbol in self.symbols:
                f.write('O, ' + str(symbol.symbol_id) + ', ' + symbol.symbol_class + ', 1.0')
                for stroke_id in symbol.stroke_list:
                    f.write(', ' + str(stroke_id))
                f.write('\n')
            for relationship in self.relationships:
                f.write('R, '+relationship.symbol_id_1+', '+str(relationship.symbol_id_2)+', '+relationship.symbol_relation+', 1.0')
                f.write('\n')


# data structure to store symbol information
class Symbol:
    def __init__(self,sym_id,sym_class,stroke_l):
        self.symbol_id = sym_id
        self.symbol_class = sym_class
        self.stroke_list = stroke_l

# data structure to store relationship information
class Relationship:
    def __init__(self,sym_id_1,sym_id_2,sym_relation):
        self.symbol_id_1 = sym_id_1
        self.symbol_id_2 = sym_id_2
        self.symbol_relation = sym_relation

# data structure to store stroke information
class Stroke:
    def __init__(self,id,coods):
        self.coods = [[row[0],row[1]] for row in coods]
        self.id = id

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 a4.py [path to inkml directory] ')
        exit(0)
    inkml_path = sys.argv[1]

    if not os.path.exists(inkml_path):
        print("Path doesn't exist")
        exit(0)

    filelist =[]

    # get all inkml files from directory and sub-directories
    for root, dirs, files in os.walk(inkml_path):
        for file in files:
            if os.path.splitext(file)[1] == '.inkml':
                filelist.append(os.path.join(root, file))

    for files in tqdm(filelist):
        e = Expression(files)
        e.ac_mst()



