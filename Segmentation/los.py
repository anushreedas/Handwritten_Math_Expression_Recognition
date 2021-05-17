import os
from scipy.spatial import ConvexHull
import numpy as np
from math import *
import matplotlib.pyplot as plt
import LoadInkml as LoadInkml

class Node(object):
    """
    Los graph node
    """
    __slots__ = ('stroke_id', 'stroke_traces', 'bbox')

    def __init__(self, *args):
        """
        Default initialization
        :param args: stroke id trace bounding box
        """

        if len(args) == 2:
            self.stroke_id = args[0]
            self.stroke_traces = args[1]
            self.bbox = self.bbox_calculation()

    def bbox_calculation(self):
        """
        Calculate Bounding box
        :return: center of bounding box
        """
        x = self.stroke_traces[:, 0]
        y = self.stroke_traces[:, 1]
        bb_center = ([x.min(), y.min()][0] + ((x.max() - x.min()) / 2)), \
                    ([x.min(), y.min()][1] + ((y.max() - y.min()) / 2))
        return tuple(bb_center)


class Graph(object):
    """
    Los Graph Class
    """
    __slots__ = ('nodes', 'edges', 'file_inst')

    def __init__(self, *args):
        """
        Default initialization
        :param args: nodes edges currentfile
        """
        if len(args) == 1:
            self.file_inst = args[0]
            self.nodes = dict()  # {'1': array([[x1,y1], [x2,y2] ...])}

            for s in self.file_inst.strokeID:
                self.nodes[s] = Node(s, self.file_inst.coordinates[s])

            self.edges = dict()  # {'1': ['2', '3', '7'], '2':['1']}
            self.graph()

    def graph(self):
        """
        Create the graph
        :return:
        """
        strokeExp = self.file_inst.strokeID
        for stroke in strokeExp:
            node1 = self.nodes[stroke]  # np array
            bb_box = node1.bbox  # bounding box center
            angles = np.ones(360)  # all are unblocked

            # sort strokes by increasing distance from s1
            ascStroke = sorted((set(strokeExp) - set(stroke)), key=lambda x1: hypot(bb_box[0] - self.nodes[x1].bbox[0],
                                                                                    bb_box[1] - self.nodes[x1].bbox[1]))

            # blocked by remaining strokes
            for stroke in ascStroke:
                node2 = self.nodes[stroke]

                theta_min = inf
                theta_max = -inf

                for n in self.convex_hull(node2.stroke_traces):

                    x, y = n  # candidate stroke
                    c0, c1 = bb_box  # bounding box center coordinates
                    plt.plot([c0, x], [c1, y], 'bo-')

                    # find the angle between vector w and a horizontal vector (1,0)
                    if y >= c1:

                        theta = degrees(acos((x - c0, y - c1)[0] / sqrt((x - c0, y - c1)[0] ** 2 +
                                (x - c0, y - c1)[1] ** 2) * sqrt((1, 0)[0] ** 2 + (1, 0)[1] ** 2)))
                        if isnan(theta):
                            theta = 0.0
                        else:
                            theta = int(theta)
                    else:
                        theta = 360 - int(degrees(acos((x - c0, y - c1)[0] / sqrt((x - c0, y - c1)[0] ** 2 +
                                (x - c0, y - c1)[1] ** 2) * sqrt((1, 0)[0] ** 2 + (1, 0)[1] ** 2))))

                    theta_min = min(theta_min, 180 - theta)
                    theta_max = max(theta_max, 360 - theta)

                hull = (int(theta_min), int(theta_max))  # hull interval
                if np.any(angles[hull[0]:hull[1]] == 1):
                    if node1.stroke_id in self.edges:
                        self.edges[node1.stroke_id].add(node2.stroke_id)
                    else:
                        self.edges[node1.stroke_id] = set(node2.stroke_id)

                    if node2.stroke_id in self.edges:
                        self.edges[node2.stroke_id].add(node1.stroke_id)
                    else:
                        self.edges[node2.stroke_id] = set(node1.stroke_id)

                angles[hull[0]:hull[1]] = 0

    def convex_hull(self, coordinates):
        """
        Check and get points on the convex hull
        :return: convex hull array
        """
        ch = ConvexHull(coordinates, False, 'QJ')
        chull = list()

        for vertex in ch.vertices:
            chull.append(coordinates[vertex])
        return np.array(chull)

    # returns lists of directed egdes in los graph
    def get_directed_graph(self):
        """
        Convert the graph to a directed graph
        :return:
        """
        visited = []
        edges = []
        for key in self.edges:
            if key not in visited:
                visited.append(key)
            for value in self.edges[key]:
                if value not in visited:
                    edges.append([key, value])
        return edges





