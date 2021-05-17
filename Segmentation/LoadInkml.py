"""
This program loads data from Inkml files as object of class parse_inkml
and preprocesses the strokes(i.e., resampling, removing duplicates anf normalizing y coordinates)

@author: Anushree D
@author: Nishi P
@author: Omkar S
"""

import numpy as np
import math
import scipy.interpolate as sp
import xml.etree.ElementTree as ET

# data structure to store symbol information obtained from the ground truth
class Symbol(object):
    __slots__ = ('id', 'label', 'stroke')

    def __init__(self, *args):
        """
        Initialization for Symbol class
        :param args: id label stroke
        """
        if len(args) == 3:
            self.id = args[0]
            self.label = args[1]
            self.stroke = args[2]
        else:
            self.id = "none"
            self.label = ""
            self.stroke = set([])

# this class parses an inkml file and stores all the information obtained from it
# it also preprocessses the stroke coordinates
class parse_inkml(object):
    """
    This class parses an inkml file and stores all the information obtained from it
    Tt also preprocessses the stroke coordinates
    """
    __slots__ = ('file', 'coordinates', 'strokeID', 'strokeClass', 'expression', 'UI')
    namespace = {'1': 'http://www.w3.org/2003/InkML', '2': 'http://www.w3.org/XML/1998/namespace'}

    def __init__(self, *args):
        """
        Default initialization
        :param args: file coordinates strokeId strokeClass expression UI
        """
        self.file = None
        self.coordinates = {}
        self.strokeID = []
        self.strokeClass = {}
        self.expression = ""
        self.UI = ""
        if len(args) == 1:
            self.file = args[0]
            self.readFile()

    # returns ground truth segmentations as list of list of stroke ids
    def getGT(self):
        """
        Get the ground truth
        :return: gt array
        """
        gt = []
        for symbol_id in self.strokeClass:
            temp = sorted(list(self.strokeClass[symbol_id].stroke))
            gt.append(temp)
        return gt

    # reads inkml file
    def readFile(self):
        """
        Read the inkml file
        :return:
        """
        try:
            tree = ET.parse(self.file, ET.XMLParser(encoding='utf-8'))
        except Exception:
            try:
                tree = ET.parse(self.file, ET.XMLParser(encoding='iso-8859-1'))
            except Exception:
                # print('Error occured while reading inkml file:',Exception)
                return
        root = tree.getroot()

        for info in root.findall('1:annotation', namespaces=parse_inkml.namespace):
            if 'type' in info.attrib:
                if info.attrib['type'] == 'truth':
                    self.expression = info.text.strip()
                if info.attrib['type'] == 'UI':
                    self.UI = info.text.strip()
        # get strokes
        for points in root.findall('1:trace', namespaces=parse_inkml.namespace):
            self.coordinates[points.attrib['id']] = points.text.strip()
            self.strokeID.append(points.attrib['id'])

        # get segmentations
        symbolClass = root.find('1:traceGroup', namespaces=parse_inkml.namespace)
        if symbolClass is None or len(symbolClass) == 0:
            return

        for sym in (symbolClass.iterfind('1:traceGroup', namespaces=parse_inkml.namespace)):
            id = sym.attrib[self.correct('2', 'id')]
            label = sym.find('1:annotation', namespaces=parse_inkml.namespace).text
            traces = set([])

            for i in sym.findall('1:traceView', namespaces=parse_inkml.namespace):
                traces.add(i.attrib['traceDataRef'])
            self.strokeClass[id] = Symbol(id, label, traces)

    def correct(self, index, value):
        return '{' + parse_inkml.namespace[index] + '}' + value

    def duplicate_point_filtering(self,stroke):
        """
        Remove redundant and repeated points from the stroke information
        :param strokes_info: stroke information
        :return: pre-processed and updated stroke information
        """
        x = stroke[:, 0]
        y = stroke[:, 1]
        x_non_duplicate = []
        y_non_duplicate = []
        non_duplicates = set()
        for point in range(len(x)):
            coordinate = (x[point], y[point])
            if coordinate not in non_duplicates:
                x_non_duplicate.append(coordinate[0])
                y_non_duplicate.append(coordinate[1])
                non_duplicates.add(coordinate)

        stroke = np.column_stack((x_non_duplicate, y_non_duplicate))
        return stroke

    def y_normalization(self,stroke):
        """
        Normalize y-axis coordinate
        :param strokes_info: stroke information
        :return: pre-processed and updated stroke information
        """
        minimum_x_coordinate = math.inf
        maximum_x_coordinate = 0
        minimum_y_coordinate = math.inf
        maximum_y_coordinate = 0

        x = stroke[:, 0]
        y = stroke[:, 1]
        maximum_y_coordinate = max(max(y), maximum_y_coordinate)
        minimum_y_coordinate = min(min(y), minimum_y_coordinate)
        maximum_x_coordinate = max(max(x), maximum_x_coordinate)
        minimum_x_coordinate = min(min(x), minimum_x_coordinate)

        if maximum_y_coordinate == minimum_y_coordinate:
            maximum_y_coordinate += 100  # can be any positive number
            minimum_y_coordinate -= 100

        x = stroke[:, 0]
        y = stroke[:, 1]
        x_normal = []
        y_normal = []

        for i in range(len(x)):
            y_val = (y[i] - minimum_y_coordinate) / (maximum_y_coordinate - minimum_y_coordinate)
            x_val = (x[i] - minimum_x_coordinate) * (1 / (maximum_y_coordinate - minimum_y_coordinate))
            x_normal.append(x_val)
            y_normal.append(y_val)

        stroke = np.column_stack((x_normal, y_normal))
        return stroke

    def resampling(self,stroke):
        """
        Sample 10 points for each stroke.
        :param strokes_info: stroke information
        :return: pre-processed and updated stroke information
        """
        stroke = self.duplicate_point_filtering(stroke)
        sample_points = 10
        x = stroke[:, 0]
        y = stroke[:, 1]

        # if 3 or less points on this stroke, generate more points
        if len(x) < 4:
            x_resampled = []
            y_resampled = []

            if len(x) == 1:
                x1, y1 = x[0], y[0]
                x2, y2 = x[0] + 1, y[0] + 1

                x_resampled = np.linspace(x1, x2, max(4, sample_points))
                y_resampled = np.linspace(y1, y2, max(4, sample_points))
            else:
                # this stroke has atleast 2 points
                for i in range(1, len(x)):
                    # add 4 points between every consecutive points
                    x1, y1 = x[i - 1], y[i - 1]
                    x2, y2 = x[i], y[i]

                    x_pts = np.linspace(x1, x2, 4)
                    y_pts = np.linspace(y1, y2, 4)

                    # remove the intersection point of segments
                    if i < len(x) - 1:
                        x_pts = x_pts[:-1]  # take all points except the last one
                        y_pts = y_pts[:-1]  # take all points except the last one
                    x_resampled.extend(x_pts.tolist())
                    y_resampled.extend(y_pts.tolist())

            x, y = x_resampled, y_resampled

        control_pts = np.vstack([x, y])
        # Bspline of degree 3, make sure m > k (i.e minimum 4 points are provided)
        tck, u = sp.splprep(control_pts, k=3)
        sample_at = np.linspace(u.min(), u.max(), sample_points)
        # print(sample_at)
        x_resampled, y_resampled = sp.splev(sample_at, tck)

        stroke = np.column_stack((x_resampled, y_resampled))
        return stroke

    def traces(self):
        """
        Populate the coordinate array for every stroke
        :return:
        """
        for stroke in self.coordinates:
            strokeGrp = [traceValue.strip() for traceValue in self.coordinates[stroke].split(",")]
            coordinates = []

            for coord in strokeGrp:
                coord = coord.split(" ")
                if len(coord) > 1:
                    x = float(coord[0])
                    y = float(coord[1])
                else:
                    x = 0.0
                    y = 0.0
                coordinates.append([x, y])
            coordinates = self.y_normalization(np.array(coordinates))
            coordinates = self.resampling(coordinates)
            self.coordinates[stroke] = np.array(coordinates)


