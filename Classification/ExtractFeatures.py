"""
This program does pre-processes the stroke information from symbols
and extracts features from them.

@author: Anushree D
@author: Nishi P
@author: Omkar S
"""

import math
import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sp
import pandas as pd


def feature_extraction(train_data):
    """
    Pre-processes the stroke information and extracts features from them.
    :param train_data: training dataset for feature extraction
    :return: feature array X and labels
    """
    X = []
    y = []
    for UI in train_data:
        features = generate_features(train_data[UI]['strokes'])
        X.append(features)
        y.append(train_data[UI]['label'])
    return np.array(X), np.array(y)


def generate_features(strokes):
    """
    Calcualte curvature and vicinity of slope as features from the stroke information.
    :param strokes: stroke information
    :return: feature vector
    """
    features = []
    strokes = duplicate_point_filtering(strokes)
    strokes = resampling(strokes)
    strokes = y_normalization(strokes)

    for stroke in strokes:

        x, y = zip(*stroke)

        n = len(stroke)

        for i in range(n):

            if 2 <= i < n - 2:
                normalized_y = y[i]

                a = np.array([x[i + 2], y[i + 2]])
                b = np.array([x[i - 2], y[i - 2]])
                c = np.array([x[i - 2] + 5, y[i - 2]])

                theta = getAngle(a, b, c)

                vicinity_slope = math.atan(theta)

                a = np.array([x[i - 2], y[i - 2]])
                b = np.array([x[i], y[i]])
                c = np.array([x[i + 2], y[i + 2]])

                theta = getAngle(a, b, c)

                curvature = math.atan(theta)

            else:
                vicinity_slope = 0.0
                curvature = 0.0
                normalized_y = y[i]

            features.append(normalized_y)
            features.append(vicinity_slope)
            features.append(curvature)
    return features


def getAngle(a, b, c):
    """
    Calculate angle between the coordinate points
    :param a: x, y coordinate points
    :param b: x, y coordinate points
    :param c: x, y coordinate points
    :return: angle
    """
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def duplicate_point_filtering(strokes_info):
    """
    Remove redundant and repeated points from the stroke information
    :param strokes_info: stroke information
    :return: pre-processed and updated stroke information
    """
    strokes = []
    for mark in strokes_info:
        x_non_duplicate = []
        y_non_duplicate = []
        non_duplicates = set()

        x, y = zip(*mark)

        for point in range(len(x)):
            coordinate = (x[point], y[point])
            if coordinate not in non_duplicates:
                x_non_duplicate.append(coordinate[0])
                y_non_duplicate.append(coordinate[1])
                non_duplicates.add(coordinate)

        stroke = np.column_stack((x_non_duplicate, y_non_duplicate))
        strokes.append(stroke)
    return strokes


def resampling(strokes_info):
    """
    Sample 30 points for each stroke in every symbol.
    :param strokes_info: stroke information
    :return: pre-processed and updated stroke information
    """
    points = 0 # adjustable points to get 30 points on each stroke of every symbol
    sample_points = int(30 // len(strokes_info))

    if 30 % len(strokes_info) != 0:
        points = 30 % len(strokes_info) # points per stroke
    duplicate_point_filtering(strokes_info)

    strokes = []

    for mark in strokes_info:
        if points > 0:
            value_adjust = 1
        else:
            value_adjust = 0

        x, y = zip(*mark)

        # if 3 or less points on this stroke, generate more points
        if len(x) < 4:
            x_resampled = []
            y_resampled = []

            if len(x) == 1:
                x1, y1 = x[0], y[0]
                x2, y2 = x[0] + 0.01, y[0] + 0.01

                x_resampled = np.linspace(x1, x2, max(4, sample_points + value_adjust))
                y_resampled = np.linspace(y1, y2, max(4, sample_points + value_adjust))
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
        sample_at = np.linspace(u.min(), u.max(), sample_points + value_adjust)
        x_resampled, y_resampled = sp.splev(sample_at, tck)

        points -= 1

        stroke = np.column_stack((x_resampled, y_resampled))
        strokes.append(stroke)
    return strokes


def y_normalization(strokes_info):
    """
    Normalize y-axis coordinate
    :param strokes_info: stroke information
    :return: pre-processed and updated stroke information
    """
    minimum_x_coordinate = math.inf
    maximum_x_coordinate = 0
    minimum_y_coordinate = math.inf
    maximum_y_coordinate = 0

    for mark in strokes_info:
        x, y = zip(*mark)
        maximum_y_coordinate = max(max(y), maximum_y_coordinate)
        minimum_y_coordinate = min(min(y), minimum_y_coordinate)
        maximum_x_coordinate = max(max(x), maximum_x_coordinate)
        minimum_x_coordinate = min(min(x), minimum_x_coordinate)

    if maximum_y_coordinate == minimum_y_coordinate:
        maximum_y_coordinate += 100  # can be any positive number
        minimum_y_coordinate -= 100
    strokes = []
    for mark in strokes_info:
        x, y = zip(*mark)
        x_normal = []
        y_normal = []

        for i in range(len(x)):
            y_val = (y[i] - minimum_y_coordinate) / (maximum_y_coordinate - minimum_y_coordinate)
            x_val = (x[i] - minimum_x_coordinate) * (1 / (maximum_y_coordinate - minimum_y_coordinate))
            x_normal.append(x_val)
            y_normal.append(y_val)

        stroke = np.column_stack((x_normal, y_normal))
        strokes.append(stroke)
    return strokes

if __name__ = "__main__":
    print('Extracting Feature..')
    # train_data = load_inkml_files('trainingSymbols')
    with open('train_data.pkl', 'rb') as pickle_file:
        train_data = pickle.load(pickle_file)
        X, y = feature_extraction(train_data)

    data = np.column_stack((X,y))
    df = pd.DataFrame(data)
    df.to_pickle('train_features.pkl')
