import math
import numpy as np
from scipy.spatial import distance


def bounding_box(stroke_traces):
    """
    Generate bounding box for current stroke
    :param stroke_traces: traces for
    :return:
    """
    x = stroke_traces[:, 0]
    y = stroke_traces[:, 1]
    bbox = {}
    bbox['xmin'] = x.min()
    bbox['ymin'] = y.min()
    bbox['xmax'] = x.max()
    bbox['ymax'] = y.max()

    return bbox


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


def getFeatures(edge0, edge1, data):
    """

    :param edge0:
    :param edge1:
    :param data:
    :return:
    """
    stroke1 = data.coordinates[edge0]
    stroke2 = data.coordinates[edge1]
    features = []
    # print(stroke1)
    normalized_stroke1 = stroke1.copy()
    normalized_stroke2 = stroke2.copy()
    xmin = min(min(stroke1[:, 0]), min(stroke2[:, 0]))
    xmax = max(max(stroke1[:, 0]), max(stroke2[:, 0]))
    ymin = min(min(stroke1[:, 1]), min(stroke2[:, 1]))
    ymax = max(max(stroke1[:, 1]), max(stroke2[:, 1]))

    normalized_stroke1[:, 0] = 0.0 if (xmax - xmin) == 0 else (stroke1[:, 0] - xmin) / (xmax - xmin)
    normalized_stroke2[:, 0] = 0.0 if (xmax - xmin) == 0 else (stroke2[:, 0] - xmin) / (xmax - xmin)
    normalized_stroke1[:, 1] = 0.0 if (xmax - xmin) == 0 else (stroke1[:, 1] - ymin) / (ymax - ymin)
    normalized_stroke2[:, 1] = 0.0 if (xmax - xmin) == 0 else (stroke2[:, 1] - ymin) / (ymax - ymin)
    # distance between avg of x coods of stoke1 and stroke2
    avg_stroke1 = np.array(normalized_stroke1[:, 0]).mean()
    avg_stroke2 = np.array(normalized_stroke2[:, 0]).mean()
    horizontal_distance = np.linalg.norm(avg_stroke1 - avg_stroke2)
    features.append(horizontal_distance)

    # bounding box area difference
    bbox_stroke1 = bounding_box(normalized_stroke1)
    bbox_stroke2 = bounding_box(normalized_stroke2)
    # print(bbox_stroke1['xmin'], bbox_stroke1['xmax'], bbox_stroke1['ymin'], bbox_stroke1['ymax'])
    stroke1_size = (bbox_stroke1['xmax'] - bbox_stroke1['xmin']) * (bbox_stroke1['ymax'] - bbox_stroke1['ymin'])
    stroke2_size = (bbox_stroke2['xmax'] - bbox_stroke2['xmin']) * (bbox_stroke2['ymax'] - bbox_stroke2['ymin'])
    size_difference = abs(stroke1_size - stroke2_size)
    features.append(size_difference)

    # bounding box vertical distance
    vertical_offset = abs((bbox_stroke1['ymax'] - bbox_stroke1['ymin']) - (bbox_stroke2['ymax'] - bbox_stroke2['ymin']))
    features.append(vertical_offset)

    # min distance between two points
    minimum_point_distance = distance.cdist(normalized_stroke1, normalized_stroke2).min(axis=1).min()
    features.append(minimum_point_distance)

    # overlapping area of bounding boxes
    x_left = max(bbox_stroke1['xmin'], bbox_stroke2['xmin'])
    y_top = max(bbox_stroke1['ymin'], bbox_stroke2['ymin'])
    x_right = min(bbox_stroke1['xmax'], bbox_stroke2['xmax'])
    y_bottom = min(bbox_stroke1['ymax'], bbox_stroke2['ymax'])
    overlapping_area = (max((x_right - x_left), 0)) * (max((y_top - y_bottom), 0))
    features.append(overlapping_area)

    # min distance between two boxes
    arr1 = [[bbox_stroke1['xmin'], bbox_stroke1['ymin']], [bbox_stroke1['xmin'], bbox_stroke1['ymax']],
            [bbox_stroke1['ymin'], bbox_stroke1['ymax']], [bbox_stroke1['xmax'], bbox_stroke1['ymax']]]
    arr2 = [[bbox_stroke2['xmin'], bbox_stroke2['ymin']], [bbox_stroke2['xmin'], bbox_stroke2['ymax']],
            [bbox_stroke2['ymin'], bbox_stroke2['ymax']], [bbox_stroke2['xmax'], bbox_stroke2['ymax']]]
    minimum_distance_bbox = distance.cdist(arr1, arr2).min(axis=1).min()
    features.append(minimum_distance_bbox)

    # horizontal overlapping distance of bounding boxes
    horizontal_overlapping_bbox = max((x_right - x_left), 0)
    features.append(horizontal_overlapping_bbox)

    #
    start_dist = np.linalg.norm(normalized_stroke1[0] - normalized_stroke2[0])
    end_dist = np.linalg.norm(normalized_stroke1[-1] - normalized_stroke2[-1])
    features.append(start_dist)
    features.append(end_dist)

    # horizontal offset
    start_offset = np.linalg.norm(normalized_stroke1[0][0] - normalized_stroke2[0][0])
    end_offset = np.linalg.norm(normalized_stroke1[-1][0] - normalized_stroke2[-1][0])
    features.append(start_offset)
    features.append(end_offset)

    # angle between two vectors representing strokes,
    angle_stroke1 = getAngle(stroke1[0], stroke1[-1], [stroke1[-1][0], stroke1[-1][1] + 1])
    angle_stroke2 = getAngle(stroke2[0], stroke2[-1], [stroke2[-1][0], stroke2[-1][1] + 1])
    parallelity = angle_stroke1 - angle_stroke2
    features.append(parallelity)

    psc_xmin = min(min(normalized_stroke1[:, 0]), min(normalized_stroke2[:, 0]))
    psc_xmax = max(max(normalized_stroke1[:, 0]), max(normalized_stroke2[:, 0]))
    psc_ymin = min(min(normalized_stroke1[:, 1]), min(normalized_stroke2[:, 1]))
    psc_ymax = max(max(normalized_stroke1[:, 1]), max(normalized_stroke2[:, 1]))

    hist_stroke1, xedges, yedges = np.histogram2d(normalized_stroke1[:, 0], normalized_stroke1[:, 1], bins=(5, 6),
                                                  range=[[psc_xmin, psc_xmax], [psc_ymin, psc_ymax]])
    features.extend(list(np.array(hist_stroke1).flatten()))

    # H = hist_stroke1.T
    # fig = plt.figure(figsize=(7, 3))
    # ax = fig.add_subplot(131, title='NonUniformImage: interpolated',
    #                      aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
    # im = NonUniformImage(ax, interpolation='bilinear')
    # xcenters = (xedges[:-1] + xedges[1:]) / 2
    # ycenters = (yedges[:-1] + yedges[1:]) / 2
    # im.set_data(xcenters, ycenters, H)
    # ax.images.append(im)

    hist_stroke2, xedges, yedges = np.histogram2d(normalized_stroke2[:, 0], normalized_stroke2[:, 1], bins=(5, 6),
                                                  range=[[psc_xmin, psc_xmax], [psc_ymin, psc_ymax]])
    features.extend(list(np.array(hist_stroke2).flatten()))

    # H = hist_stroke2.T
    # ax = fig.add_subplot(132, title='NonUniformImage: interpolated',
    #                      aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
    # im = NonUniformImage(ax, interpolation='bilinear')
    # xcenters = (xedges[:-1] + xedges[1:]) / 2
    # ycenters = (yedges[:-1] + yedges[1:]) / 2
    # im.set_data(xcenters, ycenters, H)
    # ax.images.append(im)

    coods = []
    for stroke_id in data.strokeID:
        if stroke_id != edge0 or stroke_id != edge1:
            if len(coods) == 0:
                coods = data.coordinates[stroke_id]
            else:
                coods = np.append(coods, data.coordinates[stroke_id], axis=0)

    normalized_stroke = coods.copy()
    normalized_stroke[:, 0] = 0.0 if (xmax - xmin) == 0 else (coods[:, 0] - xmin) / (xmax - xmin)
    normalized_stroke[:, 1] = 0.0 if (xmax - xmin) == 0 else (coods[:, 1] - ymin) / (ymax - ymin)

    hist_stroke_all, xedges, yedges = np.histogram2d(normalized_stroke[:, 0], normalized_stroke[:, 1], bins=(5, 6),
                                                     range=[[psc_xmin, psc_xmax], [psc_ymin, psc_ymax]])
    features.extend(list(np.array(hist_stroke_all).flatten()))

    # H = hist_stroke_all.T
    # ax = fig.add_subplot(133, title='NonUniformImage: interpolated',
    #                      aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
    # im = NonUniformImage(ax, interpolation='bilinear')
    # xcenters = (xedges[:-1] + xedges[1:]) / 2
    # ycenters = (yedges[:-1] + yedges[1:]) / 2
    # im.set_data(xcenters, ycenters, H)
    # ax.images.append(im)
    #
    # plt.show()
    # print(len(features))

    # distance between bounding box centers
    center_stroke1 = np.array(
        [(bbox_stroke1['xmax'] - bbox_stroke1['xmin']) / 2, (bbox_stroke1['ymax'] - bbox_stroke1['ymin']) / 2])
    center_stroke2 = np.array(
        [(bbox_stroke2['xmax'] - bbox_stroke2['xmin']) / 2, (bbox_stroke2['ymax'] - bbox_stroke2['ymin']) / 2])
    center_distance = np.linalg.norm(center_stroke1 - center_stroke2)
    features.append(center_distance)

    # dis- tance between centers-of-mass
    m = np.ones((normalized_stroke1[:, 0].shape))
    cm_stroke1 = np.array(
        [np.sum(normalized_stroke1[:, 0] * m) / np.sum(m), np.sum(normalized_stroke1[:, 1] * m) / np.sum(m)])
    m = np.ones((normalized_stroke2[:, 0].shape))
    cm_stroke2 = np.array(
        [np.sum(normalized_stroke2[:, 0] * m) / np.sum(m), np.sum(normalized_stroke2[:, 1] * m) / np.sum(m)])
    cm_distance = np.linalg.norm(cm_stroke1 - cm_stroke2)
    features.append(cm_distance)

    # max distance between two points
    max_pair_point_distance = distance.cdist(normalized_stroke1, normalized_stroke2).max(axis=1).max()
    features.append(max_pair_point_distance)

    # horizontal offset between the last point of the first stroke and the starting point of the second stroke
    horizontal_offset = np.linalg.norm(normalized_stroke1[-1][0] - normalized_stroke2[0][0])
    features.append(horizontal_offset)

    #  vertical distance between bounding box centers
    vertical_dist_centers = np.linalg.norm(center_stroke1[1] - center_stroke2[1])
    features.append(vertical_dist_centers)

    #  writing slope (angle between the horizontal and the line connecting the last point of the current stroke
    #  and the first point of the next stroke)
    writing_slope = getAngle(stroke1[-1], stroke2[0], [stroke2[0][0], stroke2[0][1] + 1])
    features.append(writing_slope)

    #  writing curvature (angle between the lines defined by the first and last points of each stroke)
    angle_start = getAngle(stroke1[0], stroke2[0], [stroke2[0][0], stroke1[0][1] + 1])
    angle_end = getAngle(stroke1[-1], stroke2[-1], [stroke2[-1][0], stroke2[-1][1] + 1])
    writing_curvature = angle_start - angle_end
    features.append(writing_curvature)

    # print(len(features),features)
    return features


def getAllFeatures(directed_graph, data):
    """
    Generate all features for current inkml for all strokes
    :param directed_graph:
    :param data:
    :return:
    """
    X = []
    y = []
    ground_truth = data.getGT()

    for edges in directed_graph:
        X.append(getFeatures(edges[0], edges[1], data))
        # print(edges[0],edges[1])
        found = False
        for seg in ground_truth:
            if found:
                break
            if edges[0] in seg:
                if edges[1] in seg:
                    y.append('*')
                    found = True

        if not found:
            y.append('_')

    return X, y
