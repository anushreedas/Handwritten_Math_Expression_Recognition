from xml.dom import minidom
from xml.etree import ElementTree
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import csv
import pandas as pd
import numpy as np

def load_inkml_files(path,test):
    all_files = os.listdir(path)
    filepaths = []
    ground_truth_labels = None
    if test is False:

        ground_truth_labels= dict()
        with open(path + '/iso_GT.txt', mode='r') as infile:
            for line in infile:
                UI, label = line.strip().split(",")
                ground_truth_labels[UI] = label

    data = {}
    for file in tqdm(all_files):
        if str(file).__contains__('.inkml'):
            UI,strokes,label = inkml_parser(path + '/' + file,ground_truth_labels)
            data[UI] = {'strokes':strokes,'label':label}

    df = pd.DataFrame(data)
    if test is True:
        filename= 'test_data.pkl'
    else:
        filename='training_data.pkl'
    df.to_pickle(filename)
            
    return data


def inkml_parser(filepath,ground_truth_labels=None):
    tree = ET.parse(filepath)
    root = tree.getroot()
    doc_namespace = "{http://www.w3.org/2003/InkML}"

    # get label
    UI = root.find(doc_namespace + "annotation[@type='UI']").text
    if ground_truth_labels is not None:
        label = ground_truth_labels[UI]
    else:
        label = None


    traces_all = [{'id': trace_tag.get('id'),
                   'coords': [
                       [round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
                        for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
                           else [round(float(axis_coord)) if float(axis_coord).is_integer() else round(
                           float(axis_coord) * 10000) \
                                 for axis_coord in coord.split(' ')] \
                       for coord in (trace_tag.text).replace('\n', '').split(',')]} \
                  for trace_tag in root.findall(doc_namespace + 'trace')]

    'Sort traces_all list by id to make searching for references faster'
    traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))

    'Always 1st traceGroup is a redundant wrapper'
    traceGroupWrapper = root.find(doc_namespace + 'traceGroup')
    strokes =[]
    strokes_x = []
    strokes_y = []
    if traceGroupWrapper is not None:
        for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):
            'traces of the current traceGroup'
            traces_curr = []
            for traceView in traceGroup.findall(doc_namespace + 'traceView'):
                'Id reference to specific trace tag corresponding to currently considered label'
                traceDataRef = int(traceView.get('traceDataRef'))

                index = next((index for (index, d) in enumerate(traces_all) if int(d["id"]) == traceDataRef), None)
                'Each trace is represented by a list of coordinates to connect'
                # print(np.array(traces_all[index]['coords']).shape)
                # print((np.array(traces_all[index]['coords'])[:,:2]))
                single_trace = np.array(traces_all[index]['coords'])[:,:2]
                # print(len(single_trace[0]))
                if len(single_trace[0]) == 3:
                    print('error')
                traces_curr.append(single_trace)

            strokes = traces_curr
            # print(strokes)
        if len(strokes[0][0]) == 3:
            print('error')

    return UI,strokes,label

# load_inkml_files('trainingSymbols',test =False)
load_inkml_files('testSymbols',test = True)