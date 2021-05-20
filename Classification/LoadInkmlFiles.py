'''
This program loads the inkml files from the given directory into an array
The code for parsing inkml was inspired by the following repository:
https://github.com/RobinXL/inkml2img

@author: Anushree D
@author: Nishi P
@author: Omkar S
'''
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import pandas as pd
import numpy as np

def load_inkml_files(path,test):
    """
    Parses all the inkml files at  given path and forms a dictionary
    
    :param path: path to directory containing inkml files
    :param test: flag  to see if it is a test set then the dataset won't be labeled
    :return:
    """
    ground_truth_labels = None
    # if it is not a test dataset then read the ground truth to get labels
    if test is False:
        ground_truth_labels= dict()
        with open(path + '/iso_GT.txt', mode='r') as infile:
            for line in infile:
                UI, label = line.strip().split(",")
                ground_truth_labels[UI] = label
    
    # read all inkm;l files in the directory
    all_files = os.listdir(path)
    data = {}
    for file in tqdm(all_files):
        if str(file).__contains__('.inkml'):
            # get UI, strokes and labek for that sample
            UI,strokes,label = inkml_parser(path + '/' + file,ground_truth_labels)
            # store it in the dictionary
            data[UI] = {'strokes':strokes,'label':label}
    
    save to pickle file
    df = pd.DataFrame(data)
    if test is True:
        filename= 'test_data.pkl'
    else:
        filename='training_data.pkl'
    df.to_pickle(filename)
            
    return data


def inkml_parser(filepath,ground_truth_labels=None):
    """
    Parses the inkml file at given filepath and extracts the strokes coordinates for each sample
    :param filepath: inkml filepath
    :param ground_truth_labels: ground truth to get label for that sample
    :return: UI, strokes data, label
    """
    # parse inkml file using xml parser
    tree = ET.parse(filepath)
    root = tree.getroot()
    doc_namespace = "{http://www.w3.org/2003/InkML}"

    # get label from groundtruth using UI from inkml file
    UI = root.find(doc_namespace + "annotation[@type='UI']").text
    if ground_truth_labels is not None:
        label = ground_truth_labels[UI]
    else:
        label = None

    # extract stroke coordinates
    traces_all = [{'id': trace_tag.get('id'),
                   'coords': [[round(float(axis_coord))
                               if float(axis_coord).is_integer()
                               else round(float(axis_coord) * 10000)
                               for axis_coord in coord[1:].split(' ')]
                              if coord.startswith(' ')
                              else [round(float(axis_coord))
                                    if float(axis_coord).is_integer()
                                    else round(float(axis_coord) * 10000)
                                    for axis_coord in coord.split(' ')]
                              for coord in (trace_tag.text).replace('\n', '').split(',')]}
                  for trace_tag in root.findall(doc_namespace + 'trace')]

    # Sort traces_all list by id to make searching for references faster
    traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))

    # Always 1st traceGroup is a redundant wrapper'
    traceGroupWrapper = root.find(doc_namespace + 'traceGroup')
    
    strokes =[]
    if traceGroupWrapper is not None:
        for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):
            # traces of the current traceGroup'
            traces_curr = []
            for traceView in traceGroup.findall(doc_namespace + 'traceView'):
                # Id reference to specific trace tag corresponding to currently considered label'
                traceDataRef = int(traceView.get('traceDataRef'))

                index = next((index for (index, d) in enumerate(traces_all) if int(d["id"]) == traceDataRef), None)
                # Each trace is represented by a list of coordinates to connect'
                # take only x and y coordinates and not the time
                single_trace = np.array(traces_all[index]['coords'])[:,:2]

                traces_curr.append(single_trace)

            strokes = traces_curr

    return UI,strokes,label

if __name__ =='__main__':
    load_inkml_files('trainingSymbols',test =False)
#     load_inkml_files('testSymbols',test = True)
