
from xml.dom import minidom
from xml.etree import ElementTree
import csv
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from numpy import asarray
import pandas as pd

Labels = []

def get_traces_data(inkml_file_abs_path,classresult):
    traces_data = []

    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    doc_namespace = "{http://www.w3.org/2003/InkML}"

    # get label
    ui = root.find(doc_namespace + "annotation[@type='UI']")
    ui = ui.text.replace('"','')
    label = classresult[ui]
    label=label.replace('\\','slash_')
    if label not in Labels:
        print(label)
        Labels.append(label)
    # process only 10 files for each class to reduce execution time for now
    if os.path.exists(outputdir +'/'+ label) and len(os.listdir(outputdir +'/'+ label)) > 1:
        return None

    'Stores traces_all with their corresponding id'
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

    if traceGroupWrapper is not None:
        for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):
            'traces of the current traceGroup'
            traces_curr = []
            for traceView in traceGroup.findall(doc_namespace + 'traceView'):
                'Id reference to specific trace tag corresponding to currently considered label'
                traceDataRef = int(traceView.get('traceDataRef'))

                index = next((index for (index, d) in enumerate(traces_all) if int(d["id"]) == traceDataRef), None)
                # print(index)
                'Each trace is represented by a list of coordinates to connect'
                single_trace = traces_all[index]['coords']
                traces_curr.append(single_trace)

            traces_data.append({'label': label, 'trace_group': traces_curr})

    return traces_data


def inkml2img(input_path, output_path,classresult):
    # get coords and label
    traces = get_traces_data(input_path,classresult)

    path = input_path.split('/')
    path = path[len(path) - 1].split('.')
    path = path[0]

    if traces is not None:
        for elem in traces:

            #         print(elem)
            #         print('-------------------------')
            #         print(elem['label'])

            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis('off')
            trace = elem['trace_group']
            output_path = output_path

            # plot coords
            for subls in trace:
                # print(subls)
                data = np.array(subls)
                if len(data[0]) == 2:
                    x, y = zip(*data)
                else:
                    # x,y and time
                    x, y, z = zip(*data)
                plt.plot(x, y, linewidth=2, c='black')

            label = elem['label']
            # print(label)
            ind_output_path = output_path +'/'+ label
            try:
                os.mkdir(ind_output_path)
            except OSError:
                            # print ("Folder %s Already Exists" % ind_output_path)
                            # print(OSError.strerror)
                pass
            else:
                            # print ("Successfully created the directory %s " % ind_output_path)
                pass

            plt.savefig(ind_output_path + '/' + path + '.png', bbox_inches='tight', dpi=100)
            plt.gcf().clear()
            image = Image.open(ind_output_path + '/' + path + '.png').convert('L')
            image_arr = asarray(image)
            return image_arr,trace,label



outputdir = 'images'


if __name__ == "__main__":
    # create directory for output images
    try:
        os.mkdir(outputdir)
    except OSError:
        # print("Folder %s Already Exists" % outputdir)
        # print(OSError.strerror)
        pass
    else:
        # print("Successfully created the directory %s " % outputdir)
        pass

    path = 'trainingSymbols'
    # get labels for each input
    classresult = dict()
    with open(path + '/iso_GT.txt', mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            classresult[row[0]] = row[1]
    filenames = []
    labels = []
    traces = []
    images = []
    # convert all inkml from training data to png
    files = os.listdir(path)
    for file in tqdm(files):
        if str(file).__contains__('.inkml'):
            # print(file)
            result = inkml2img(path + '/' + file, outputdir,classresult)
            if result is not None:
                image,trace,label = result
                filenames.append(file)
                images.append(image)
                traces.append(trace)
                labels.append(label)

    df = pd.DataFrame({'filenames':filenames,'images':images,'traces':traces,'labels':labels})
    df.to_csv('trainingdata.csv', index=False, header=True)




