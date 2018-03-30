"""
Copyright 2017 The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = 'jhuapl'
__version__ = 0.1

import json
import os
import errno
import numpy as np
import string
import dateutil.parser as dparser
from PIL import Image
from sklearn.utils import class_weight
from keras.preprocessing import image
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import warnings

def prepare_data(params):
    """
    Saves sub images, converts metadata to feature vectors and saves in JSON files, 
    calculates dataset statistics, and keeps track of saved files so they can be loaded as batches
    while training the CNN.
    :param params: global parameters, used to find location of the dataset and json file
    :return: 
    """

    # suppress decompression bomb warnings for Pillow
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)

    walkDirs = ['train', 'test']

    executor = ProcessPoolExecutor(max_workers=params.num_workers)
    futures = []
    paramsDict = vars(params)
    keysToKeep = ['image_format', 'target_img_size', 'category_names']
    paramsDict = {keepKey: paramsDict[keepKey] for keepKey in keysToKeep}
    
    for currDir in walkDirs:
        isTrain = (currDir == 'train') 
        if isTrain:
            outDir = params.directories['train_data']
        else:
            outDir = params.directories['test_data']

        print('Queuing sequences in: ' + currDir)
        for root, dirs, files in tqdm(os.walk(os.path.join(params.directories['dataset'], currDir))):
            if len(files) > 0:
                slashes = [i for i,ltr in enumerate(root) if ltr == '/']
                        
            for file in files:
                if file.endswith('.jpg'): 
                    task = partial(_process_file, file, slashes, root, isTrain, outDir, paramsDict)
                    futures.append(executor.submit(task))

    record_files (params, futures)
    executor.shutdown()

def record_files (params, futures):

    print('Wait for all preprocessing tasks to complete...')
    results = []
    [results.extend(future.result()) for future in tqdm(futures)]
    
    trainingData = [r[0] for r in results if r[0] is not None]
    testData = [r[1] for r in results if r[1] is not None]

    json.dump(testData, open(params.files['test_struct'], 'w'))
    json.dump(trainingData, open(params.files['train_struct'], 'w'))

def _process_file(file, slashes, root, isTrain, outDir, params):
    """
    Helper for prepare_data that actually loads and resizes each image and computes
    feature vectors. This function is designed to be called in parallel for each file
    :param file: file to process
    :param slashes: location of slashes from root walk path
    :param root: root walk path
    :param isTrain: flag on whether or not the current file is from the train set
    :param outDir: output directory for processed data
    :param params: dict of the global parameters with only the necessary fields
    :return (allFeatures, allTrainResults, allTestResults)
    """
    noResult = [(None, None)]
    baseName = file[:-4]

    imgFile = baseName + '.' + params['image_format']

    if not os.path.isfile(os.path.join(root, imgFile)):
        return noResult
    
    allResults = []

    try:
        img = image.load_img(os.path.join(root, imgFile))
    except:
        return noResult

    category = root[slashes[-1] + 1:]
    currOut = os.path.join(outDir, category)
    if not os.path.isdir(currOut):
        try:
            os.makedirs(currOut)
        except OSError as e:
            if e.errno == errno.EEXIST:
               pass
    
    imgPath = os.path.join(outDir, category, baseName + '.jpg')
    subImg = img.resize(params['target_img_size'])
    subImg.save(imgPath)
    
    if isTrain:
        allResults.append(({"img_path": imgPath, "category": params['category_names'].index(category)}, None))
    else:
        allResults.append((None, {"img_path": imgPath, "category": params['category_names'].index(category)}))

    return allResults


def get_batch_inds(batch_size, idx, N):
    """
    Generates an array of indices of length N
    :param batch_size: the size of training batches
    :param idx: data to split into batches
    :param N: Maximum size
    :return batchInds: list of arrays of data of length batch_size
    """
    batchInds = []
    idx0 = 0

    toProcess = True
    while toProcess:
        idx1 = idx0 + batch_size
        if idx1 > N:
            idx1 = N
            idx0 = idx1 - batch_size
            toProcess = False
        batchInds.append(idx[idx0:idx1])
        idx0 = idx1

    return batchInds

def calculate_class_weights(params):
    """
    Computes the class weights for the training data and writes out to a json file 
    :param params: global parameters, used to find location of the dataset and json file
    :return: 
    """
    
    counts = {}
    for i in range(0,params.num_labels):
        counts[i] = 0

    trainingData = json.load(open(params.files['training_struct1']))

    ytrain = []
    for i,currData in enumerate(trainingData):
        ytrain.append(currData['category'])
        counts[currData['category']] += 1
        print(i)

    classWeights = class_weight.compute_class_weight('balanced', np.unique(ytrain), np.array(ytrain))

    with open(params.files['class_weight'], 'w') as json_file:
        json.dump(classWeights.tolist(), json_file)
    
    
