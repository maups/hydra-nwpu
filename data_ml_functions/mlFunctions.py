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
from keras import backend as K
from keras.applications import VGG16,imagenet_utils
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.layers import Dense,Input,merge,Flatten,Dropout,LSTM
from keras.models import Sequential,Model
from keras.preprocessing import image
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

from data_ml_functions.densenet import DenseNetImageNet161
from data_ml_functions.dataFunctions import get_batch_inds

from concurrent.futures import ProcessPoolExecutor
from functools import partial

def get_cnn_model (params, algorithm):   
    """
    Load base CNN model and add metadata fusion layers if 'use_metadata' is set in params.py
    :param params: global parameters, used to find location of the dataset and json file
    :return model: CNN model with or without depending on params
    """
    
    ishape = (params.target_img_size[0],params.target_img_size[1],params.num_channels)
    itensor = Input(shape=ishape)

    if (algorithm == 'densenet'):
       print ('CNN = densenet')
       baseModel = DenseNetImageNet161(input_shape=ishape, include_top=False, input_tensor=itensor)
       modelStruct = baseModel.layers[-1].output
    elif (algorithm == 'resnet50'):
       print ('CNN = resnet50')
       baseModel = ResNet50(weights='imagenet', include_top=False, input_tensor=itensor, input_shape=ishape)
       modelStruct = baseModel.output
       modelStruct = Flatten(input_shape=baseModel.output_shape[1:])(modelStruct)
    elif (algorithm == 'xception'):
       print ('CNN = xception')
       baseModel = Xception(weights='imagenet', include_top=False, input_tensor=itensor, input_shape=ishape)
       modelStruct = baseModel.output 
       modelStruct = Flatten(input_shape=baseModel.output_shape[1:])(modelStruct)
    elif (algorithm == 'inceptionv3'):
       print ('CNN = inceptionv3')
       baseModel = InceptionV3(weights='imagenet', include_top=False, input_tensor=itensor, input_shape=ishape)
       modelStruct = baseModel.output
       modelStruct = Flatten(input_shape=baseModel.output_shape[1:])(modelStruct)
    elif (algorithm == 'vgg16'):
       print ('CNN = vgg16')
       baseModel = VGG16(weights='imagenet', include_top=False, input_tensor=itensor)
       modelStruct = baseModel.output
       modelStruct = Flatten(input_shape=baseModel.output_shape[1:])(modelStruct)
    else:
       print ("Error: define a valid CNN model!")

    l2_lambda = 0.0001

    modelStruct = Dense(params.cnn_last_layer_length, activation='relu', name='fc1')(modelStruct)
    modelStruct = Dropout(0.5)(modelStruct)
    modelStruct = Dense(params.cnn_last_layer_length, activation='relu', name='fc2')(modelStruct)
    modelStruct = Dropout(0.5)(modelStruct)
    modelStruct = Dense(params.cnn_last_layer_length, activation='relu', name='fc3')(modelStruct)
    modelStruct = Dropout(0.5)(modelStruct)
    
    predictions = Dense(params.num_labels, kernel_initializer='glorot_uniform', kernel_regularizer=l2(l2_lambda), activation='softmax')(modelStruct)

    model = Model(input=[itensor], output=predictions)

    for i,layer in enumerate(model.layers):
        layer.trainable = True

    return model

def img_generator (params, data, augmentation):
    """
    Custom generator that yields images or (image,metadata) batches and their
    category labels (categorical format).
    :param params: global parameters, used to find location of the dataset and json file
    :param data: list of objects containing the category labels and paths to images and metadata features
    :param metadataStats: metadata stats used to normalize metadata features
    :yield (imgdata,labels) or (imgdata,metadata,labels): image data, metadata (if params set to use), and labels (categorical form)
    """

    N = len(data)

    idx = np.random.permutation(N)

    batchInds = get_batch_inds(params.batch_size_cnn, idx, N)

    executor = ProcessPoolExecutor(max_workers=params.num_workers)

    while True:
        for inds in batchInds:
            batchData = [data[ind] for ind in inds]
            imgdata, labels = load_cnn_batch(params, batchData, executor)
            if (augmentation == 'flip'):
                 datagen = ImageDataGenerator(
                              horizontal_flip=True, 
                              vertical_flip=True
                           ) 
            elif (augmentation == 'zoom'):
                 datagen = ImageDataGenerator(
                              zoom_range=[0.9, 1.0],
                              horizontal_flip=True, 
                              vertical_flip=True
                           ) 
            elif (augmentation == 'shift'):
                 datagen = ImageDataGenerator(
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              horizontal_flip=True, 
                              vertical_flip=True
                           ) 
            #datagen.fit(imgdata)
            #batches = datagen.flow(imgdata, labels, batch_size=params.batch_size_cnn, shuffle=False, save_to_dir='output', save_prefix='aug', save_format='jpg')
            batches = datagen.flow(imgdata, labels, batch_size=params.batch_size_cnn, shuffle=False)
            idx0 = 0
            for batch in batches:
               #print (idx)
               idx1 = idx0 + batch[0].shape[0]
               yield (batch[0], batch[1])
               idx0 = idx1
               if idx1 >= imgdata.shape[0]:
                   break

def read_data (params, data):
    """
    Custom generator that yields images or (image,metadata) batches and their
    category labels (categorical format).
    :param params: global parameters, used to find location of the dataset and json file
    :param data: list of objects containing the category labels and paths to images and metadata features
    :param metadataStats: metadata stats used to normalize metadata features
    :yield (imgdata,labels) or (imgdata,metadata,labels): image data, metadata (if params set to use), and labels (categorical form)
    """

    N = len(data)

    idx = np.random.permutation(N)

    batchInds = get_batch_inds(params.batch_size_cnn, idx, N)

    executor = ProcessPoolExecutor(max_workers=params.num_workers)

    while True:
        for inds in batchInds:
            batchData = [data[ind] for ind in inds]
            imgdata, labels = load_cnn_batch(params, batchData, executor)
            yield (imgdata, labels)


def load_cnn_batch(params, batchData, executor):
    """
    Load batch of images and metadata and preprocess the data before returning.
    :param params: global parameters, used to find location of the dataset and json file
    :param batchData: list of objects in the current batch containing the category labels and paths to CNN codes and images
    :param metadataStats: metadata stats used to normalize metadata features
    :return imgdata,metadata,labels: numpy arrays containing the image data, metadata, and labels (categorical form)
    """

    futures = []
    imgdata = np.zeros((params.batch_size_cnn, params.target_img_size[0],
                        params.target_img_size[1], params.num_channels))
    labels = np.zeros(params.batch_size_cnn)
    for i in range(0, len(batchData)):
        currInput = {}
        currInput['data'] = batchData[i]
        task = partial(_load_batch_helper, currInput)
        futures.append(executor.submit(task))

    results = [future.result() for future in futures]

    for i, result in enumerate(results):
        imgdata[i, :, :, :] = result['img']
        labels[i] = result['labels']

    imgdata = imagenet_utils.preprocess_input(imgdata)
    imgdata = imgdata / 255.0

    labels = to_categorical(labels, params.num_labels)

    return imgdata, labels

def _load_batch_helper(inputDict):
    """
    Helper for load_cnn_batch that actually loads imagery and supports parallel processing
    :param inputDict: dict containing the data and metadataStats that will be used to load imagery
    :return currOutput: dict with image data, metadata, and the associated label
    """

    data = inputDict['data']
    img = image.load_img(data['img_path'])
    img = image.img_to_array(img)
    labels = data['category']
    currOutput = {}
    currOutput['img'] = img
    currOutput['labels'] = labels
    return currOutput

