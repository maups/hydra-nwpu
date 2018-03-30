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
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.applications import imagenet_utils
from data_ml_functions.mlFunctions import get_cnn_model, img_generator, read_data
from data_ml_functions.dataFunctions import prepare_data,calculate_class_weights
from sklearn.utils import class_weight
import numpy as np
import glob
import os

from data_ml_functions.multi_gpu import make_parallel

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import time

class FMOWBaseline:
    
    def __init__(self, params=None, parser=None):
        """
        Initialize baseline class, prepare data, and calculate class weights.
        :param params: global parameters, used to find location of the dataset and json file
        :return: 
        """
        self.params = params
       
        if (parser['prepare']):
           prepare_data(params)
        if (parser['train']):
           self.params.train_cnn = True
        if (parser['test']):
           self.params.test_cnn = True
        if (parser['num_gpus']):
           self.params.num_gpus = parser['num_gpus']
        if (parser['num_epochs']):
           self.params.cnn_epochs = parser['num_epochs']
        if (parser['batch_size']):
           self.params.batch_size_cnn = parser['batch_size'] 

    def hydra_base (self, nepochs, algorithm):
        
        trainData = json.load(open(self.params.files['train_struct']))

        evalData = json.load(open(self.params.files['test_struct']))

        model = get_cnn_model(self.params, algorithm)

        if self.params.num_gpus > 1:
            model = make_parallel(model, self.params.num_gpus)

        model.compile(optimizer=Adam(lr=self.params.cnn_adam_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        train_datagen = read_data (self.params, trainData)

        eval_datagen = read_data (self.params, evalData)

        def lr_scheduler (epoch):
          if self.params.lr_mode is 'progressive_drops':
             lr = 1e-4
          print('lr_scheduler (epoch: %d, lr: %f)' % (epoch, lr))
          return lr

        lr_decay = LearningRateScheduler(lr_scheduler)

        print(("Hydra base (%d epochs): ") % (nepochs))

        callbacks_list = [lr_decay]

        model.fit_generator(train_datagen,
                            steps_per_epoch=(len(trainData) / self.params.batch_size_cnn + 1),
                            epochs=nepochs, callbacks=callbacks_list,
                            validation_data=eval_datagen,
                            validation_steps=(len(evalData) / self.params.batch_size_cnn + 1))
        
        fileName = 'weights.hydra.base.' + algorithm + '.hdf5'

        filePath = os.path.join(self.params.directories['cnn_checkpoint_weights'], fileName)

        model.save(filePath)
 
    def hydra_head (self, nepochs, weight_name, algorithm, prefix, augmentation):
        
        trainData = json.load(open(self.params.files['train_struct']))

        evalData = json.load(open(self.params.files['test_struct']))

        model = get_cnn_model(self.params, algorithm)

        if self.params.num_gpus > 1:
            model = make_parallel(model, self.params.num_gpus)
           
        print ('Loading weights: ', weight_name)
        model.load_weights(weight_name, by_name=True)

        model.compile(optimizer=Adam(lr=self.params.cnn_adam_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        if (augmentation == 'no'):
           train_datagen = read_data (self.params, trainData)
        else:
           train_datagen = img_generator (self.params, trainData, augmentation)

        eval_datagen = read_data (self.params, evalData)

        def lr_scheduler (epoch):
          if self.params.lr_mode is 'progressive_drops':
            if epoch >= 0.75 * nepochs:
                lr = 1e-6
            elif epoch >= 0.15 * nepochs:
                lr = 1e-5
            else:
                lr = 1e-4
          print('lr_scheduler (epoch: %d, lr: %f)' % (epoch, lr))
          return lr

        lr_decay = LearningRateScheduler(lr_scheduler)

        print(("Hydra head (%d epochs): ") % (nepochs))

        #fileName = 'weights.' + algorithm + '.' + self.params.prefix + '.{epoch:02d}.hdf5'
        #filePath = os.path.join(self.params.directories['cnn_checkpoint_weights'], fileName)
        #checkpoint = ModelCheckpoint(filepath=filePath, monitor='loss', verbose=0, save_best_only=False,save_weights_only=False, mode='auto', period=1)
        
        #checkpoint = ModelCheckpoint(monitor='loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
        #callbacks_list = [checkpoint,lr_decay]
        callbacks_list = [lr_decay]

        model.fit_generator(train_datagen,
                            steps_per_epoch=(len(trainData) / self.params.batch_size_cnn + 1),
                            epochs=nepochs, callbacks=callbacks_list,
                            validation_data=eval_datagen,
                            validation_steps=(len(evalData) / self.params.batch_size_cnn + 1))
        
        fileName = 'weights.hydra.head.' + algorithm + '.' + prefix + '.hdf5'

        filePath = os.path.join(self.params.directories['cnn_checkpoint_weights'], fileName)

        model.save(filePath)

    def test_models (self, algorithm, model_weights):

        cnnModel = get_cnn_model(self.params, algorithm)
        if self.params.num_gpus > 1:
            cnnModel = make_parallel(cnnModel, self.params.num_gpus)
        cnnModel.load_weights(model_weights)
        cnnModel = cnnModel.layers[-2]
     
        index = 0
        timestr = time.strftime("%Y%m%d-%H%M%S")
        
        fidCNN1 = open(os.path.join(self.params.directories['predictions'], 'predictions-%s-clas-cnn-%s.txt' % (algorithm, timestr)), 'w')
        fidCNN2 = open(os.path.join(self.params.directories['predictions'], 'predictions-%s-vect-cnn-%s.txt' % (algorithm, timestr)), 'w')

        currBatchSize = 1
        ind = 0
        hit = 0
        miss = 0
        total = 0
        for root, dirs, files in tqdm(os.walk(os.path.join(self.params.directories['test_data']))):
            if len(files) > 0:
                slashes = [i for i,ltr in enumerate(root) if ltr == '/']

            for file in files:
                if file.endswith('.jpg'):
                   baseName = file[:-4]
                   category = root[slashes[-1] + 1:]
                   filename = os.path.join(root,file)
                   img = image.load_img(filename)
                   img = image.img_to_array(img)
                   img.setflags(write=True)
                   imgdata = np.zeros((currBatchSize, self.params.target_img_size[0], self.params.target_img_size[1], self.params.num_channels))
                   imgdata[ind,:,:,:] = img
                   imgdata = imagenet_utils.preprocess_input(imgdata)
                   imgdata = imgdata / 255.0
                   predictionsCNN = cnnModel.predict(imgdata, batch_size=currBatchSize)
                   predCNN = np.argmax(predictionsCNN)
                   oursCNNStr = self.params.category_names[predCNN]
                   print('%s;%s;%s;\n' % (baseName, category, oursCNNStr))
                   fidCNN1.write('%s;%s;%s;\n' % (baseName, category, oursCNNStr))
         
                   fidCNN2.write("%s %s %s " % (baseName, category, oursCNNStr)),
                   for pred in predictionsCNN[0]:
                      fidCNN2.write("%5.12f " % (pred)),
                   fidCNN2.write("\n")

                   if (category == oursCNNStr):
                      hit += 1
                   elif (category != oursCNNStr):
                      miss += 1
                   total += 1
        print ('hit: ', hit, ' miss: ', miss, ' total: ', total, ' percentage: ', float(hit)/float(total)) 
                
        fidCNN1.close()
        fidCNN2.close()

    def test_ensemble (self):

        algorithm = 'densenet'
        weights_densenet = glob.glob(self.params.directories['cnn_checkpoint_weights'] + '/weights.hydra.head.' + algorithm + '*.hdf5')
        
        algorithm = 'resnet50'
        weights_resnet = glob.glob(self.params.directories['cnn_checkpoint_weights'] + '/weights.hydra.head.' + algorithm + '*.hdf5')

        print ('List size:', len(weights_densenet) + len(weights_resnet))

        ListModel = [0] * (len(weights_densenet) + len(weights_resnet))

        i = 0
        for classifier in range(len(weights_densenet)):
          print ("Weight list number: %s" % (weights_densenet[i]))
          cnnModel = get_cnn_model(self.params, 'densenet')
          if self.params.num_gpus > 1:
            cnnModel = make_parallel(cnnModel, self.params.num_gpus)
          cnnModel.load_weights(weights_densenet[i])
          cnnModel = cnnModel.layers[-2]
          ListModel[i] = cnnModel
          i = i + 1

        j = i
        i = 0
        for classifier in range(len(weights_resnet)):
           print ("Weight list number: %s" % (weights_resnet[i]))
           cnnModel = get_cnn_model(self.params, 'resnet50')
           cnnModel = make_parallel(cnnModel, 4)
           if self.params.num_gpus > 1:
              cnnModel.load_weights(weights_resnet[i])
           cnnModel = cnnModel.layers[-2]
           print ('Adding descriptor: ', i, ' in position: ', j)
           ListModel[j] = cnnModel
           j = j + 1
           i = i + 1


        timestr = time.strftime("%Y%m%d-%H%M%S")
        
        fileCNN1 = open(os.path.join(self.params.directories['predictions'], 'predictions-clas-cnn-%s.txt' % (timestr)), 'w')
        fileCNN2 = open(os.path.join(self.params.directories['predictions'], 'predictions-vect-cnn-%s.txt' % (timestr)), 'w')
        fileCNN3 = open(os.path.join(self.params.directories['predictions'], 'predictions-all-cnn-%s.txt' % (timestr)), 'w')

        currBatchSize = 1
        ind = 0
        hit = 0
        miss = 0
        total = 0
        for root, dirs, files in tqdm(os.walk(os.path.join(self.params.directories['test_data']))):
            if len(files) > 0:
                slashes = [i for i,ltr in enumerate(root) if ltr == '/']

            for file in files:
                if file.endswith('.jpg'):
                   baseName = file[:-4]
                   true_category = root[slashes[-1] + 1:]
                   filename = os.path.join(root,file)
                   img = image.load_img(filename)
                   img = image.img_to_array(img)
                   img.setflags(write=True)
                   imgdata = np.zeros((currBatchSize, self.params.target_img_size[0], self.params.target_img_size[1], self.params.num_channels))
                   imgdata[ind,:,:,:] = img
                   imgdata = imagenet_utils.preprocess_input(imgdata)
                   imgdata = imgdata / 255.0

                   predictionsCNN = []

                   fileCNN3.write("%s;%s;" % (baseName,true_category))
                   for i in range(len(weights_densenet) + len(weights_resnet)):
                      
                      cnnModel = ListModel[i]
                       
                      predictionsPartial = cnnModel.predict(imgdata, batch_size=currBatchSize)
                      predCNNPartial = np.argmax(predictionsPartial)
                      classification = self.params.category_names[predCNNPartial]
                      fileCNN3.write('%s;' % (classification))

                      predictionsCNN.append(predictionsPartial[0])

                   fileCNN3.write("\n")

                   predFinal = np.sum(predictionsCNN, axis=0)
                   predCNN = np.argmax(predFinal)
                   oursCNNStr = self.params.category_names[predCNN]
                   fileCNN1.write('%s;%s;%s;\n' % (baseName, true_category, oursCNNStr))
         
                   fileCNN2.write("%s %s %s " % (baseName, true_category, oursCNNStr)),
                   for pred in predFinal:
                      fileCNN2.write("%5.12f " % (pred)),
                   fileCNN2.write("\n")

                   if (true_category == oursCNNStr):
                      hit += 1
                   elif (true_category != oursCNNStr):
                      miss += 1
                   total += 1
        print ('hit: ', hit, ' miss: ', miss, ' total: ', total, ' percentage: ', float(hit)/float(total)) 
                
        fileCNN1.close()
        fileCNN2.close() 
        fileCNN3.close() 


