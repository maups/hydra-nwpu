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

import os
from multiprocessing import cpu_count

num_workers = cpu_count()

lr_mode = 'progressive_drops'

cnn_last_layer_length = 4096
#cnn_last_layer_length = 1024

cnn_adam_learning_rate = 1e-4
cnn_adam_loss = 'categorical_crossentropy'

train_cnn = False
test_cnn = False
cnn_epochs = 18
num_gpus = 1
target_img_size = (256,256)
batch_size_cnn = 64
batch_size_eval = 128

num_channels = 3
image_format = 'jpg'

#DIRECTORIES AND FILES
directories = {}
directories['dataset'] = './tmp/NWPU10'
#directories['dataset'] = './tmp/NWPU20'
directories['input'] = os.path.join('.', 'tmp', 'data', 'input')
directories['output'] = os.path.join('.', 'tmp', 'data', 'output')
directories['working'] = os.path.join('.', 'tmp', 'data', 'working')
directories['train_data'] = os.path.join(directories['input'], 'train_data')
directories['test_data'] = os.path.join(directories['input'], 'test_data')
directories['predictions'] = os.path.join(directories['output'], 'predictions')
directories['cnn_checkpoint_weights'] = os.path.join(directories['working'], 'cnn_checkpoint_weights')

files = {}
files['train_struct'] = os.path.join(directories['working'], 'train_struct.json')
files['test_struct'] = os.path.join(directories['working'], 'test_struct.json')

category_names = ['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection', 'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland']

num_labels = len(category_names)

for directory in directories.values():
    if not os.path.isdir(directory):
        os.makedirs(directory)

