from fmowBaseline import FMOWBaseline
import params
import argparse as ap
from keras import backend as K 

#This code is a version from the original code of:
#__author__ = 'jhuapl'
#__version__ = 0.1
#https://github.com/fMoW/baseline

def main (parser):

    baseline = FMOWBaseline(params, parser)

    nepochs_base = int(parser['num_epochs'] * 0.5)
    nepochs_head = int(parser['num_epochs'] * 0.5)

    if (parser['train']):

        #----------------------
        algorithm = 'densenet'

        K.clear_session() 
        print ('Running base algorithm: ', algorithm)
        baseline.hydra_base(nepochs_base, algorithm)

        K.clear_session() 
        print ('Running head algorithm: ', algorithm, ' prefix 01 (no data augmentation)')
        baseline.hydra_head (nepochs_head, params.directories['cnn_checkpoint_weights'] + '/weights.hydra.base.' + algorithm + '.hdf5', algorithm, '01', 'no')
        
        K.clear_session() 
        print ('Running head algorithm: ', algorithm, ' prefix 02 (flip)')
        baseline.hydra_head (nepochs_head, params.directories['cnn_checkpoint_weights'] + '/weights.hydra.base.' + algorithm + '.hdf5', algorithm, '02', 'flip')

        K.clear_session() 
        print ('Running head algorithm: ', algorithm, ' prefix 03 (zoom)')
        baseline.hydra_head (nepochs_head, params.directories['cnn_checkpoint_weights'] + '/weights.hydra.base.' + algorithm + '.hdf5', algorithm, '03', 'zoom')

        K.clear_session() 
        print ('Running head algorithm: ', algorithm, ' prefix 04 (shift)')
        baseline.hydra_head (nepochs_head, params.directories['cnn_checkpoint_weights'] + '/weights.hydra.base.' + algorithm + '.hdf5', algorithm, '04', 'shift')

        #----------------------
        algorithm = 'resnet50'

        K.clear_session() 
        print ('Running base algorithm: ', algorithm)
        baseline.hydra_base(nepochs_base, algorithm)

        K.clear_session() 
        print ('Running head algorithm: ', algorithm, ' prefix 01 (no data augmentation)')
        baseline.hydra_head (nepochs_head, params.directories['cnn_checkpoint_weights'] + '/weights.hydra.base.' + algorithm + '.hdf5', algorithm, '01', 'no')

        K.clear_session() 
        print ('Running head algorithm: ', algorithm, ' prefix 02 (flip)')
        baseline.hydra_head (nepochs_head, params.directories['cnn_checkpoint_weights'] + '/weights.hydra.base.' + algorithm + '.hdf5', algorithm, '02', 'flip')

        K.clear_session() 
        print ('Running head algorithm: ', algorithm, ' prefix 03 (zoom)')
        baseline.hydra_head (nepochs_head, params.directories['cnn_checkpoint_weights'] + '/weights.hydra.base.' + algorithm + '.hdf5', algorithm, '03', 'zoom')

        K.clear_session() 
        print ('Running head algorithm: ', algorithm, ' prefix 04 (shift)')
        baseline.hydra_head (nepochs_head, params.directories['cnn_checkpoint_weights'] + '/weights.hydra.base.' + algorithm + '.hdf5', algorithm, '04', 'shift')

    if (parser['test']):
        #baseline.test_models('densenet', '../data/working/cnn_checkpoint_weights/weights.hydra.head.densenet.01.hdf5')
        baseline.test_ensemble()
    
if __name__ == "__main__":
    arg = ap.ArgumentParser()
    arg.add_argument ("--train", default="", type=str)
    arg.add_argument ("--test", default="", type=str)
    arg.add_argument ("--prepare", default="", type=str)
    arg.add_argument ("--num_gpus", default=1, type=int)
    arg.add_argument ("--num_epochs", default=12, type=int)
    arg.add_argument ("--batch_size", default=64, type=int)
    parser = vars(arg.parse_args())
    print (parser)
    main (parser)
