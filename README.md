# Hydra: an Ensemble of Convolutional Neural Networks for Geospatial Land Classification

This repository releases the code for the experimental evaluation of the [Hydra framework](http://arxiv.org/abs/1802.03518) using the NWPU-RESISC45 dataset. If you are looking for code and models for the functional Map of the World (fMoW) challenge, please go to the following repository: [https://github.com/maups/hydra-fmow/](https://github.com/maups/hydra-fmow/)

## Authors

- Rodrigo Minetto - Universidade Tecnológica Federal do Paraná (UTFPR)
- Mauricio Pamplona Segundo - Universidade Federal da Bahia (UFBA)
- Sudeep Sarkar - University of South Florida (USF)

This research was conducted while the authors were at the Computer Vision and Pattern Recognition Group, USF.

## NWPU-RESISC45 description

The [NWPU-RESISC45 dataset](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html) has 45 classes, with 700 images per class, each one with resolution of 256×256 pixels, totaling 31,500 images. Classes may represent different things, such as land use (e.g. basketball court, baseball diamond and tennis court), objects (e.g. airplane and ship) and vegetation (e.g. desert, forest and wetland). This dataset does not provide satellite metadata.

## Hydra description

Hydra is a framework that creates ensembles of Convolutional Neural Networks (CNN) for land use classification in satellite images. The idea behind Hydra is to create an initial CNN that is coarsely optimized but provides a good starting pointing for further optimization, which will serve as the Hydra's body. Then, the obtained weights are fine tuned multiple times to form an ensemble of CNNs that represent the Hydra's heads. The Hydra framework tackles one of the most common problem in multiclass classification, which is the existence of several local minima that prioritize some classes over others and the eventual absence of a global minimum within the classifier search space. The ensemble ends up expanding this space by combining multiple classifiers that converged to local minima and reaches a better global approximation. To stimulate convergence to different end points, we exploit different strategies, such as using online data augmentation, variations in the size of the region of interest, and different image formats when available. The classifiers employed in our Hydra framework are variations of the [fMoW baseline code](https://github.com/fmow/baseline).

## Requirements

- Keras with TensorFlow backend
- nvidia-docker

## Instructions

Download the [NWPU-RESISC45 dataset](https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs) and uncompress it. To run our framework, execute the following sequence of commands:

```
$ git clone https://github.com/maups/hydra-nwpu
$ cd hydra-nwpu
$ ./random_split_10.sh /path_to/NWPU-RESISC45/
$ python runBaseline.py --prepare True
$ python runBaseline.py --train True --num_gpus 4 --num_epochs 16 --batch_size 64
$ python runBaseline.py --test True --num_gpus 4
```

If you want to use 20% of the images for training, use the *random_split_20.sh* script instead and uncomment the line 48 of the file *params.py*.
