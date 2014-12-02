# How To Extract Features using Matlab

Once you have trained your network, you can use it to predict new images and extract features. You probably want to create a deploy.prototxt file from your architecture in order to avoid the image cropping which is done by the image data layer used in training. To do so, copy the `train_val.prototxt` and name it `deploy.prototxt`. Now remove the image data layer and the accuracy layer at the end, if there is one. Similar to the `imagenet_deploy.prototxt` file, insert four lines at the beginning after the net name. It should look like: 

```prototxt
name: "CaffeNet"
input: "data"
input_dim: 10
input_dim: 3
input_dim: 227
input_dim: 227
layers {
  name: "conv1"
...
```

The first line specifies the batch size, that means how many images are processed in parallel. Even if you are extraction only a single image, the library always processes 10 (in this case your image and nine dummy images). The batch size should always be equal or greater 2 (1 does not work for some reason). Values larger than 10 or 20 usually do not result in a better performance. 

The second line is the number of input channels. Usually you want to keep the 3 for the three color channels of an RGB image here. 

The third and fourth dimension is the width and height of the image. This should be the same as the crop size in your train_val.prototxt.

Next up is the feature extraciton. Execute the following commands in a bash command line to get started:

```sh
$ ssh -X herkules
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/home/simon/Research/lib/gflags/lib:/usr/local/leveldb/leveldb-1.15.0:/home/simon/Research/lib/lmdb/libraries/liblmdb:/opt/intel/composer_xe_2013_sp1.0.080/mkl/lib/intel64:/usr/lib64
$ /home/matlab/8.2/research/bin/matlab 
```

Matlab version other than 8.2 (that is 2013b) might not work since the MEX files might need to be compiled again. In Matlab, execute the following code to initialize Caffe:

```matlab
>> addpath('/home/simon/Research/lib/caffe/matlab/caffe/')
>> matcaffe_init(1,'/home/simon/Research/lib/caffe/examples/imagenet/imagenet_deploy.prototxt ','/home/simon/Research/lib/caffe/examples/imagenet/caffe_reference_imagenet_model',1);
```

This initialization has to be done only once for each Matlab instance. If you use parfor, that this initialization has to be done on each thread. If you used the imagenet mean file and the usual 227x227 crop size in training, you can now simpy run:

```matlab
>> f=caffe_features('/path/to/filelist.txt'); 
```

to extract feature using a filelist with the default parameter. The first parameter can also be a cell array containing an image (=3D matrix) in each row. If you just want to extract features from a single image, this will look like 

```matlab
>> im=imread('image.jpg');
>> f=caffe_features({im}); 
```

`caffe_features` has more paramters that might be required depending on your needs and the architecture:

```matlab
function [ features ] = caffe_features( images, layer, meanfile, batch_size, width)
%CAFFE_FEATURES Calculates the intermediate activations of a cnn for all 
% images in the filelist. 
% @param images Path to text file containing a list of paths to images or
% cell array containing a list of image arrays
% @param layer    One of the layer of the CNN, check the prototxt file for
% their names
% @param meanfile The average image of your dataset. This should be the
% same that was used during training of the CNN model.
% @param batch_size This is the number of images, that are processed
% simultaneously. This number has to be the same as the batch size that was
% used for validation during training!
% @param width    The width (and at the same time height) of your network
% input. Can be found in the prototxt. 
```

The return value is a n x p - matrix containing n features vectors for each images processed. p is the feature dimension and depends on the layer you chose.

If you did not use the imagenet mean file, you need to compute the mean yourself by adding up all the images in your train dataset and computing the mean. If all images have the same size, you can use `caffe_compute_mean( filelist, is_train )` for it. You might need to copy and modify it according to your needs.