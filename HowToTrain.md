# How to train a net

by Marcel Simon

This guide is a step-by-step explanation on how to train a deep convolutional network for a classification task. 

## Path Setup
Set up the environment for using the caffe tools. LD_LIBRARY_PATH is required to run caffe. Setting the PATH is optional, but makes everything more convenient. There is no need to copy the caffe folder. 

    ssh [computer with caffe like herkules]
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/home/simon/Research/lib/gflags/lib:/usr/local/leveldb/leveldb-1.15.0:/home/simon/Research/lib/lmdb/libraries/liblmdb:/opt/intel/composer_xe_2013_sp1.0.080/mkl/lib/intel64:/usr/lib64
    export PATH=/home/simon/Research/lib/caffe/build/tools:$PATH

## Preparing the dataset

In order to use your dataset in caffe, you need to convert it to the _LevelDB_ format, which is a lightweight key-data-storage engine. It is possible to use caffe without having your dataset in the leveldb format, but this would require you to write some custom code. 


The first step is to create two file lists with the corresponding labels, one for the training and one for the test set. Suppose the former is called `train.txt` and the latter `val.txt`. The paths in these file lists should be relative. The labels should start at 0! The content should look like:

    relative/path/img1.jpg 0
    relative/path/img2.jpg 0
    relative/path/img3.jpg 1
    relative/path/img4.jpg 1
    relative/path/img5.jpg 2

For each of these two sets, we will create a separate LevelDB. We will use the `convert_imageset.bin` tool, which comes with the caffe framework. You can use the following sample shell script to create these. 

> Resizing all images to the same size is mandatory, 256x256 usually is a good choice. 


    #Images will be resized accordingly, if these are greater than zero
    export RESIZE_HEIGHT=256
    export RESIZE_WIDTH=256
    
    echo "Creating train leveldb..."
    GLOG_logtostderr=1 convert_imageset.bin \
	/path/to/training_images/ \
	train.txt \
	/path/to/your_train_leveldb 1 leveldb \
	$RESIZE_HEIGHT $RESIZE_WIDTH
    
    echo "Creating val leveldb..."
    GLOG_logtostderr=1 convert_imageset.bin \
	/path/to/val_images/ \
	val.txt \
	/path/to/your_val_leveldb 1 leveldb \
	$RESIZE_HEIGHT $RESIZE_WIDTH

## Describing the Architecture

Now you need to describe the architecture in a text file (with the file extension .prototxt). Besides the actual architecture, this files also specifies the initialization, the data, that is used for training und testing, the loss function, layer specific learning rates and output layers like accuracy calculation. 

There are some important things to remember when writing the the architecture specification. The first aspect is the batch size, which specifies the number of images processed simultaneously, and the image size. These two paramters are only set once in the first (data input) layer. In every consecutive layer, the batch size is copied from the layer below and the image size is calculated based on the stride of the convolutional layers. The second aspect is the difference between the training architecture and the deploy architecture. What we will create now is the training architecture. It needs to include the location of the training (and evaluation) data as well as initialization etc. In contrast, the deploy version of the architecture strips all training specific parts and only includes input dimensions and each layers architecture. 

With this in mind, let us now create a training configuration of Alex Krizhevsky's net. In order to get started quickly, just copy the `alexnet_train_val.prototxt` or `imagenet_train_val.prototxt` from the `examples/imagenet/` dir and modify it to your needs. You probably want to change the source of the input data for training as well as testing and the output dimension in order to match the dataset's number of classes. 

An important parameter of the CNN is the image mean. Subtracting the mean image of the dataset helps to significantly boost the performance. You should compute a dataset specific mean by calling 

    GLOG_logtostderr=1 compute_image_mean.bin /path/to/your_train_leveldb /path/to/your/mean.binaryproto

and adjusting the path to the mean in the `.prototxt` file.

## Training
> You should read this paragraph even if you are just want to fine tune a pre-trained network. The concept of a solver file is explained here, which is required for fine tuning as well. 

There is only one last thing required for the training: the solver. The solver file tells Caffe what parameters it should use for training. You can tell Caffe which learning rate it should use and how it should be adjusted over time. In addition to this, it is possible to create snapshots every x iterations to avoid a complete data loss in case of a power outage or hardware failure. For a quick and easy start, copy the `imagenet_solver.prototxt` from the examples/imagenet/ folder and adjust the net and prefix parameter. For a more detailed descriptions of all the parameters, see the message SolverParameter in the file `src/caffe/proto/caffe.proto`. As far as I can see, the number of iterations describes the number of mini batches, which is not the same as the number of iterations through the whole datasets. 

Now it finally is time to get the training started. With everything set up, simply run 

    GLOG_logtostderr=1 caffe.bin train --solver_proto_file=/path/to/your_solver.prototxt

This might take a looong time. The imagenet example trains about three days on a NVidia Tesla K40 GPU. Don't forget to add the GLOG_logtostderr=1 in front of the actual training command in order to see some output. The result are a bunch of snapshot files and a final model. The model contains all the parameters in a binary file and is about 240 MiB for the default imagenet model.

> For some strange reason, you cannot use absolute paths to the net in the solver file. Hence, the prototxt file should be in the working directory. 

During training, a number of snapshots are created according to the parameters you set in the solver file. These snapshots can be used to resume training at this point by calling 

    GLOG_logtostderr=1 caffe.bin train --solver_proto_file=/path/to/your_solver.prototxt \
	--resume_point_file=/path/to/your_snapshot.solverstate

Once it finished the training process, you can continue training with a new learning rate etc by following the instructions of the following paragraph. Just skip changing the architecture in that case. 

## Finetuning
If you do not have enough data to train a network from scratch, you can use a model pre-trained on imagenet and fine tune the parameters. The first step is copying the original architecture as well as the solver file. First, in the solver file, adjust the net parameter to point to your new architecture file (that is something like imagenet_train_val_ft.prototxt, ft for fine tuned). Now you can simply modify the architecture file to fit your needs. If you modify a layer or want to reinitialize its parameters, you have to give it and the corresponding output blobs a new name. Typically you just need to change the last fully connected layer fc8 as well as the loss layer to 

    layers {
      name: "fc8_ft"
      type: INNER_PRODUCT
      bottom: "fc7"
      top: "fc8_ft"
      blobs_lr: 10
      blobs_lr: 20
      weight_decay: 1
      weight_decay: 0
      inner_product_param {
	num_output: 21
	weight_filler {
	  type: "gaussian"
	  std: 0.01
	}
	bias_filler {
	  type: "constant"
	  value: 0
	}
      }
    }
    layers {
      name: "loss"
      type: SOFTMAX_LOSS
      bottom: "fc8_ft"
      bottom: "label"
    }

Note the changed blobs_lr (increased by a factor of 10) and the changed layer as well as blobs name. 

The remaining part is similar to the regular training:

    GLOG_logtostderr=1 caffe.bin train --solver_proto_file=/path/to/your_solver.prototxt --pretrained_net_file=/path/to/pretrained_model
