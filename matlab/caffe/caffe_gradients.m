function [ gradients ] = caffe_gradients( im, layer, channels, meanfile, batch_size, width )
%CAFFE_GRADIENTS Calculates the gradients of a intermediate layer output
% with respect the CNN input (that is the input image). This function adds
% the gradients of all elements of the same channel implictly.
% @param im       A regular image, that was read using the imread(..)
% function. 
% @param layer    One of the layer of the CNN, check the prototxt file for
% their names.
% @channels       The ids of the channels, that you want to calculate the
% gradients from. The ids start at 1! You should make, that the ids do not
% exceed the number of channels in your selected layer. In that case, the
% behavior is undefined. 
% @param meanfile The average image of your dataset. This should be the
% same that was used during training of the CNN model.
% @param batch_size This is the number of images, that are processed
% simultaneously. This number has to be the same as the batch size that was
% used for validation during training!
% @param width    The width (and at the same time height) of your network
% input. Can be found in the prototxt. 

    if nargin < 2
        layer = 'pool5';
    end
    if nargin < 3
        channels = (1:256)';
    end
    if nargin < 4
        meanfile = 'ilsvrc_2012_mean.mat';
    end
    if nargin < 5
        batch_size=10;
    end
    if nargin < 6
        width = 227;
    end
    d = load(meanfile);
    mean = d.image_mean;
    channels = channels-1;
    im = caffe_prepare_image(im,mean,width);
    im = repmat(im,1,1,1,batch_size);
    % transform to shape num x width x height x channels
    gradients = caffe('get_gradients',{im}, layer, channels);
    
    % Transpose the gradients map, for caffe uses row major order and
    % Matlab column major
    gradients=permute(gradients,[2 1 3 4]);
    % GBR -> RGB
    gradients=gradients(:,:,[3 2 1],:);
    % Make double as this is Matlab default
    gradients=double(gradients);
end

