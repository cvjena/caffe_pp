function [ gradients ] = caffe_gradients( im, layer, channels, meanfile, batch_size, width )
%CAFFE_GRADIENTS Calculates the gradient maps for all specified channels. 
% The channel ids start at 1, not 0!
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
end

