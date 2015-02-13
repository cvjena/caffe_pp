function [scores, maxlabel] = matcaffe_grad_demo(im, use_gpu)
% scores = matcaffe_grad_demo(im, use_gpu)
%
% Demo of the matlab wrapper using the ILSVRC network.
%
% input
%   im       color image as uint8 HxWx3
%   use_gpu  1 to use the GPU, 0 to use the CPU
%
% output
%   scores   1000-dimensional ILSVRC score vector
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda-5.5/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system
%
% Usage:
%  im = imread('../../examples/images/cat.jpg');
%  scores = matcaffe_demo(im, 1);
%  [score, class] = max(scores);
% Five things to be aware of:
%   caffe uses row-major order
%   matlab uses column-major order
%   caffe uses BGR color channel order
%   matlab uses RGB color channel order
%   images need to have the data mean subtracted

% Data coming in from matlab needs to be in the order 
%   [width, height, channels, images]
% where width is the fastest dimension.
% Here is the rough matlab for putting image data into the correct
% format:
%   % convert from uint8 to single
%   im = single(im);
%   % reshape to a fixed size (e.g., 227x227)
%   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
%   % permute from RGB to BGR and subtract the data mean (already in BGR)
%   im = im(:,:,[3 2 1]) - data_mean;
%   % flip width and height to make width the fastest dimension
%   im = permute(im, [2 1 3]);

% If you have multiple images, cat them with cat(4, ...)

% The actual forward function. It takes in a cell array of 4-D arrays as
% input and outputs a cell array. 

if nargin < 1
  % For demo purposes we will use the peppers image
  im = imread('peppers.png');
end
tic;
gradients = caffe_gradients(im,'pool5',(1:256)');
toc;
% tic 
% scores=caffe('forward',{repmat(caffe_prepare_image(im),1,1,1,10)});
% g2 = caffe('backward',{ones(1,1,1000,10)});
% toc
% g2=g2{1};
% gradients=g2;
gradients=gradients(:,:,:,1:16);
for i=1:size(gradients,4)
    g=gradients(:,:,:,i);
    g=abs(g);
    g=sum(g,3);
    g=g/max(gradients(:));
%     g=g*4;
    subplot(ceil(sqrt(size(gradients,4))),ceil(sqrt(size(gradients,4))),i);
    imshow(g);
end
end
