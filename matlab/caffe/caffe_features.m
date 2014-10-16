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

    if caffe('is_initialized') == 0
        error('You need to initialize caffe first!');
    end
    if (nargin<2)
        layer='relu7';
    end
    if (nargin<3)
        meanfile = 'ilsvrc_2012_mean.mat';
    end
    if (nargin<4)
        batch_size=10;
    end
    if (nargin<5)
        width=227;
    end
    
    filelistmode=ischar(images);
    
    % load the mean
    d = load(meanfile);
    mean = d.image_mean;
    
    if (filelistmode)
        % load the file list
        fid=fopen(images);
        fl=textscan(fid,'%s');
        fl=fl{1};
        fclose(fid);
    else
        fl=images;
    end
    % create tmp for batch
    batch_data = {zeros(width,width,3,batch_size,'single')};
    % Calculate the starting indices of every batch
    slices=1:batch_size:size(fl,1);
    slices(end+1)=size(fl,1)+1;
    % for every slice
    for i=1:numel(slices)-1
%         fprintf('Running batch number %i of %i\n',i, numel(slices)-1);
        % load the image of the next slice
        for j=slices(i):slices(i+1)-1;
            if (filelistmode)
                batch_data{1}(:,:,:,j-slices(i)+1)=caffe_prepare_image(imread(fl{j}),mean,width);
            else
                batch_data{1}(:,:,:,j-slices(i)+1)=caffe_prepare_image(fl{j},mean,width);
            end
        end
        
        tmp_feat = caffe('get_features',batch_data,layer);
        tmp_feat=reshape(tmp_feat{1},size(tmp_feat{1},1)*size(tmp_feat{1},2)*size(tmp_feat{1},3),size(tmp_feat{1},4))';
        if (~exist('features','var'))
            features = zeros(size(fl,1),size(tmp_feat,2),'single');
        end
        features(slices(i):(slices(i+1)-1),:)=tmp_feat(1:(slices(i+1)-slices(i)),:);
    end
    features=double(features);
end

