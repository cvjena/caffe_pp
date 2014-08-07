function [ features ] = caffe_features( filelist, layer, meanfile, batch_size, width)
%CAFFE_FEATURES Summary of this function goes here
%   Detailed explanation goes here
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
    % load the mean
    d = load(meanfile);
    mean = d.image_mean;
    
    % load the file list
    fid=fopen(filelist);
    fl=textscan(fid,'%s');
    fl=fl{1};
    fclose(fid);
    % create tmp for batch
    batch_data = {zeros(width,width,3,batch_size,'single')};
    % Calculate the starting indices of every batch
    slices=1:batch_size:size(fl,1);
    slices(end+1)=size(fl,1)+1;
    % for every slice
    for i=1:numel(slices)-1
        fprintf('Running batch number %i of %i\n',i, numel(slices));
        % load the image of the next slice
        for j=slices(i):slices(i+1)-1;
            batch_data{1}(:,:,:,j-slices(i)+1)=caffe_prepare_image(imread(imread(fl{j})),mean,227);
        end
        tmp_feat = caffe('get_features',batch_data,layer);
        tmp_feat = squeeze(tmp_feat{1})';
        if (~exist('features','var'))
            features = zeros(size(fl,1),size(tmp_feat,2),'single');
        end
        features(slices(i):(slices(i+1)-1),:)=tmp_feat(1:(slices(i+1)-slices(i)),:);
    end
    features=double(features);
end

