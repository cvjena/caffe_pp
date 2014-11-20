function [ output_args ] = test_svm_classification( input_args )
%TEST_SVM_CLASSIFICATION Summary of this function goes here
%   Detailed explanation goes here
%     matcaffe_init(1,'/home/simon/Research/lib/caffe/examples/imagenet/imagenet_deploy.prototxt','/home/simon/Research/lib/caffe/examples/imagenet/caffe_reference_imagenet_model',1);
    matcaffe_init(1,'/home/simon/tmp/caffe-ft/ft1_results/cub200_ft_deploy.prototxt','/home/simon/tmp/caffe-ft/ft1_results/cub200_ft_train_iter_100000',1);
    f=caffe_features('/home/simon/Datasets/CUB_200_2011/cropped_scaled_alex.txt');
    f2 = caffe_features('caltech_filelist.txt');
    
    
    load('cub200_cropped_caffe_imagenet_fc7.mat','data','labels','tr_ID');
    tr_ID=logical([tr_ID;ones(size(f2,1),1)]);
    labels = [labels;(max(labels(:))+1)*ones(size(f2,1),1)];
    data=[f;f2];
    
    params='';
    fprintf('Params: %s\n',params);
    model = train(labels(tr_ID,:),sparse(data(tr_ID,:)),params);
    acc = predict(labels(~tr_ID,:),sparse(data(~tr_ID,:)),model);
end

