function [  ] = matcaffe_perf_testing( input_args )
    
    im = imread('peppers.png');
    
    %% Gradient calculation
    matcaffe_init(0,'../../examples/imagenet/imagenet_deploy.prototxt','../../examples/imagenet/caffe_reference_imagenet_model',1);
    
    fprintf('1 forward pass (CPU), caffes implementation\n');
    tic 
    scores=caffe('forward',{repmat(caffe_prepare_image(im),1,1,1,10)});
    toc
    
    fprintf('256 gradients (CPU), FSU Jena implementation.\n');
    tic;
    gradients = caffe_gradients(im,'pool5',(1:256)');
    toc;
    
    fprintf('10 gradients (CPU), FSU Jena implementation. \n');
    tic;
    gradients = caffe_gradients(im,'pool5',(1:10)');
    toc;
    
    fprintf('1 backward pass (CPU), caffes implemenation\n');
    tic 
    g2 = caffe('backward',{ones(1,1,1000,10)});
    toc
    
    %% Gradient calculation
    matcaffe_init(1,'../../examples/imagenet/imagenet_deploy.prototxt','../../examples/imagenet/caffe_reference_imagenet_model',1);
    
    im = imread('peppers.png');
    fprintf('256 gradients (GPU), FSU Jena implementation.\n');
    tic;
    gradients = caffe_gradients(im,'pool5',(1:256)');
    toc;
    
    fprintf('10 gradients (GPU), FSU Jena implementation. \n');
    tic;
    gradients = caffe_gradients(im,'pool5',(1:10)');
    toc;
    
    fprintf('1 backward pass (GPU), caffes implemenation\n');
    tic 
    g2 = caffe('backward',{ones(1,1,1000,10)});
    toc
    
    
    fprintf('1 forward pass (GPU), caffes implementation\n');
    tic 
    scores=caffe('forward',{repmat(caffe_prepare_image(im),1,1,1,10)});
    toc
end

