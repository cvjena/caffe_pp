function [ im ] = caffe_prepare_image( im, mean, width )
    if nargin < 3
        width = size(mean,2);
    end
    % make sure it's single type
    im = single(im);
    % resize to mean image
    im = imresize(im,[size(mean,1) size(mean,2)],'bilinear');
    % catch gray scale images
    if (size(im,3)==1)
        im=repmat(im,1,1,3);
    else
        im = im(:,:,[3 2 1]);
    end
    % subtract mean
    im = im - mean;
    %  transpose 
    im = permute(im, [2 1 3]);
    % resize to desire output
    im = imresize(im,[width width],'bilinear');
end

