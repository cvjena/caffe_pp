function [ im ] = caffe_prepare_image( im, mean, width, crop )
    if nargin < 2
        d= load('ilsvrc_2012_mean.mat');
        mean = d.image_mean;
    end
    if nargin < 3
        width = 227;
    end
    if nargin < 4
        crop=1;
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
    
    if (crop)
        % crop to output
        offset_row=int32((size(im,1)-width)/2);
        offset_col=int32((size(im,2)-width)/2);
        im=im(offset_row:offset_row+width-1,offset_col:offset_col+width-1,:);
    else
        % resize to desire output
        im = imresize(im,[width width],'bilinear');
    end
end

