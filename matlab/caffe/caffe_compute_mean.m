function [ m ] = caffe_compute_mean( filelist, is_train )
    fid=fopen(filelist);
    fl=textscan(fid,'%s');
    fl=fl{1};
    fclose(fid);
    im = int32(imread(fl{1}));
    dims=size(im);
    m = int32(zeros(size(im,1),size(im,2),3));
    for i=1:size(fl,1)
        if (is_train(i,:))
            disp(i);
            im=imresize(imread(fl{i}),dims(1:2),'bilinear');
            if (size(im,3)==1)
                im=repmat(im,1,1,3);
            end
            m = m + int32(im);
        end
    end
    m = single(m) / size(fl,1);
    m(m<0)=0;
    m(m>255)=255;
    m=m(:,:,[3 2 1]);
end

