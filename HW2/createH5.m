clear;
file_path = '.\CroppedYale\yaleB';
trainImage = zeros(192, 168, 1330);
trainLabel = zeros(1,1330);
dif = 0;
for i=1:39
    if i == 14
        dif = 1;
        continue;
    elseif i<10
        folder = [file_path '0' num2str(i)];
    else
        folder = [file_path num2str(i)];
    end
    img_path_list = dir([folder '\*.pgm']);
    num = length(img_path_list);%filenumber
    for j = (1:35)
        if (img_path_list(j).name(13:23) == 'Ambient.pgm')
            continue;
        end
    X = imread([folder '\' img_path_list(j).name]);
    [m n] =size(X);
    %fix size
    if m~=192 || n~=168 
        Y(:,:) = X(1:192,1:168);
    else
        Y = X;
    end
    trainImage(:,:,j+(i-1-dif)*35) = Y;
    trainLabel(1,j+(i-1-dif)*35)=i; 
    %imshow(X);
    end
end
trainImage = reshape(trainImage,192,168,1,1330);
trainImage = permute(trainImage,[2 1 3 4]);

h5create('train.hdf5','/data',size(trainImage),'Datatype','double');
h5create('train.hdf5','/label',size(trainLabel),'Datatype','double');
h5write('train.hdf5','/data',trainImage);
h5write('train.hdf5','/label',trainLabel);

testImage = zeros(192, 168, 1324);
testLabel = zeros(1,1324);
dif = 0;
for i=1:39
    if i == 14
        dif = 1;
        continue;
    elseif i<10
        folder = [file_path '0' num2str(i)];
    else
        folder = [file_path num2str(i)];
    end
    img_path_list = dir([folder '\*.pgm']);
    num = length(img_path_list);%filenumber
    for j = (36:num)
        if (img_path_list(j).name(13:23) == 'Ambient.pgm')
            continue;
        end
    X = imread([folder '\' img_path_list(j).name]);
    [m n] =size(X);
    %fix size
    if m~=192 || n~=168 
        Y(:,:) = X(1:192,1:168);
    else
        Y = X;
    end
    testImage(:,:,j-35+(i-1-dif)*35) = Y;
    testLabel(1,j-35+(i-1-dif)*35)=i; 
    %imshow(X);
    end
end
testImage = reshape(testImage,192,168,1,1324);
testImage = permute(testImage,[2 1 3 4]);

h5create('test.hdf5','/data',size(testImage),'Datatype','double');
h5create('test.hdf5','/label',size(testLabel),'Datatype','double');
h5write('test.hdf5','/data',testImage);
h5write('test.hdf5','/label',testLabel);
