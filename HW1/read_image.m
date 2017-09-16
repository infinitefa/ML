clear
% Read File
file_path = '.\CroppedYale\yaleB';
SADac = zeros(1,39);
SSDac = zeros(1,39);
testnum = zeros(1,39);
TR = zeros(192, 168, 2000);
TE = zeros(192, 168, 2000);

for i=1:39
    if i == 14
        continue;
    elseif i<10
        folder = [file_path '0' num2str(i)];
    else
        folder = [file_path num2str(i)];
    end
    img_path_list = dir([folder '\*.pgm']);
    num = length(img_path_list);%filenumber
    num = num - 1;
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
    TR(:,:,j+(i-1)*35) = Y;
    %imshow(X);
    end
end

%Find NN
for i=1:39
    if i == 14
        continue;
    elseif i<10
        folder = [file_path '0' num2str(i)];
    else
        folder = [file_path num2str(i)];
    end
    img_path_list = dir([folder '\*.pgm']);
    num = length(img_path_list);%filenumber
    testnum(i) = num -36;
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
   
    SADmin = 9999999;
    SSDmin = 99999999999;
    SADtemp =0;
    SSDtemp =0;
    for k = 1:35*39
        temp = sum(sum(abs(double(Y)-TR(:,:,k))));
        if SADmin >= temp
            SADmin = temp;
            SADtemp = ceil(k/35);
        end
        
    end
    if SADtemp == i
        SADac(i) = SADac(i) + 1;     
    end
    for k = 1:35*39
        temp =  sum(sum((double(Y)-TR(:,:,k)).^2));
        if SSDmin >= temp
            SSDmin = temp;
            SSDtemp = ceil(k/35);
        end
    end
    if SSDtemp == i
    SSDac(i) = SSDac(i) + 1;
    end  
    end
end
%caculate
SADacrate = sum(SADac)/sum(testnum)
SSDacrate = sum(SSDac)/sum(testnum)

% 顯示結果
%subplot(1,2,1);
%image(X);
%title ('Original Image');
%subplot(1,2,2);
%image(Z);
%title ('Generated Image')
