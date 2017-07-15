%my loss
clear;
clc;
close all;
train_log_file = '.\log\caffe.exe.FA.FA.log.INFO.20170329-235824.1924' ;
train_interval = 1000 ;
test_interval = 1000 ;

[~, string_output] = dos(['type ' , train_log_file ]) ;
%pat='1 = .*? loss';
pat = 'accuracy = 0.?\d*';
o1=regexp(string_output,pat,'start');%用'start'??指定?出o1?匹配正?表?式的子串的起始位置
o2=regexp(string_output,pat,'end');%用'start'??指定?出o1?匹配正?表?式的子串的?束位置
o3=regexp(string_output,pat,'match');%用'match'??指定?出o2?匹配正?表?式的子串 

accuracy=zeros(1,size(o1,2));
for i=1:size(o1,2)
    accuracy(i) = str2num(string_output(o1(i)+11:o2(i)));
    %loss(i)=str2num(string_output(o3(i):o3(i)+6));
end
plot(accuracy)
%10
clear;
train_log_file = '.\log\caffe.exe.FA.FA.log.INFO.20170329-222703.1836' ;
train_interval = 1000 ;
test_interval = 1000 ;

[~, string_output] = dos(['type ' , train_log_file ]) ;
%pat='1 = .*? loss';
pat = 'accuracy = 0.?\d*';
o1=regexp(string_output,pat,'start');%用'start'??指定?出o1?匹配正?表?式的子串的起始位置
o2=regexp(string_output,pat,'end');%用'start'??指定?出o1?匹配正?表?式的子串的?束位置
o3=regexp(string_output,pat,'match');%用'match'??指定?出o2?匹配正?表?式的子串 

accuracy=zeros(1,size(o1,2));
for i=1:size(o1,2)
    accuracy(i) = str2num(string_output(o1(i)+11:o2(i)));
    %loss(i)=str2num(string_output(o3(i):o3(i)+6));
end
figure;
plot(accuracy)
%50
clear;
train_log_file = '.\log\caffe.exe.FA.FA.log.INFO.20170329-224334.2176' ;
train_interval = 1000 ;
test_interval = 1000 ;

[~, string_output] = dos(['type ' , train_log_file ]) ;
%pat='1 = .*? loss';
pat = 'accuracy = 0.?\d*';
o1=regexp(string_output,pat,'start');%用'start'??指定?出o1?匹配正?表?式的子串的起始位置
o2=regexp(string_output,pat,'end');%用'start'??指定?出o1?匹配正?表?式的子串的?束位置
o3=regexp(string_output,pat,'match');%用'match'??指定?出o2?匹配正?表?式的子串 

accuracy=zeros(1,size(o1,2));
for i=1:size(o1,2)
    accuracy(i) = str2num(string_output(o1(i)+11:o2(i)));
    %loss(i)=str2num(string_output(o3(i):o3(i)+6));
end
figure;
plot(accuracy)