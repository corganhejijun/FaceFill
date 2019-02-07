clc;clear;
originPath = './images/origin/';
gtPath = './images/GT/';
fileName = '000019';

origin = imread([originPath,fileName,'.jpg']);
gt = imread([gtPath,fileName,'.png']);

result = zeros(128, 128, 3, 'uint8');
result = origin(:,1:128,:);
imwrite(result, [originPath,fileName,'.jpg']);
result = zeros(128, 128, 3, 'uint8');
result(:,:,1) = gt(:,1:128)*128;
imwrite(result, [gtPath,fileName,'.png']);