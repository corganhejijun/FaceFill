function seg()
clc;clear;
addpath('./UGM');

origin_path = './images/origin';
test_path = './images/test';
origin_seg_path = './images/origin_seg';
test_seg_path = './images/test_seg_0.006';

dirSeg(test_path, test_seg_path);

function dirSeg(imagePath, segPath)
files = dir(imagePath);

for i = 1:length(files)
    % 如果是目录则跳过
    if isequal(files(i).name, '.')||isequal(files(i).name, '..')||files(i).isdir
        continue;
    end
    im = imread(fullfile(imagePath, files(i).name));

    s1 = 11;        % patch size for scale 1
    s2 = 15;        % patch size for scale 2
    s3 = 21;        % patch size for scale 3

    alphaval = 0.5;    % weight for multiscale inference

    blurSensitivity = 0.006;  %T_lbp

    fprintf('Extracting scale 1 feature...\n');
    feature.scale1 = localSharpScoreLBP(im, s1, blurSensitivity);
    fprintf('Extracting scale 2 feature...\n');
    feature.scale2 = localSharpScoreLBP(im, s2, blurSensitivity);
    fprintf('Extracting scale 3 feature...\n');
    feature.scale3 = localSharpScoreLBP(im, s3, blurSensitivity);

    fprintf('Multiscale Inference...\n');
    final_map = multiScaleBlurInferenceFull(feature, alphaval);

    close all;
    figure(1),imshow(im);
    figure(2),imshow(final_map);
    
    imwrite(final_map, fullfile(segPath, files(i).name));
end
end
end