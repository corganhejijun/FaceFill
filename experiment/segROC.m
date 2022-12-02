gt_path = './images/GT';
GRUBCUT = 1;
MM = 2;
ONECUT = 3;
dataType = GRUBCUT;

if dataType == MM
    type = 'GC';
    img_path = ['./images/',type,'_origin'];
    test_path = ['./images/',type,'_test'];
elseif dataType == GRUBCUT
    img_path = './grabcut/origin_seg';
    test_path = './grabcut/test_seg';
elseif dataType == ONECUT
    img_path = './images/oneCut_origin';
    test_path = './images/oneCut_test';
end

files = dir(img_path);
FPR = zeros(1,256);
TPR = zeros(1,256);
FPR_test = zeros(1,256);
TPR_test = zeros(1,256);
P_list = [];
R_list = [];
F_list = [];
P_test_list = [];
R_test_list = [];
F_test_list = [];
for i = 1:length(files)
    % 如果是目录则跳过
    if isequal(files(i).name, '.')||isequal(files(i).name, '..')||files(i).isdir
        continue;
    end
    gt = imread(fullfile(gt_path, [files(i).name(1:length(files(i).name)-4),'.png']));
    if dataType == MM
        fileName = [files(i).name(1:length(files(i).name)-4),'_',type,'.png'];
        img = imread(fullfile(img_path, fileName));
        test = imread(fullfile(test_path, fileName));
    elseif dataType == GRUBCUT
        fileName = files(i).name(1:length(files(i).name)-4);
        img = imread(fullfile(img_path, [fileName, '.jpg']));
        test = imread(fullfile(test_path, [fileName, '.png']));
    elseif dataType == ONECUT
        fileName = files(i).name;
        img = imread(fullfile(img_path, fileName));
        test = imread(fullfile(test_path, fileName));
    end
%% 将灰度分为256个阈值，根据不同阈值进行分割，之后计算AUC
    begin = 128;
    endIdx = 128;
    P = [];
    R = [];
    F = [];
    P_test = [];
    R_test = [];
    F_test = [];
    for j = endIdx:-1:begin
        img_bi = img>j-1;
        gt_bi = gt>0;
        test_bi = test>j-1;
        TP = sum(sum(gt_bi&img_bi));     %预测为正但实际为正的样本
        FP = sum(sum(img_bi&~gt_bi));    %预测为正但实际为负的样本
        FN = sum(sum(~img_bi&gt_bi));    %预测为负但实际为正的样本
        TP_test = sum(sum(gt_bi&test_bi));
        FP_test = sum(sum(test_bi&~gt_bi));
        FN_test = sum(sum(~test_bi&gt_bi));
        P(length(P)+1) = TP / (TP + FP);
        R(length(R)+1) = TP / (TP + FN);
        F(length(F)+1) = 2 * P * R / (P + R);
        P_test(length(P_test)+1) = TP_test / (TP_test + FP_test);
        R_test(length(R_test)+1) = TP_test / (TP_test + FN_test);
        F_test(length(F_test)+1) = 2 * P_test * R_test / (P_test + R_test);
        GT = sum(gt_bi(:));
        GTM = sum(sum(~gt_bi));
        if FPR(j) == 0
            FPR(j) = FP/GTM;    %预测为正但实际为负的样本占所有负例样本的比例
        else
            FPR(j) = (FPR(j)+FP/GTM)/2;
        end
        if TPR(j) == 0
            TPR(j) = TP/GT;     %预测为正且实际为正的样本占所有正例样本的比例
        else
            TPR(j) = (TPR(j)+TP/GT)/2;
        end
        if FPR_test(j) == 0
            FPR_test(j) = FP_test/GTM;
        else
            FPR_test(j) = (FPR_test(j)+FP_test/GTM)/2;
        end
        if TPR_test(j) == 0
            TPR_test(j) = TP_test/GT;
        else
            TPR_test(j) = (TPR_test(j)+TP_test/GT)/2;
        end
    end
    P_list(length(P_list)+1) = mean(P);
    R_list(length(R_list)+1) = mean(R);
    F_list(length(F_list)+1) = mean(F);
    P_test_list(length(P_test_list)+1) = mean(P_test);
    R_test_list(length(R_test_list)+1) = mean(R_test);
    F_test_list(length(F_test_list)+1) = mean(F_test);
    fprintf('file name is %s\n', files(i).name);
end
TPR = [1,TPR];
FPR = [1,FPR];
TPR_test = [1,TPR_test];
FPR_test = [1,FPR_test];
%%
AUC = 0;
AUC_test = 0;
for i = 1:256
    AUC = AUC + (TPR(i) + TPR(i+1))*(FPR(i)-FPR(i+1))/2;
    AUC_test = AUC_test + (TPR_test(i) + TPR_test(i+1))*(FPR_test(i)-FPR_test(i+1))/2;
end
fprintf('AUC = %f, AUC_test = %f\n', AUC, AUC_test);
fprintf('P = %f, R = %f, F = %f\n', mean(P_list), mean(R_list), mean(F_list));
fprintf('P_test = %f, R_test = %f, F_test = %f\n', mean(P_test_list), mean(R_test_list), mean(F_test_list));

%%
figDisplay(FPR, TPR, FPR_test, TPR_test);