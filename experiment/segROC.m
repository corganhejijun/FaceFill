gt_path = './images/GT';
type = 'CSD';
img_path = ['./images/',type,'_origin'];
test_path = ['./images/',type,'_test'];

files = dir(gt_path);
FPR = zeros(1,256);
TPR = zeros(1,256);
FPR_test = zeros(1,256);
TPR_test = zeros(1,256);
for i = 1:length(files)
    % 如果是目录则跳过
    if isequal(files(i).name, '.')||isequal(files(i).name, '..')||files(i).isdir
        continue;
    end
    fileName = [files(i).name(1:length(files(i).name)-4),'_',type,'.png'];
    img = imread(fullfile(img_path, fileName));
    gt = imread(fullfile(gt_path, files(i).name));
    test = imread(fullfile(test_path, fileName));
    for j = 256:-1:1
        img_bi = img>j-1;
        gt_bi = gt>0;
        test_bi = test>j-1;
        TP = sum(sum(gt_bi&img_bi));     %预测为正但实际为负的样本
        FP = sum(sum(img_bi&~gt_bi));    %预测为正但实际为负的样本
        TP_test = sum(sum(gt_bi&test_bi));
        FP_test = sum(sum(test_bi&~gt_bi));
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
    fprintf('file name is %s\n', files(i).name);
end

%%
AUC = 0;
AUC_test = 0;
for i = 1:255
    AUC = AUC + (TPR(i) + TPR(i+1))*(FPR(i)-FPR(i+1))/2;
    AUC_test = AUC_test + (TPR_test(i) + TPR_test(i+1))*(FPR_test(i)-FPR_test(i+1))/2;
end
fprintf('AUC = %f, AUC_test = %f\n', AUC, AUC_test);

%%
figDisplay(FPR, TPR, FPR_test, TPR_test);