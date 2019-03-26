close all;
data_root = '/home/jason/matlab/PSPNet/data/VOCdevkit/VOC2012/JPEGImages/';
res_list = fullfile(pwd,'mc_result/VOC2012/test/find_class/');
ori_list = fullfile(pwd,'mc_result/VOC2012/test/ori_img/');
save_root = 'mc_result/VOC2012/test/pspnet101_473/color/';
img_dir = fullfile(pwd,save_root);
img_list = dir([img_dir '*.png']);
n=1;
load('./visualizationCode/objectName21.mat');
while n <=  length(img_list)
   img_name = img_list(n).name;
   img_fullname = fullfile(img_dir,img_name);
   [im,map] = imread(img_fullname);
   uni_num = unique(im);
   count = find(uni_num==9);
   class_total = '';
   for i = 1 : length(uni_num);
       class_total = [class_total ' ' objectNames(uni_num(i)+1)];
   end
   if(~isempty(count))
%        imwrite(im,colormap,[res_list img_name]);

       im2 = imread([data_root img_name(1:end-4) '.jpg']);
       subplot(1,2,1);  imshow(im,map); title(class_total);
       subplot(1,2,2); imshow(im2);
       [x,y,button] = ginput(1);
       if(button==30) n = n-2; end
       
%        imwrite(im2,[ori_list img_name ]);
   end
   n = n +1;
   disp(n);
end
