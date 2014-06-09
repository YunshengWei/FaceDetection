%% Sliding window face detection with linear SVM.

%% set up environment variables. 
close all;
clear;
run('vlfeat/toolbox/vl_setup');

[~, ~, ~] = mkdir('visualizations');

data_path = '../data/';
train_path_pos = fullfile(data_path, 'caltech_faces/Caltech_CropFaces');
non_face_scn_path = fullfile(data_path, 'train_non_face_scenes');
test_scn_path = fullfile(data_path, 'test_scenes/test_jpg');
label_path = fullfile(data_path, 'test_scenes/ground_truth_bboxes.txt');

% All parameters I need to tune lie below
% set parameters of HoG
feature_params = struct('template_size', 36, 'hog_cell_size', 4);
% penalty parameter for SVM
lambda = 0.0001;
% confidence threshold for face candidates
conf_threshold = 0.7;
% stepsize for moving sliding window
step_size = 4;
% scale factor for resizing image
scale_factor = 0.8;

%% Load positive training crops and random negative examples

features_pos = get_positive_features(train_path_pos, feature_params);

num_negative_examples = 30000;
features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_negative_examples);
    
%% Train Classifier

X = [features_pos', features_neg'];
num_pos = size(features_pos, 1);
num_neg = size(features_neg, 1);

Y = [ones(num_pos, 1);
     ones(num_neg, 1) * -1];
[w, b] = vl_svmtrain(X, Y, lambda);

% save the trained classifier
save('detector.mat', 'w', 'b');
save('detector.mat', '-struct', 'feature_params', '-append');

% for memory efficiency, clear intermediate variables
clear X Y

%% Examine learned classifier

fprintf('Initial classifier performance on train data:\n');

confidences = [features_pos; features_neg] * w + b;
label_vector = [ones(num_pos, 1);
                ones(num_neg, 1) * -1];
[tp_rate, fp_rate, tn_rate, fn_rate] = report_accuracy(confidences, label_vector);

% Visualize the learned HoG detector.
n_hog_cells = sqrt(length(w) / 31);
imhog = vl_hog('render', single(reshape(w, [n_hog_cells, n_hog_cells, 31])), 'verbose') ;
figure(3);
imagesc(imhog); 
colormap gray; 
set(3, 'Color', [.988, .988, .988]);

pause(0.1);
hog_template_image = frame2im(getframe(3));
imwrite(hog_template_image, 'visualizations/hog_template.png')

%% Run detector on test set.
[bboxes, confidences, image_ids] = run_detector(test_scn_path, ...
    w, b, feature_params, conf_threshold, step_size, scale_factor);

%% Evaluate and Visualize detections

[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections(bboxes, confidences, image_ids, label_path);

visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp, test_scn_path, label_path);
