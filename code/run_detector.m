function [bboxes, confidences, image_ids] = run_detector(...
    test_scn_path, w, b, feature_params, ...
    conf_threshold, step_size, scale_factor)
% 'test_scn_path' is a string. This directory contains test images.
% 'w' and 'b' are the SVM classifier parameters
% 'feature_params' is a struct, with fields feature_params.template_size, 
% the number of pixels spanned by each train / test template and
% feature_params.hog_cell_size, the number of pixels in each
% HoG cell. template size should be evenly divisible by hog_cell_size.
%
% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
% [x_min, y_min, x_max, y_max] for detection i.
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
% detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
% for detection i.

ext = '.jpg';
wildcard = '*';
test_scenes = dir(fullfile(test_scn_path, strcat(wildcard, ext)));

num_images = length(test_scenes);

features_dim = (feature_params.template_size / feature_params.hog_cell_size) ^ 2 * 31;

% set initial scale
init_scale = 1;

% pre-allocate space to avoid frequent array expanding
face_count = 0;
init_capacity_cur = 1000;
init_capacity_all = 10000;
bboxes = zeros(init_capacity_all, 4);
confidences = zeros(init_capacity_all, 1);
image_ids = cell(init_capacity_all, 1);

for i = 1:num_images
    
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = im2single(...
            imread(fullfile(test_scn_path, test_scenes(i).name)));
    
    % if img is RGB, then convert to gray
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
    cur_bboxes = zeros(init_capacity_cur, 4);
    cur_confidences = zeros(init_capacity_cur, 1);
    cur_image_ids = cell(init_capacity_cur, 1);
    cur_count = 0;
    
    scale = init_scale;
    
    img_scale = imresize(img, scale);
    [height, width] = size(img_scale);
    
    while height >= feature_params.template_size ...
       && width >= feature_params.template_size
   
        x_end = width - feature_params.template_size + 1;
        y_end = height - feature_params.template_size + 1;
        
        
        for x = 1:step_size:x_end
            for y = 1:step_size:y_end
                hog = vl_hog(img_scale(y:y + feature_params.template_size - 1, ...
                                 x:x + feature_params.template_size - 1), ...
                             feature_params.hog_cell_size);
                hog = reshape(hog, 1, features_dim);
                confidence = hog * w+ b;
                if confidence > conf_threshold
                    cur_count = cur_count + 1;
                    cur_bboxes(cur_count, :) = [x, y, ...
                                 x + feature_params.template_size - 1, ...
                                 y + feature_params.template_size - 1] / scale;
                    cur_confidences(cur_count, :) = confidence;
                    cur_image_ids{cur_count, :} = test_scenes(i).name;
                end
            end
        end
        
        scale = scale * scale_factor;
        img_scale = imresize(img, scale);
        [height, width] = size(img_scale);
    end
    
    cur_bboxes = cur_bboxes(1:cur_count, :);
    cur_confidences = cur_confidences(1:cur_count, :);
    cur_image_ids = cur_image_ids(1:cur_count, :);
    
    % non-maxima suppression to select the strongest response
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));
    
    cur_confidences = cur_confidences(is_maximum, :);
    cur_bboxes = cur_bboxes(is_maximum, :);
    cur_image_ids = cur_image_ids(is_maximum, :);
    
    bboxes(face_count + 1:face_count + length(cur_confidences), :) = cur_bboxes;
    confidences(face_count + 1:face_count + length(cur_confidences), :) = cur_confidences;
    image_ids(face_count + 1:face_count + length(cur_confidences), :) = cur_image_ids;
    
    face_count = face_count + length(cur_confidences);
end

bboxes = bboxes(1:face_count, :);
confidences = confidences(1:face_count, :);
image_ids = image_ids(1:face_count, :);

end