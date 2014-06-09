function features_neg = get_random_negative_features(...
    non_face_scn_path, feature_params, num_samples)
% get random features from non-face images of multiple scales
% 'non_face_scn_path' is a string. This directory contains non-face images.
% 'feature_params' is a struct, with fields feature_params.template_size, 
% the number of pixels spanned by each train / test template and
% feature_params.hog_cell_size, the number of pixels in each
% HoG cell. template size should be evenly divisible by hog_cell_size.
% num_samples is the expected sampled negative features.
% the returned features_neg will not be exactly the same with num_samples.
%
% 'features_neg' is N by D matrix where N is the number of negative features 
% and D is the template dimensionality, which is
% (feature_params.template_size / feature_params.hog_cell_size) ^ 2 * 31

ext = '.jpg';
wildcard = '*';
image_files = dir(fullfile(non_face_scn_path, ...
                           strcat(wildcard, ext)));
num_images = length(image_files);
features_dim = (feature_params.template_size ...
                / feature_params.hog_cell_size) ^ 2 * 31;

% preallocate space to avoid frequent array expanding
features_neg = zeros(num_samples + 20000, features_dim);
samples_count = 0;

num_scales = 10;
scales = linspace(0.01, 1, num_scales);
samples_per_scale = round(linspace(0, num_samples / num_images / 4, num_scales));

for i = 1:num_images
    img = imread(fullfile(non_face_scn_path, image_files(i).name));
    
    % because the face examples are gray, so also convert non-face images gray
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = im2single(img);
    
    % print some information about the sampling process
    % because it will take a relatively long time
    fprintf('sampling from the %dth image, already have %d negative features.\n', ...
            i, samples_count);
    
    for j = num_scales:-1:1
        img_scale = imresize(img, scales(j));
        [height, width] = size(img_scale);
        if height < feature_params.template_size ...
           || width < feature_params.template_size
            break;
        end
        
        samples_cur_scale = min(samples_per_scale(j), ...
            (height - feature_params.template_size + 1) * ...
            (width - feature_params.template_size + 1));
        
        for k = 1:samples_cur_scale
            left_x = randi([1, width - feature_params.template_size + 1]);
            upper_y = randi([1, height - feature_params.template_size + 1]);
            
            img_sample = ...
            img_scale(upper_y:upper_y + feature_params.template_size - 1, ...
                      left_x:left_x + feature_params.template_size - 1);
            hog = vl_hog(img_sample, feature_params.hog_cell_size);
            samples_count = samples_count + 1;
            features_neg(samples_count, :) = hog(:)';
        end
    end
end

features_neg = features_neg(1:samples_count, :);

end