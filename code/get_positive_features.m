function features_pos = get_positive_features(train_path_pos, feature_params)
% 'train_path_pos' is a string. This directory contains 36x36 images of faces
% 'feature_params' is a struct, with fields feature_params.template_size, 
% the number of pixels spanned by each train / test template and
% feature_params.hog_cell_size, the number of pixels in each
% HoG cell. template size should be evenly divisible by hog_cell_size.
% 
% 'features_pos' is N by D matrix where N is the number of faces and D
% is the template dimensionality, which is
% (feature_params.template_size / feature_params.hog_cell_size) ^ 2 * 31

% Caltech Faces stored as '.jpg'
ext = '.jpg';
wildcard = '*';
image_files = dir(fullfile(train_path_pos, strcat(wildcard, ext)));

num_images = length(image_files);
features_dim = (feature_params.template_size / feature_params.hog_cell_size) ^ 2 * 31;
features_pos = zeros(num_images, features_dim);

for i = 1:num_images
    % read image and convert to single (since vl_hog takes image of class 'single')
    I = im2single(...
            imread(fullfile(train_path_pos, image_files(i).name)));
    % compute HOG features
    hog = vl_hog(I, feature_params.hog_cell_size);
    % reshape hog into one row
    features_pos(i, :) = hog(:)';
end

end