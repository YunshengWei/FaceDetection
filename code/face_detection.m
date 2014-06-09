function face_detection(image_to_detect, varargin)
% Sliding window face detection with linear SVM and HOG features.
% the last 3 arguments are optional
% Usage example:
%     face_detection('abc.jpg', 'conf_threshold', 0.7, 'step_size', 4, 'scale_factor', 0.8);
% or just:
%     face_detection('abc.jpg');

close all;
% need to use 'vl_hog' in this library
run('vlfeat/toolbox/vl_setup');

% directory to save the detection results
if ~exist('visualizations', 'dir')
    mkdir('visualizations');
end

% handle optional arguments
for i = 1:length(varargin)
    if strcmp(varargin{i}, 'step_size') == 1
        step_size = varargin{i + 1};
    elseif strcmp(varargin{i}, 'conf_threshold') == 1
        conf_threshold = varargin{i + 1};
    elseif strcmp(varargin{i}, 'scale_factor') == 1
        scale_factor = varargin{i + 1};
    end
end

detector = load('detector.mat');

% set default parameters if not given
if ~exist('conf_threshold', 'var') || isempty(conf_threshold)
    conf_threshold = 0.7;
end
if ~exist('step_size', 'var') || isempty(step_size)
    step_size = detector.hog_cell_size;
end
if ~exist('scale_factor', 'var') || isempty(scale_factor)
    scale_factor = 0.8;
end

% print parameters information
fprintf('\n');
fprintf('hog template size: %d\n', detector.template_size);
fprintf('hog cell size: %d\n', detector.hog_cell_size);
fprintf('confidence threshold: %.1f\n', conf_threshold);
fprintf('step size for sliding window: %d\n', step_size);
fprintf('scale factor: %.1f\n\n', scale_factor);
fprintf('Begin detecting, please wait, waiting time depends on the size of image\n');

figure();
imshow(imread(image_to_detect));
% let's UI rendering catch up
pause(0.1);
[bboxes, confidences] = sliding_window_detector(image_to_detect, detector, ...
    conf_threshold, step_size, scale_factor);
visualize_detections(bboxes, confidences, image_to_detect);
fprintf('Detection Succeeded.\n')

end