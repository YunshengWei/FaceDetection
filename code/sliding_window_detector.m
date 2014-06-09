function [bboxes, confidences] = sliding_window_detector(imagefile, ...
    detector, conf_threshold, step_size, scale_factor)
% a multiscale sliding window dector for face detection.
% 'bboxes' is N * 4. N is the number of detections.
% bboxes(i, :) is [x_min, y_min, x_max, y_max] for detection i.
% 'confidences' is N * 1.
% confidences(i, 1) is confidence for detection i.


% vl_hog needs image to be of class 'single'
image = im2single(imread(imagefile));

% if image is RGB, convert to gray
if (size(image, 3) > 1)
    image = rgb2gray(image);
end

% preallocate to avoid frequent expanding
bboxes = zeros(500, 4);
confidences = zeros(500, 1);

% sliding window detector

face_counts = 0;

% set_initial scale
init_scale = 1;
scale = init_scale;

image_scale = imresize(image, scale);
[height, width] = size(image_scale);
while height >= detector.template_size ...
   && width >= detector.template_size
 
    x_end = width - detector.template_size + 1;
    y_end = height - detector.template_size + 1;
    for x = 1:step_size:x_end
        for y = 1:step_size:y_end
            x_min = x;
            y_min = y;
            x_max = x + detector.template_size - 1;
            y_max = y + detector.template_size - 1;
            image_sample = image_scale(y_min:y_max, x_min:x_max);
            hog = vl_hog(image_sample, detector.hog_cell_size);
            % hog(:)' is faster than reshape(hog, 1, [])
            hog = hog(:)';
            confidence = hog * detector.w + detector.b;
            if confidence >= conf_threshold
                face_counts = face_counts + 1;
                bboxes(face_counts, :) = [x_min, y_min, x_max, y_max] / scale;
                confidences(face_counts) = confidence;
            end
        end
    end
    
    scale = scale * scale_factor;
    image_scale = imresize(image, scale);
    [height, width] = size(image_scale);
end

bboxes = bboxes(1:face_counts, :);
confidences = confidences(1:face_counts, :);

% non-max suppression
[is_maximum] = non_max_supr_bbox(bboxes, confidences, size(image));
confidences = confidences(is_maximum, :);
bboxes = bboxes(is_maximum, :);

end