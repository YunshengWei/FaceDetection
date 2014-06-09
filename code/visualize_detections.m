function visualize_detections(bboxes, confidences, filename)
% visualize all detections in an image

[~, image_name, ext] = fileparts(filename);

hold on;

num_detections = length(confidences);
for i = 1:num_detections
    bb = bboxes(i, :);
    plot(bb([1, 3, 3, 1, 1]), bb([2, 2, 4, 4, 2]), 'g:', 'linewidth', 2);
end

hold off;
axis image;
axis off;
title(sprintf('image: "%s" \n green = detection', strcat(image_name, ext)), 'interpreter', 'none');
set(gca, 'Color', [0.988, 0.988, 0.988]);
pause(0.1);

% save detection results
detection_image = frame2im(getframe(gca));
imwrite(detection_image, sprintf('visualizations/detections_%s.png', strcat(image_name, ext)))

end