import math
import numpy as np
import detectron2
import detectron2.model_zoo, detectron2.config, detectron2.engine, detectron2.evaluation, detectron2.utils.visualizer, detectron2.data, detectron2.structures
import cv2

# REF [function] >> simple_detection_example() in ${SWDT_PYTHON_HOME}/rnd/test/object_detection/detectron2_test.py
def detect_objects_by_faster_rcnn(image_filepath):
	im = cv2.imread(image_filepath)
	if im is None:
		print('Failed to load an image: {}.'.format(image_filepath))
		return None
	#cv2.imshow('Image', im)

	confidence_threshold = 0.3  # Confidence threshold.
	#nms_threshold = 0.4  # Non-maximum suppression threshold.

	cfg = detectron2.config.get_cfg()
	# Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library.
	cfg.merge_from_file(detectron2.model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model.
	# Find a model from detectron2's model zoo.
	# You can use the https://dl.fbaipublicfiles... url as well.
	cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')

	predictor = detectron2.engine.DefaultPredictor(cfg)
	outputs = predictor(im)

	objects = list()
	for class_id, score, bbox in zip(outputs['instances'].pred_classes, outputs['instances'].scores, outputs['instances'].pred_boxes):
		if score > confidence_threshold:
			left, top, right, bottom = max(math.floor(bbox[0]), 0), max(math.floor(bbox[1]), 0), min(math.ceil(bbox[2]), im.shape[1] - 1), min(math.ceil(bbox[3]), im.shape[0] - 1)

			bbox_points = np.array([(left, top), (right, bottom)], dtype=np.int32)
			#objects.append((class_names[class_id] if class_names else None, bbox_points, 'rectangle'))
			objects.append((class_id, bbox_points, 'rectangle'))

	return objects

# REF [function] >> simple_keypoint_detection_example() in ${SWDT_PYTHON_HOME}/rnd/test/object_detection/detectron2_test.py
def detect_objects_by_mask_rcnn(image_filepath):
	im = cv2.imread(image_filepath)
	if im is None:
		print('Failed to load an image: {}.'.format(image_filepath))
		return None
	#cv2.imshow('Image', im)

	confidence_threshold = 0.3  # Confidence threshold.
	#nms_threshold = 0.4  # Non-maximum suppression threshold.
	#mask_threshold = 0.5

	# Infer with a keypoint detection model.
	cfg = detectron2.config.get_cfg()
	cfg.merge_from_file(detectron2.model_zoo.get_config_file('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'))
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set threshold for this model.
	cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml')

	predictor = detectron2.engine.DefaultPredictor(cfg)
	outputs = predictor(im)

	objects = list()
	for class_id, score, bbox in zip(outputs['instances'].pred_classes, outputs['instances'].scores, outputs['instances'].pred_boxes):
		if score > confidence_threshold:
			left, top, right, bottom = max(math.floor(bbox[0]), 0), max(math.floor(bbox[1]), 0), min(math.ceil(bbox[2]), im.shape[1] - 1), min(math.ceil(bbox[3]), im.shape[0] - 1)

			bbox_points = np.array([(left, top), (right, bottom)], dtype=np.int32)
			#objects.append((class_names[class_id] if class_names else None, bbox_points, 'rectangle'))
			objects.append((class_id, bbox_points, 'rectangle'))

	return objects

if '__main__' == __name__:
	image_filepath = './labelme/swl/2011_000006.jpg' 
	objects = detect_objects_by_faster_rcnn(image_filepath)
	#objects = detect_objects_by_mask_rcnn(image_filepath)
