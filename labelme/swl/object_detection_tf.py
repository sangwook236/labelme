import os, time
import numpy as np
import tensorflow as tf
import cv2 as cv
import labelme.util

# REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/tensorflow/tensorflow_object_detection.py
# REF [site] >> https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
def detect_objects_by_faster_rcnn(image_filepath):
	#model_url = 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz'
	#model_url = 'http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gzz'
	model_url = 'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz'

	output_dir_path = './pretrained_model'
	model_filename = 'frozen_inference_graph.pb'

	model_dir_name = model_url.split('/')[-1].split('.')[0]
	model_filepath = os.path.join(output_dir_path, model_dir_name, model_filename)
	if not os.path.exists(model_filepath) or not os.path.isfile(model_filepath):
		print('Start downloading model files...')
		start_time = time.time()
		labelme.util.download(model_url, output_dir_path)
		print('End downloading model files to {}: {} secs.'.format(model_filepath, time.time() - start_time))

	"""
	model: Binary file contains trained weights.
		The following file extensions are expected for models from different frameworks:
			*.caffemodel (Caffe, http://caffe.berkeleyvision.org/)
			*.pb (TensorFlow, https://www.tensorflow.org/)
			*.t7 | *.net (Torch, http://torch.ch/)
			*.weights (Darknet, https://pjreddie.com/darknet/)
			*.bin (DLDT, https://software.intel.com/openvino-toolkit)
			*.onnx (ONNX, https://onnx.ai/)
	"""
	class_filepath = None  # A text file with names of classes.
	color_filepath = None  # A text file with colors for an every class. An every color is represented with three values from 0 to 255 in BGR channels order.

	confidence_threshold = 0.3  # Confidence threshold.
	#nms_threshold = 0.4  # Non-maximum suppression threshold.

	#image_filepath = './coco.jpgs'
	img = cv.imread(image_filepath)
	if img is None:
		print('Failed to load an image, {}.'.format(image_filepath))
		return
	image_width, image_height = 300, 300

	# Load names of classes.
	if class_filepath:
		with open(class_filepath, 'rt') as fd:
			class_names = fd.read().rstrip('\n').split('\n')

		num_classes = len(class_names)
	else:
		class_names = None
		num_classes = 80  # For COCO dataset.

	# Load or generate colors.
	if color_filepath:
		with open(color_filepath, 'rt') as fd:
			colors = [np.array(color.split(' '), np.uint8) for color in fd.read().rstrip('\n').split('\n')]
	else:
		colors = [np.array([0, 0, 0], np.uint8)]
		for i in range(1, num_classes + 1):
			colors.append((colors[i - 1] + np.random.randint(0, 256, [3], np.uint8)) / 2)
		del colors[0]

	# Read the graph.
	with tf.gfile.FastGFile(model_filepath, 'rb') as fd:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(fd.read())

	with tf.Session() as sess:
		# Restore session.
		sess.graph.as_default()
		tf.import_graph_def(graph_def, name='')

		# Read and preprocess an image.
		rows, cols = img.shape[:2]
		inp = cv.resize(img, (image_width, image_height))
		inp = inp[:, :, [2, 1, 0]]  # BGR -> RGB.

		# Run the model.
		out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
			sess.graph.get_tensor_by_name('detection_scores:0'),
			sess.graph.get_tensor_by_name('detection_boxes:0'),
			sess.graph.get_tensor_by_name('detection_classes:0')],
			feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

	# Visualize detected bounding boxes.
	objects = list()
	num_detections = int(out[0][0])
	for i in range(num_detections):
		class_id = int(out[3][0][i])
		score = float(out[1][0][i])
		bbox = [float(v) for v in out[2][0][i]]
		if score > confidence_threshold:
			left, top, right, bottom = max(int(bbox[1] * cols), 0), max(int(bbox[0] * rows), 0), min(int(bbox[3] * cols), cols - 1), min(int(bbox[2] * rows), rows - 1)
			#cv.rectangle(img, (left, top), (right, bottom), (125, 255, 51), thickness=2)

			bbox_points = np.array([(left, top), (right, bottom)], dtype=np.int32)
			objects.append((class_names[class_id] if class_names else None, bbox_points, 'rectangle'))

	#cv.imshow('Object Detection', img)
	#cv.waitKey()

	return objects

# REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/tensorflow/tensorflow_object_detection.py
# REF [site] >> https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
def detect_objects_by_mask_rcnn(image_filepath):
	model_url = 'http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz'

	output_dir_path = './pretrained_model'
	model_filename = 'frozen_inference_graph.pb'

	model_dir_name = model_url.split('/')[-1].split('.')[0]
	model_filepath = os.path.join(output_dir_path, model_dir_name, model_filename)
	if not os.path.exists(model_filepath) or not os.path.isfile(model_filepath):
		print('Start downloading model files...')
		start_time = time.time()
		labelme.util.download(model_url, output_dir_path)
		print('End downloading model files to {}: {} secs.'.format(model_filepath, time.time() - start_time))

	"""
	model: Binary file contains trained weights.
		The following file extensions are expected for models from different frameworks:
			*.caffemodel (Caffe, http://caffe.berkeleyvision.org/)
			*.pb (TensorFlow, https://www.tensorflow.org/)
			*.t7 | *.net (Torch, http://torch.ch/)
			*.weights (Darknet, https://pjreddie.com/darknet/)
			*.bin (DLDT, https://software.intel.com/openvino-toolkit)
			*.onnx (ONNX, https://onnx.ai/)
	"""
	class_filepath = None  # A text file with names of classes.
	color_filepath = None  # A text file with colors for an every class. An every color is represented with three values from 0 to 255 in BGR channels order.

	confidence_threshold = 0.3  # Confidence threshold.
	#nms_threshold = 0.4  # Non-maximum suppression threshold.
	mask_threshold = 0.5

	#image_filepath = './coco.jpg'
	img = cv.imread(image_filepath)
	if img is None:
		print('Failed to load an image, {}.'.format(image_filepath))
		return
	image_width, image_height = 300, 300

	# Load names of classes.
	if class_filepath:
		with open(class_filepath, 'rt') as fd:
			class_names = fd.read().rstrip('\n').split('\n')

		num_classes = len(class_names)
	else:
		class_names = None
		num_classes = 80  # For COCO dataset.

	# Load or generate colors.
	if color_filepath:
		with open(color_filepath, 'rt') as fd:
			colors = [np.array(color.split(' '), np.uint8) for color in fd.read().rstrip('\n').split('\n')]
	else:
		colors = [np.array([0, 0, 0], np.uint8)]
		for i in range(1, num_classes + 1):
			colors.append((colors[i - 1] + np.random.randint(0, 256, [3], np.uint8)) / 2)
		del colors[0]

	# Read the graph.
	with tf.gfile.FastGFile(model_filepath, 'rb') as fd:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(fd.read())

	with tf.Session() as sess:
		# Restore session.
		sess.graph.as_default()
		tf.import_graph_def(graph_def, name='')

		# Read and preprocess an image.
		rows, cols = img.shape[:2]
		inp = cv.resize(img, (image_width, image_height))
		inp = inp[:, :, [2, 1, 0]]  # BGR -> RGB.

		# Run the model.
		out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
			sess.graph.get_tensor_by_name('detection_scores:0'),
			sess.graph.get_tensor_by_name('detection_boxes:0'),
			sess.graph.get_tensor_by_name('detection_classes:0'),
			sess.graph.get_tensor_by_name('detection_masks:0')],
			feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

	# Visualize detected bounding boxes and masks.
	objects = list()
	num_detections = int(out[0][0])
	for i in range(num_detections):
		class_id = int(out[3][0][i])
		score = float(out[1][0][i])
		bbox = [float(v) for v in out[2][0][i]]
		mask = out[4][0][i]
		if score > confidence_threshold:
			left, top, right, bottom = max(int(bbox[1] * cols), 0), max(int(bbox[0] * rows), 0), min(int(bbox[3] * cols), cols - 1), min(int(bbox[2] * rows), rows - 1)
			#cv.rectangle(img, (left, top), (right, bottom), (125, 255, 51), 2, cv.LINE_AA)

			mask = cv.resize(mask, (right - left + 1, bottom - top + 1))
			mask = mask > mask_threshold

			contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
			contours = list(map(lambda xy: xy + [left, top], contours))
			##cv.drawContours(img, contours, 0, (0, 0, 255), 2)
			#cv.drawContours(img, contours, -1, (0, 0, 255), 2)

			#roi = img[top:bottom+1, left:right+1][mask]
			#img[top:bottom+1, left:right+1][mask] = (0.7 * colors[class_id] + 0.3 * roi).astype(np.uint8)

			#bbox_points = np.array([(left, top), (right, bottom)], dtype=np.int32)
			#objects.append((class_names[class_id] if class_names else None, bbox_points, 'rectangle'))
			boundary_points = contours[0][:,0,:].astype(np.int32)
			objects.append((class_names[class_id] if class_names else None, boundary_points, 'polygon'))

	#cv.imshow('Mask R-CNN', img)
	#cv.waitKey()

	return objects
