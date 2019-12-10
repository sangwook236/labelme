import os, math, time
import numpy as np
import tensorflow as tf
import cv2 as cv
import keras_ocr
import util

# REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/language_processing/keras_ocr_test.py
def detect_text_by_craft(image_filepath):
	detector = keras_ocr.detection.Detector(pretrained=True)

	image = keras_ocr.tools.read(image_filepath)

	# Boxes will be an Nx4x2 array of box quadrangles, where N is the number of detected text boxes.
	bboxes = detector.detect(images=[image])[0]
	canvas = keras_ocr.detection.drawBoxes(image, bboxes)

	#plt.imshow(canvas)
	#plt.show()

	texts = list()
	for bbox in bboxes:
		texts.append(('text', bbox, 'polygon'))

	return texts

def decode(scores, geometry, scoreThresh):
	############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
	assert len(scores.shape) == 4, 'Incorrect dimensions of scores'
	assert len(geometry.shape) == 4, 'Incorrect dimensions of geometry'
	assert scores.shape[0] == 1, 'Invalid dimensions of scores'
	assert geometry.shape[0] == 1, 'Invalid dimensions of geometry'
	assert scores.shape[1] == 1, 'Invalid dimensions of scores'
	assert geometry.shape[1] == 5, 'Invalid dimensions of geometry'
	assert scores.shape[2] == geometry.shape[2], 'Invalid dimensions of scores and geometry'
	assert scores.shape[3] == geometry.shape[3], 'Invalid dimensions of scores and geometry'

	detections = []
	confidences = []

	height = scores.shape[2]
	width = scores.shape[3]
	for y in range(0, height):
		# Extract data from scores.
		scoresData = scores[0][0][y]
		x0_data = geometry[0][0][y]
		x1_data = geometry[0][1][y]
		x2_data = geometry[0][2][y]
		x3_data = geometry[0][3][y]
		anglesData = geometry[0][4][y]
		for x in range(0, width):
			score = scoresData[x]

			# If score is lower than threshold score, move to next x.
			if(score < scoreThresh):
				continue

			# Calculate offset.
			offsetX = x * 4.0
			offsetY = y * 4.0
			angle = anglesData[x]

			# Calculate cos and sin of angle.
			cosA = math.cos(angle)
			sinA = math.sin(angle)
			h = x0_data[x] + x2_data[x]
			w = x1_data[x] + x3_data[x]

			# Calculate offset.
			offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

			# Find points for rectangle.
			p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
			p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
			center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
			detections.append((center, (w, h), -angle * 180.0 / math.pi))
			confidences.append(float(score))

	# Return detections and confidences.
	return [detections, confidences]

# REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/machine_vision/opencv/opencv_text.py
def detect_objects_by_east(image_filepath):
	# REF [site] >> https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/download_models.py
	model_url = 'https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1'

	output_dir_path = './pretrained_model'
	model_filename = 'frozen_east_text_detection.pb'

	model_filepath = os.path.join(output_dir_path, model_filename)
	if not os.path.exists(model_filepath) or not os.path.isfile(model_filepath):
		print('Start downloading model files...')
		start_time = time.time()
		util.download(model_url, output_dir_path)
		print('End downloading model files to {}: {} secs.'.format(model_filepath, time.time() - start_time))

	confThreshold = 0.5  # Confidence threshold.
	nmsThreshold = 0.4  # Non-maximum suppression threshold.
	inpWidth = 320
	inpHeight = 320

	# Load network.
	net = cv.dnn.readNet(model_filepath)

	outNames = []
	outNames.append('feature_fusion/Conv_7/Sigmoid')
	outNames.append('feature_fusion/concat_3')

	img = cv.imread(image_filepath, cv.IMREAD_COLOR)
	if img is None:
		print('Failed to load an image, {}.'.format(image_filepath))
		return

	# Get image height and width.
	height_ = img.shape[0]
	width_ = img.shape[1]
	rW = width_ / float(inpWidth)
	rH = height_ / float(inpHeight)

	# Create a 4D blob from image.
	blob = cv.dnn.blobFromImage(img, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

	# Run the model.
	net.setInput(blob)
	outs = net.forward(outNames)
	#t, _ = net.getPerfProfile()
	#label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

	# Get scores and geometry.
	scores = outs[0]
	geometry = outs[1]
	[boxes, confidences] = decode(scores, geometry, confThreshold)

	# Apply NMS.
	texts = list()
	indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
	for i in indices:
		# Get 4 corners of the rotated rect.
		vertices = cv.boxPoints(boxes[i[0]])
		# Scale the bounding box coordinates based on the respective ratios.
		#for j in range(4):
		#	vertices[j][0] *= rW
		#	vertices[j][1] *= rH
		#for j in range(4):
		#	p1 = (vertices[j][0], vertices[j][1])
		#	p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
		#	cv.line(img, p1, p2, (0, 255, 0), 1);
		pts = list()
		for j in range(4):
			pts.append((vertices[j][0] * rW, vertices[j][1] * rH))
		texts.append(('text', np.array(pts, dtype=np.int32), 'polygon'))

	return texts

# REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/machine_vision/opencv/opencv_text.py
def detect_objects_by_textboxes(image_filepath):
	# REF [site] >> https://github.com/MhLiao/TextBoxes

	# REF [file] >> ${OPENCV_CONTRIB_HOME}/modules/text/samples/textbox.prototxt
	# REF [file] >> ${TextBoxes_HOME}/examples/TextBoxes/deploy.prototxt
	textbox_prototxt_filepath = './pretrained_model/TextBox.prototxt'
	# REF [site] >> https://www.dropbox.com/s/g8pjzv2de9gty8g/TextBoxes_icdar13.caffemodel?dl=0
	textbox_caffemodel_filepath = './pretrained_model/TextBoxes_icdar13.caffemodel'

	threshold = 0.6

	textSpotter = cv.text.TextDetectorCNN_create(textbox_prototxt_filepath, textbox_caffemodel_filepath)

	img = cv.imread(image_filepath, cv.IMREAD_COLOR)
	if img is None:
		print('Failed to load an image, {}.'.format(image_filepath))
		return

	rects, outProbs = textSpotter.detect(img)

	texts = list()
	vis = img.copy()
	for r in range(np.shape(rects)[0]):
		if outProbs[r] > threshold:
			rect = rects[r]
			#cv.rectangle(vis, , (255, 0, 0), 2)
			texts.append(('text', np.array([(rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3])]), 'rectangle'))

	return texts
