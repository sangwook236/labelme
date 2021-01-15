import keras_ocr

# REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/language_processing/keras_ocr_test.py
def detect_text_by_craft(image_filepath):
	detector = keras_ocr.detection.Detector()

	image = keras_ocr.tools.read(image_filepath)

	# Boxes will be an Nx4x2 array of box quadrangles, where N is the number of detected text boxes.
	bboxes = detector.detect(images=[image])[0]
	#canvas = keras_ocr.detection.drawBoxes(image, bboxes)

	#plt.imshow(canvas)
	#plt.show()

	texts = list()
	for bbox in bboxes:
		texts.append(('text', bbox, 'polygon'))

	return texts
