import os

# REF [function] >> recognize_text_using_aihub_data() in ${SWL_PYTHON_HOME}/test/language_processing/run_text_recognition.py
def recognize_text_by_transformer(image_filepath, rects):
	recognized_texts = list()
	for idx, rct in enumerate(rects):
		recognized_texts.append('text_{}'.format(idx))
	return recognized_texts