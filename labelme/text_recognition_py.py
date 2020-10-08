import os, math, time
import numpy as np
import torch, torchvision
import cv2, PIL.Image
import labelme.dia_engine.dia_facade as dia_facade
import labelme.dia_engine.text_generation_util as tg_util

def recognize_text_by_transformer(image_filepath, rects):
	image_shape = 64, 1280, 3
	max_label_len = 50
	batch_size = 64
	gpu = 0

	model_filepath_to_load = './labelme/dia_model/dia_20201002.pth'
	is_pil = True
	logger = None

	#if gpu >= 0:
	#	os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
	device = torch.device(('cuda:{}'.format(gpu) if gpu >= 0 else 'cuda') if torch.cuda.is_available() else 'cpu')
	print('Device: {}.'.format(device))

	#--------------------
	charset = tg_util.construct_charset(hangeul=True)

	# Create a label converter.
	label_converter_type = 'sos+eos'
	label_converter, SOS_ID, EOS_ID, BLANK_LABEL, num_suffixes = dia_facade.create_label_converter(label_converter_type, charset)
	print('#classes = {}.'.format(label_converter.num_tokens))
	print('<PAD> = {}, <SOS> = {}, <EOS> = {}, <UNK> = {}.'.format(label_converter.pad_id, SOS_ID, EOS_ID, label_converter.encode([label_converter.UNKNOWN], is_bare_output=True)[0]))

	#--------------------
	print('Start loading image patches...')
	start_time = time.time()
	image = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
	if image is None:
		print('File not found, {}.'.format(image_filepath))
		return None
	patches = list()
	for rct in rects:
		patch = image[math.floor(rct[1]):math.ceil(rct[3])+1, math.floor(rct[0]):math.ceil(rct[2])+1]
		patches.append(PIL.Image.fromarray(patch) if is_pil else patch)
	inputs = dia_facade.images_to_tensor(patches, image_shape, is_pil, logger)
	print('End loading image patches: {} secs.'.format(time.time() - start_time))

	#--------------------
	# Build a model.
	model = dia_facade.build_text_model_for_inference(model_filepath_to_load, image_shape, max_label_len, label_converter, SOS_ID, EOS_ID, num_suffixes, logger=logger, device=device)

	if model and label_converter:
		#batch_size = 1  # Infer one-by-one.

		# Infer by the model.
		print('Start inferring...')
		start_time = time.time()
		model.eval()
		predictions = dia_facade.recognize_text(model, inputs, batch_size, logger=logger, device=device)
		print('End inferring: {} secs.'.format(time.time() - start_time))
		print('Inference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(predictions.shape, predictions.dtype, np.min(predictions), np.max(predictions)))

		recognized_texts = list(label_converter.decode(pred) for pred in predictions)
		return recognized_texts
	else: return None
