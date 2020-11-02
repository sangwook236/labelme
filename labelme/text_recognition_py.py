import os, math, time
import numpy as np
import torch, torchvision
import cv2, PIL.Image
import labelme.dia_engine.dia_facade as dia_facade
import labelme.dia_engine.text_generation_util as tg_util

def recognize_text_by_transformer(patches):
	input_shape = 64, 1280, 3
	max_label_len = 50
	batch_size = 64
	gpu = -1

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
	if is_pil:
		patches = list(PIL.Image.fromarray(patch) for patch in patches)
	inputs = dia_facade.images_to_tensor(patches, input_shape, is_pil, logger)

	#--------------------
	# Build a model.
	model = dia_facade.build_text_model_for_inference(model_filepath_to_load, input_shape, max_label_len, label_converter, SOS_ID, EOS_ID, num_suffixes, logger=logger, device=device)

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
