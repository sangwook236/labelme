import os, urllib.request, time
import torch
import labelme.swl.craft.imgproc as imgproc
#import labelme.swl.craft.file_utils as file_utils
import labelme.swl.craft.test_utils as test_utils

# REF [function] >> run_word_craft() in ${SWL_PYTHON_HOME}/test/language_processing/craft/test_utils.py
def detect_text_by_craft(image_filepath):
	cache_dir_path = os.path.expanduser(os.path.join('~', '.labelme'))
	#craft_trained_model_filepath = os.path.join(cache_dir_path, 'craft_mlt_25k.pth')
	craft_trained_model_filepath = './labelme/swl/craft/craft_mlt_25k.pth'
	#craft_trained_model_filepath = os.path.join(cache_dir_path, 'craft_refiner_CTW1500.pth')
	craft_refiner_model_filepath = './labelme/swl/craft/craft_refiner_CTW1500.pth'  # Pretrained refiner model.
	if not os.path.exists(craft_trained_model_filepath):
		url = 'https://www.mediafire.com/file/qh2ullnnywi320s/craft_mlt_25k.pth/file'
		os.makedirs(cache_dir_path, exist_ok=True)
		urllib.request.urlretrieve(url, craft_trained_model_filepath)
	if not os.path.exists(craft_refiner_model_filepath):
		url = 'https://www.mediafire.com/file/qh2ullnnywi320s/craft_refiner_CTW1500.pth/file'
		os.makedirs(cache_dir_path, exist_ok=True)
		urllib.request.urlretrieve(url, craft_refiner_model_filepath)
	craft_refine = False  # Enable link refiner.
	craft_cuda = torch.cuda.is_available()  # Use cuda for inference.

	print('Start loading CRAFT...')
	start_time = time.time()
	craft_net, craft_refine_net = test_utils.load_craft(craft_trained_model_filepath, craft_refiner_model_filepath, craft_refine, craft_cuda)
	print('End loading CRAFT: {} secs.'.format(time.time() - start_time))

	print('Start running CRAFT...')
	start_time = time.time()
	rgb = imgproc.loadImage(image_filepath)  # RGB order.
	bboxes, polys, score_text = test_utils.run_word_craft(rgb, craft_net, craft_refine_net, craft_cuda)
	print('End running CRAFT: {} secs.'.format(time.time() - start_time))

	# bboxes = N x 4 x 2. np.float32.
	return list(('text', bbox, 'polygon') for bbox in bboxes)
