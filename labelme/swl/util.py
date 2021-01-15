def download(url, output_dir_path):
	import urllib.request
	response = urllib.request.urlopen(url)

	file_names = url.split('/')[-1].split('.')
	filename, fileext = file_names[0], file_names[-1]
	#if 'zip' == fileext or fileext.find('zip') != -1:
	if 'zip' == fileext:
		import zipfile
		response = zipfile.ZipFile(response)
		content = zipfile.ZipFile.open(response).read()

		import os
		output_dir_path = os.path.join(output_dir_path, filename)
		with open(output_dir_path, 'w') as fd:
			fd.write(content.read())
	elif 'gz' == fileext or fileext.find('gz') != -1:
	#elif 'gz' == fileext:
		import tarfile
		tar = tarfile.open(mode='r:gz', fileobj=response)
		tar.extractall(output_dir_path)
	#elif 'bz2' == fileext or fileext.find('bz2') != -1 or 'bzip2' == fileext or fileext.find('bzip2') != -1:
	elif 'bz2' == fileext or 'bzip2' == fileext:
		import tarfile
		tar = tarfile.open(mode='r:bz2', fileobj=response)
		tar.extractall(output_dir_path)
	#elif 'xz' == fileext or fileext.find('xz') != -1:
	elif 'xz' == fileext:
		import tarfile
		tar = tarfile.open(mode='r:xz', fileobj=response)
		tar.extractall(output_dir_path)
	else:
		raise ValueError('Unexpected file extention, {}'.format(fileext))
		return False

	return True
