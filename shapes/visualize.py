import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

def display_image(img):
	plt.clf()

	img = np.transpose(img, (2, 0, 1))
	b, g, r = img[0,...], img[1,...], img[2,...]
	# print(b.shape)
	# print(g.shape)
	# print(r.shape)
	rgb = np.asarray([r, g, b])
	# print(rgb.shape)
	transp = np.transpose(rgb, (1, 2, 0))


	plt.imshow(transp)
	plt.show()


assert len(sys.argv) > 1, 'Folder name, train/val/test, index'

folder = sys.argv[1]
set_name = sys.argv[2]
if set_name == 'train':
	str_set_name = 'train.large'
else:
	str_set_name = set_name

npy_file = '{}/{}.input.npy'.format(folder, str_set_name)
metadata_file = '{}/{}.metadata.p'.format(folder, str_set_name)
index = int(sys.argv[3])

img = np.load(npy_file)[index]
print(img.shape)

metadata = pickle.load(open(metadata_file, 'rb'))[index]

if len(img.shape) == 4: # We have tuples because this is 4-D
	n_images = img.shape[0]
	for i in range(n_images):
		display_image(img[i])
		print(metadata[i])
else:
	display_image(img)
	print(metadata)
