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



folder = 'dummy'#'different_targets'
npy_file = '{}/train.large.input.npy'.format(folder)
metadata_file = '{}/train.large.metadata.p'.format(folder)

assert len(sys.argv) > 1, 'Indicate an index!'

index = int(sys.argv[1])

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
