import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms
from PIL import Image
import torch.utils.data as data
from torch.utils.data import DataLoader
import os


use_gpu = torch.cuda.is_available()


class ShapesDataset(data.Dataset):

    def __init__(self, images):
        super().__init__()

        self.data = images

        self.transforms = torchvision.transforms.Compose([
	        torchvision.transforms.ToPILImage(),
	        torchvision.transforms.Resize((250, 250), Image.LINEAR),
	        torchvision.transforms.ToTensor(),
	        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Needed for pretrained models
	    ])

    def __getitem__(self, index):
        image = self.data[index, :, :, :]
        
        image = self.transforms(image)
        
        return image        

    def __len__(self):
        return self.data.shape[0]



def get_features(dataloader, output_data_folder, file_id):
	# n_features = 4096

	# features = np.zeros((len(dataloader), n_features))

	# if use_gpu:
	# 	features = features.cuda()

	for i, x in enumerate(dataloader):
		if use_gpu:
			x = x.cuda()

		y = vgg16.features(x)
		# y = vgg16.avgpool(y)
		y = y.view(y.size(0), -1)
		y = vgg16.classifier[:5](y)

		y = y.detach()

		if use_gpu:
			y = y.cpu()


		np.save('{}/{}_{}_features.npy'.format(output_data_folder, file_id, i), y.numpy())

		if not use_gpu and i == 5:
			break

		# print(y)
		# print(y.shape)
		

		# features[i] = y.numpy()
	
	# return features


def stitch_files(output_data_folder, file_id):
	file_names = ['{}/{}'.format(output_data_folder, f) for f in os.listdir(output_data_folder) if file_id in f]
	file_names.sort(key=os.path.getctime)

	for i, f in enumerate(file_names):
		arr = np.load(f)
		if i == 0:
			features = arr
		else:
			features = np.concatenate((features, arr))

	np.save('{}/{}_features.npy'.format(output_data_folder, file_id), features)



batch_size = 128 if use_gpu else 2

folder = 'balanced'
train_images = np.load('{}/train.large.input.npy'.format(folder))
val_images = np.load('{}/val.input.npy'.format(folder))
test_images = np.load('{}/test.input.npy'.format(folder))

train_dataset = ShapesDataset(train_images)
val_dataset = ShapesDataset(val_images)
test_dataset = ShapesDataset(test_images)

train_dataloader = DataLoader(train_dataset, num_workers=8, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, num_workers=8, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, num_workers=8, batch_size=batch_size)

vgg16 = models.vgg16(pretrained=True)
if use_gpu:
	vgg16 = vgg16.cuda()

# print(vgg16)
# print(vgg16.classifier[:5])

vgg16.eval()
for name,param in vgg16.named_parameters():
	if param.requires_grad:
		param.requires_grad = False

output_data_folder = '../data/shapes/{}'.format(folder)

if not os.path.exists(output_data_folder):
	os.mkdir(output_data_folder)

train_features = get_features(train_dataloader, output_data_folder, 'train')
valid_features = get_features(val_dataloader, output_data_folder, 'valid')
test_features = get_features(test_dataloader, output_data_folder, 'test')


# Stitch into one file
stitch_files(output_data_folder, 'train')
stitch_files(output_data_folder, 'valid')
stitch_files(output_data_folder, 'test')