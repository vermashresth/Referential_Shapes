import torch
import os
import sys
from datetime import datetime
import time
import pickle
import numpy as np
from torchvision.utils import save_image

from rnn_decoder import Autoencoder
from utils import EarlyStopping, AverageMeter, to_img
from data_loader import load_message_image_data

use_gpu=torch.cuda.is_available()

EPOCHS = 1000 if use_gpu else 2
BATCH_SIZE = 128

def train(model, data, current_model_dir):
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5) # Check lr and adam
	es = EarlyStopping(mode="min", patience=30, threshold=0.005, threshold_mode="rel") # Check threshold

	loss_meters = []

	model.train()

	dc_folder = '{}/dc_img'.format(current_model_dir)
	if not os.path.exists(dc_folder):
		os.mkdir(dc_folder)

	start_time = time.time()

	for e in range(EPOCHS):
		loss_meter = AverageMeter()

		for d in data:
			message, img = d

			output, loss = model(message, img)

			loss.backward()
			optimizer.step()

			loss_meter.update(loss.item())
		
		print('Epoch {}, loss {}'.format(e, loss_meter.avg))

		# Save only the best model
		if e == 0 or loss_meter.avg < np.min([m.avg for m in loss_meters]):
			# First delete the old model
			if e > 0:
				old_model_files = ['{}/{}'.format(current_model_dir, f) for f in os.listdir(current_model_dir) if f.endswith('_model')]
				if len(old_model_files) > 0:
					os.remove(old_model_files[0])

			torch.save(model.state_dict(), '{}/{}_model'.format(current_model_dir, e))

		loss_meters.append(loss_meter)
		es.step(loss_meter.avg)

		if e % 10 == 0:			
			pic = to_img(output.cpu().data if use_gpu else output.data)
			save_image(pic, '{}/image_{}.png'.format(dc_folder, e))

		if es.is_converged:
			print("Converged in epoch {}".format(e))
			break

	print('Training took {} seconds'.format(time.time() - start_time))

	pickle.dump(loss_meters, open('{}/{}_loss_meters.p'.format(current_model_dir, e), 'wb'))
	return loss_meters
		

def evaluate(model, data, current_model_dir, best_epoch):
	loss_meter = AverageMeter()
	acc_meter = AverageMeter()

	model.eval()

	dc_folder = '{}/dc_img_test'.format(current_model_dir)
	if not os.path.exists(dc_folder):
		os.mkdir(dc_folder)

	for i, d in enumerate(data):
		message, img = d

		output, loss = model(message, img)

		loss_meter.update(loss.item())
		# acc_meter.update(acc.item())

		if i % 10 == 0:
			pic = to_img(output.cpu().data if use_gpu else output.data)
			save_image(pic, '{}/image_{}.png'.format(dc_folder, i))

	print('Test loss {}'.format(loss_meter.avg))
	# print('Test acc {}'.format(acc_meter.avg))
	pickle.dump(loss_meter, open('{}/{}_test_loss_meter.p'.format(current_model_dir, best_epoch), 'wb'))
	# pickle.dump(acc_meter, open('{}/{}_test_acc_meter.p'.format(current_model_dir, best_epoch), 'wb'))




data_folder_id = sys.argv[1]
shapes_dataset = sys.argv[2]

print('===========================')
print('Using messages from model: {}'.format(data_folder_id))
print('Shapes: {}'.format(shapes_dataset))
print('===========================')


dumps_dir = './rnndecoderdumps'
if not os.path.exists(dumps_dir):
	os.mkdir(dumps_dir)

model_id = '{:%m%d%H%M%S%f}'.format(datetime.now())

print('Model id: {}'.format(model_id))

current_model_dir = '{}/{}'.format(dumps_dir, model_id)

os.mkdir(current_model_dir)

model = Autoencoder()
if use_gpu:
	model = model.cuda()

# Load training data
train_data, _val_data, test_data = load_message_image_data(data_folder_id, shapes_dataset, BATCH_SIZE)

loss_meters = train(model, train_data, current_model_dir)

best_epoch = np.argmin([m.avg for m in loss_meters])
best_model = Autoencoder()
best_model_name = '{}/{}_model'.format(current_model_dir, best_epoch)
print('Loading best model for evaluation: {}'.format(best_model_name))
state = torch.load(best_model_name, map_location= lambda storage, location: storage)
best_model.load_state_dict(state)

if use_gpu:
	best_model = best_model.cuda()

evaluate(best_model, test_data, current_model_dir, best_epoch)