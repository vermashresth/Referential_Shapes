import torch
import os
import sys
from datetime import datetime
import time
import pickle
import numpy as np

from feature_rnn import FeatureRNN
from utils import EarlyStopping, AverageMeter
from data_loader import load_messages_data, Property

use_gpu=torch.cuda.is_available()

EPOCHS = 1000 if use_gpu else 2
BATCH_SIZE = 128

def train(model, data, property, current_model_dir):
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Check lr and adam
	es = EarlyStopping(mode="min", patience=30, threshold=0.005, threshold_mode="rel") # Check threshold

	model_prop = str(property).split('.')[-1].lower()

	loss_meters = []

	model.train()

	print('Training model {}'.format(property))
	start_time = time.time()

	for e in range(EPOCHS):
		loss_meter = AverageMeter()

		for d in data:
			message, metadata = d

			if property == Property.COLOR:
				one_hot_prop = metadata[:,0:3]
			elif property == Property.SHAPE:
				one_hot_prop = metadata[:,3:6]
			elif property == Property.SIZE:
				one_hot_prop = metadata[:,6:8]
			elif property == Property.ROW:
				one_hot_prop = metadata[:,9:12]
			elif property == Property.COLUMN:
				one_hot_prop = metadata[:,12:15]

			loss = model(message, one_hot_prop)

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

			torch.save(model.state_dict(), '{}/{}_{}_model'.format(current_model_dir, model_prop, e))

		loss_meters.append(loss_meter)
		es.step(loss_meter.avg)

		if es.is_converged:
			print("Converged in epoch {}".format(e))
			break

	print('Training took {} seconds'.format(time.time() - start_time))

	pickle.dump(loss_meters, open('{}/{}_{}_loss_meters.p'.format(current_model_dir, model_prop, e), 'wb'))
	return loss_meters
		

def evaluate(model, data, property, current_model_dir, best_epoch):
	model_prop = str(property).split('.')[-1].lower()

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()

	model.eval()

	for d in data:
		message, metadata = d

		if property == Property.COLOR:
			one_hot_prop = metadata[:,0:3]
		elif property == Property.SHAPE:
			one_hot_prop = metadata[:,3:6]
		elif property == Property.SIZE:
			one_hot_prop = metadata[:,6:8]
		elif property == Property.ROW:
			one_hot_prop = metadata[:,9:12]
		elif property == Property.COLUMN:
			one_hot_prop = metadata[:,12:15]

		loss, acc = model(message, one_hot_prop)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())

	print('Test loss {}'.format(loss_meter.avg))
	print('Test acc {}'.format(acc_meter.avg))
	print()
	pickle.dump(loss_meter, open('{}/{}_{}_test_loss_meter.p'.format(current_model_dir, model_prop, best_epoch), 'wb'))
	pickle.dump(acc_meter, open('{}/{}_{}_test_acc_meter.p'.format(current_model_dir, model_prop, best_epoch), 'wb'))

	return acc_meter.avg


data_folder_id = sys.argv[1]
shapes_dataset = sys.argv[2]

print('===========================')
print('Using messages from model: {}'.format(data_folder_id))
print('Shapes: {}'.format(shapes_dataset))
print('===========================')

comma_separated_str = data_folder_id + ',' + shapes_dataset + ','

dumps_dir = './rnndumps'
if not os.path.exists(dumps_dir):
	os.mkdir(dumps_dir)

model_id = '{:%m%d%H%M%S%f}'.format(datetime.now())

print('Model id: {}'.format(model_id))

current_model_dir = '{}/{}'.format(dumps_dir, model_id)

os.mkdir(current_model_dir)

color_rnn = FeatureRNN(3)
shape_rnn = FeatureRNN(3)
size_rnn = FeatureRNN(2)
row_rnn = FeatureRNN(3)
column_rnn = FeatureRNN(3)

if use_gpu:
	color_rnn = color_rnn.cuda()
	shape_rnn = shape_rnn.cuda()
	size_rnn = size_rnn.cuda()
	row_rnn = row_rnn.cuda()
	column_rnn = column_rnn.cuda()

# Load training data
train_data, _val_data, test_data = load_messages_data(data_folder_id, shapes_dataset, BATCH_SIZE)

for rnn, prop in [(color_rnn, Property.COLOR),
				   (shape_rnn, Property.SHAPE),
				   (size_rnn, Property.SIZE),
				   (row_rnn, Property.ROW),
				   (column_rnn, Property.COLUMN)]:

	loss_meters = train(rnn, train_data, prop, current_model_dir)
	best_epoch = np.argmin([m.avg for m in loss_meters])
	best_model = FeatureRNN(2 if prop == Property.SIZE else 3)
	best_model_name = '{}/{}_{}_model'.format(current_model_dir, str(prop).split('.')[-1].lower(), best_epoch)
	print('Loading best model for evaluation: {}'.format(best_model_name))
	state = torch.load(best_model_name, map_location= lambda storage, location: storage)
	best_model.load_state_dict(state)

	if use_gpu:
		best_model = best_model.cuda()

	acc = evaluate(best_model, test_data, prop, current_model_dir, best_epoch)

	comma_separated_str += str(acc) + ','

print()
print(comma_separated_str[0:-1])