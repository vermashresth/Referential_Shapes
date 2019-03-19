import pickle
import numpy as np
import random
from datetime import datetime
import os
import sys

import torch
from model import Model
from run import train_one_epoch, evaluate
from utils import EarlyStopping #get_lr_scheduler
from dataloader import load_dictionaries, load_data
from build_shapes_dictionaries import *
from decode import dump_words


use_gpu = torch.cuda.is_available()
debugging = not use_gpu
should_dump = not debugging
should_covert_to_words = not debugging

seed = 42
torch.manual_seed(seed)
if use_gpu:
	torch.cuda.manual_seed(seed)

prev_model_file_name = None#'dumps/01_26_00_16/01_26_00_16_915_model'

EPOCHS = 1000 if not debugging else 1
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 128 if not debugging else 4
MAX_SENTENCE_LENGTH = 13 if not debugging else 5
K = 3  # number of distractors

vocab_size = 10
shapes_dataset = 'balanced'

if len(sys.argv) > 1:
	vocab_size = int(sys.argv[1])
	MAX_SENTENCE_LENGTH = int(sys.argv[2])
	shapes_dataset = sys.argv[3]


# Create vocab if there is not one for the desired size already
if not does_vocab_exist(vocab_size):
	build_vocab(vocab_size)

# Load vocab
word_to_idx, idx_to_word, bound_idx = load_dictionaries('shapes', vocab_size)
#vocab_size = len(word_to_idx) # mscoco: 10000

# Load data
n_image_features, train_data, valid_data, test_data = load_data('shapes/{}'.format(shapes_dataset), BATCH_SIZE, K)

# Settings
dumps_dir = './dumps'
if should_dump and not os.path.exists(dumps_dir):
	os.mkdir(dumps_dir)

if prev_model_file_name == None:
	model_id = '{:%m%d%H%M%S%f}'.format(datetime.now())
	starting_epoch = 0
else:
	last_backslash = prev_model_file_name.rfind('/')
	last_underscore = prev_model_file_name.rfind('_')
	second_last_underscore = prev_model_file_name[:last_underscore].rfind('_')
	model_id = prev_model_file_name[last_backslash+1:second_last_underscore]
	starting_epoch = int(prev_model_file_name[second_last_underscore+1:last_underscore])


################# Print info ####################
print('----------------------------------------')
print('Model id: {}'.format(model_id))
print('|V|: {}'.format(vocab_size))
print('L: {}'.format(MAX_SENTENCE_LENGTH))
print('Using gpu: {}'.format(use_gpu))
print('Dataset: {}'.format(shapes_dataset))
#################################################

current_model_dir = '{}/{}_{}_{}'.format(dumps_dir, model_id, vocab_size, MAX_SENTENCE_LENGTH)

if should_dump and not os.path.exists(current_model_dir):
	os.mkdir(current_model_dir)


model = Model(n_image_features, vocab_size,
	EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, 
	bound_idx, MAX_SENTENCE_LENGTH, use_gpu)


if prev_model_file_name is not None:
	state = torch.load(prev_model_file_name, map_location= lambda storage, location: storage)
	model.load_state_dict(state)


if use_gpu:
	model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# lr_scheduler = get_lr_scheduler(optimizer)
es = EarlyStopping(mode="max", patience=30, threshold=0.005, threshold_mode="rel")

# Train
if prev_model_file_name == None:
	losses_meters = []
	eval_losses_meters = []

	accuracy_meters = []
	eval_accuracy_meters = []
else:
	losses_meters = pickle.load(open('{}/{}_{}_losses_meters.p'.format(current_model_dir, model_id, starting_epoch), 'rb'))
	eval_losses_meters = pickle.load(open('{}/{}_{}_eval_losses_meters.p'.format(current_model_dir, model_id, starting_epoch), 'rb'))

	accuracy_meters = pickle.load(open('{}/{}_{}_accuracy_meters.p'.format(current_model_dir, model_id, starting_epoch), 'rb'))
	eval_accuracy_meters = pickle.load(open('{}/{}_{}_eval_accuracy_meters.p'.format(current_model_dir, model_id, starting_epoch), 'rb'))


for epoch in range(EPOCHS):
	e = epoch + starting_epoch

	epoch_loss_meter, epoch_acc_meter = train_one_epoch(
		model, train_data, optimizer, debugging)

	losses_meters.append(epoch_loss_meter)
	accuracy_meters.append(epoch_acc_meter)

	eval_loss_meter, eval_acc_meter, eval_messages = evaluate(
		model, valid_data, debugging)

	eval_losses_meters.append(eval_loss_meter)
	eval_accuracy_meters.append(eval_acc_meter)

	# Skip for now
	print('Epoch {}, average train loss: {}, average val loss: {}, average accuracy: {}, average val accuracy: {}'.format(
		e, losses_meters[e].avg, eval_losses_meters[e].avg, accuracy_meters[e].avg, eval_accuracy_meters[e].avg))

	# lr_scheduler.step(eval_acc_meter.avg)
	es.step(eval_acc_meter.avg)

	if should_dump:
		# Dump models
		if epoch == 0 or eval_acc_meter.avg > np.max([v.avg for v in eval_accuracy_meters[:-1]]):
			if epoch > 0:
				# First delete old model file
				old_model_files = ['{}/{}'.format(current_model_dir, f) for f in os.listdir(current_model_dir) if f.endswith('_model')]
				if len(old_model_files) > 0:
					os.remove(old_model_files[0])

			torch.save(model.state_dict(), '{}/{}_{}_model'.format(current_model_dir, model_id, e))

		# Dump messages
		#pickle.dump(messages, open('{}/{}_{}_messages.p'.format(current_model_dir, model_id, e), 'wb')) # Cannot do this wth ST-GS
		# Skip for now
		#pickle.dump(eval_messages, open('{}/{}_{}_eval_messages.p'.format(current_model_dir, model_id, e), 'wb'))

	if es.is_converged:
		print("Converged in epoch {}".format(e))
		break


if should_dump:
	# Dump latest stats
	pickle.dump(losses_meters, open('{}/{}_{}_losses_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	pickle.dump(eval_losses_meters, open('{}/{}_{}_eval_losses_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	pickle.dump(accuracy_meters, open('{}/{}_{}_accuracy_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	pickle.dump(eval_accuracy_meters, open('{}/{}_{}_eval_accuracy_meters.p'.format(current_model_dir, model_id, e), 'wb'))



# Evaluate best model on test data

if debugging:
	best_model = model
	best_epoch = e
else:
	best_epoch = np.argmax([m.avg for m in eval_accuracy_meters])
	best_model = Model(n_image_features, vocab_size,
		EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, 
		bound_idx, MAX_SENTENCE_LENGTH, use_gpu)
	best_model_name = '{}/{}_{}_model'.format(current_model_dir, model_id, best_epoch)
	state = torch.load(best_model_name, map_location= lambda storage, location: storage)
	best_model.load_state_dict(state)

if use_gpu:
	best_model = best_model.cuda()

_, test_acc_meter, test_messages = evaluate(best_model, test_data, debugging)

print('Test accuracy: {}'.format(test_acc_meter.avg))

if should_dump:
	pickle.dump(test_acc_meter, open('{}/{}_{}_test_accuracy_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
	pickle.dump(test_messages, open('{}/{}_{}_test_messages.p'.format(current_model_dir, model_id, best_epoch), 'wb'))

	if should_covert_to_words:
		dump_words(current_model_dir, test_messages, idx_to_word, '{}_{}_test_messages'.format(model_id, best_epoch))