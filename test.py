import pickle
import numpy as np
import random
from datetime import datetime
import os
import sys
import time
import math

import torch
from model import Model
from run import train_one_epoch, evaluate
from utils import EarlyStopping
from dataloader import load_dictionaries, load_data
from build_shapes_dictionaries import *
from decode import dump_words


use_gpu = torch.cuda.is_available()
debugging = not use_gpu
should_dump = not debugging
should_covert_to_words = not debugging
# should_print_images_metadata = True

seed = 42
torch.manual_seed(seed)
if use_gpu:
	torch.cuda.manual_seed(seed)

prev_model_file_name = None#'dumps/01_26_00_16/01_26_00_16_915_model'

EPOCHS = 1000 if not debugging else 2
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 128 if not debugging else 2
MAX_SENTENCE_LENGTH = 13 if not debugging else 5
K = 3  # number of distractors

vocab_size = 10
shapes_dataset = 'balanced'
vl_loss_weight = 0.0
bound_weight = 1.0

if len(sys.argv) > 1:
	vocab_size = int(sys.argv[1])
	MAX_SENTENCE_LENGTH = int(sys.argv[2])
	shapes_dataset = sys.argv[3]
	vl_loss_weight = float(sys.argv[4])
	bound_weight = float(sys.argv[5])

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
print('========================================')
print('Model id: {}'.format(model_id))
print('|V|: {}'.format(vocab_size))
print('L: {}'.format(MAX_SENTENCE_LENGTH))
print('Using gpu: {}'.format(use_gpu))
print('Dataset: {}'.format(shapes_dataset))
print('Lambda: {}'.format(vl_loss_weight))
print('Alpha: {}'.format(bound_weight))
#################################################

current_model_dir = '{}/{}_{}_{}'.format(dumps_dir, model_id, vocab_size, MAX_SENTENCE_LENGTH)

if should_dump and not os.path.exists(current_model_dir):
	os.mkdir(current_model_dir)


model = Model(n_image_features, vocab_size,
	EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, 
	bound_idx, MAX_SENTENCE_LENGTH, vl_loss_weight, bound_weight, use_gpu)


if prev_model_file_name is not None:
	state = torch.load(prev_model_file_name, map_location= lambda storage, location: storage)
	model.load_state_dict(state)


if use_gpu:
	model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
es = EarlyStopping(mode="max", patience=30, threshold=0.005, threshold_mode="rel")

if prev_model_file_name == None:
	losses_meters = []
	eval_losses_meters = []

	accuracy_meters = []
	eval_accuracy_meters = []

	entropy_meters = []
	eval_entropy_meters = []
else:
	losses_meters = pickle.load(open('{}/{}_{}_losses_meters.p'.format(current_model_dir, model_id, starting_epoch), 'rb'))
	eval_losses_meters = pickle.load(open('{}/{}_{}_eval_losses_meters.p'.format(current_model_dir, model_id, starting_epoch), 'rb'))

	accuracy_meters = pickle.load(open('{}/{}_{}_accuracy_meters.p'.format(current_model_dir, model_id, starting_epoch), 'rb'))
	eval_accuracy_meters = pickle.load(open('{}/{}_{}_eval_accuracy_meters.p'.format(current_model_dir, model_id, starting_epoch), 'rb'))


word_counts = torch.zeros([vocab_size])
if use_gpu:
	word_counts = word_counts.cuda()

eval_word_counts = torch.zeros([vocab_size])
if use_gpu:
	eval_word_counts = eval_word_counts.cuda()

is_loss_nan = False
should_evaluate_best = True

# Train
for epoch in range(EPOCHS):
	epoch_start_time = time.time()

	e = epoch + starting_epoch

	epoch_loss_meter, epoch_acc_meter, messages, epoch_w_counts, epoch_entropy_meter = train_one_epoch(
		model, train_data, optimizer, word_counts, debugging)

	if math.isnan(epoch_loss_meter.avg):
		print("The train loss in NaN. Stop training")
		is_loss_nan = True
		break

	losses_meters.append(epoch_loss_meter)
	accuracy_meters.append(epoch_acc_meter)
	entropy_meters.append(epoch_entropy_meter)
	word_counts += epoch_w_counts

	eval_loss_meter, eval_acc_meter, eval_messages, _w_counts, eval_entropy_meter = evaluate(
		model, valid_data, eval_word_counts, debugging)

	eval_losses_meters.append(eval_loss_meter)
	eval_accuracy_meters.append(eval_acc_meter)
	eval_entropy_meters.append(eval_entropy_meter)

	print('Epoch {}, average train loss: {}, average val loss: {}, average accuracy: {}, average val accuracy: {}'.format(
		e, losses_meters[e].avg, eval_losses_meters[e].avg, accuracy_meters[e].avg, eval_accuracy_meters[e].avg))

	print('--(Took {} seconds)'.format(time.time() - epoch_start_time))

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

if is_loss_nan:
	should_dump = False
	should_evaluate_best = False

if should_dump:
	# Dump latest stats
	pickle.dump(losses_meters, open('{}/{}_{}_losses_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	pickle.dump(eval_losses_meters, open('{}/{}_{}_eval_losses_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	pickle.dump(accuracy_meters, open('{}/{}_{}_accuracy_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	pickle.dump(eval_accuracy_meters, open('{}/{}_{}_eval_accuracy_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	pickle.dump(entropy_meters, open('{}/{}_{}_entropy_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	pickle.dump(eval_entropy_meters, open('{}/{}_{}_eval_entropy_meters.p'.format(current_model_dir, model_id, e), 'wb'))


# Evaluate best model on test data
if should_evaluate_best:

	if debugging:
		best_model = model
		best_epoch = e
	else:
		best_epoch = np.argmax([m.avg for m in eval_accuracy_meters])
		best_model = Model(n_image_features, vocab_size,
			EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, 
			bound_idx, MAX_SENTENCE_LENGTH, vl_loss_weight, bound_weight, use_gpu)
		best_model_name = '{}/{}_{}_model'.format(current_model_dir, model_id, best_epoch)
		state = torch.load(best_model_name, map_location= lambda storage, location: storage)
		best_model.load_state_dict(state)

	if use_gpu:
		best_model = best_model.cuda()

	test_word_counts = torch.zeros([vocab_size])
	if use_gpu:
		test_word_counts = test_word_counts.cuda()

	_, test_acc_meter, test_messages, _w_counts = evaluate(best_model, test_data, test_word_counts, debugging)

	print('Test accuracy: {}'.format(test_acc_meter.avg))

	if should_dump:
		pickle.dump(test_acc_meter, open('{}/{}_{}_test_accuracy_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_messages, open('{}/{}_{}_test_messages.p'.format(current_model_dir, model_id, best_epoch), 'wb'))

		if should_covert_to_words:
			dump_words(current_model_dir, test_messages, idx_to_word, '{}_{}_test_messages'.format(model_id, best_epoch))