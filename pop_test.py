import pickle
import numpy as np
import random
from datetime import datetime
import os
import sys
import time
import math

import torch
from model import ModelPop
from run import train_one_epoch, evaluate
from utils import EarlyStopping
from dataloader import load_dictionaries, load_images, load_pretrained_features
from build_shapes_dictionaries import *
from metadata import does_shapes_onehot_metadata_exist, create_shapes_onehot_metadata, load_shapes_onehot_metadata
from decode import dump_words
from visual_module import CNN
from dump_cnn_features import save_features
import argparse

import wandb

use_gpu = torch.cuda.is_available()
debugging = not use_gpu
should_dump = True#not debugging
should_covert_to_words = True#not debugging
should_dump_indices = True#not debugging


EPOCHS = 15 if not debugging else 3
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 128 if not debugging else 8
K = 3  # number of distractors
n_image_features = 2048#4096

# Default settings
vocab_size = 10
max_sentence_length = 5
dataset_type = 0
vl_loss_weight = 0.0
bound_weight = 1.0
use_random_model = 0
should_train_visual = 1
cnn_model_file_name = None
rsa_sampling = 50
seed = 42
use_symbolic_input = False
noise_strength = 0
use_distractors_in_sender = False

pop_size = 10
use_bullet = False

cmd_parser = argparse.ArgumentParser()
cmd_parser.add_argument('--K', type=int, default=K)
cmd_parser.add_argument('--seed', type=int, default=seed)
cmd_parser.add_argument('--vocab_size', type=int, default=vocab_size)
cmd_parser.add_argument('--max_sentence_length', type=int, default=max_sentence_length)
cmd_parser.add_argument('--vl_loss_weight', type=float, default=vl_loss_weight)
cmd_parser.add_argument('--bound_weight', type=float, default=bound_weight)
cmd_parser.add_argument('--noise_strength', type=int, default=noise_strength)
cmd_parser.add_argument('--dataset_type', type=int, default=dataset_type)
cmd_parser.add_argument('--pop_size', type=int, default=pop_size)
cmd_parser.add_argument('--use_symbolic_input', action='store_true', default=use_symbolic_input)
cmd_parser.add_argument('--use_distractors_in_sender', action='store_true', default=use_distractors_in_sender)
cmd_parser.add_argument('--use_bullet', action='store_true', default=use_bullet)

cmd_parser.add_argument('--use_random_model', type=int, default=use_random_model)
cmd_parser.add_argument('--should_train_visual', type=int, default=should_train_visual)
# cmd_parser.add_argument('--cnn_model_file_name', type=str, default=cnn_model_file_name)

cmd_parser.add_argument('--rsa_sampling', type=int, default=rsa_sampling)

cmd_args = cmd_parser.parse_args()

# Overwrite default settings if given in command line
# if len(sys.argv) > 1:
K = cmd_args.K #int(sys.argv[1])
seed = cmd_args.seed #int(sys.argv[1])
vocab_size = cmd_args.vocab_size #int(sys.argv[2])
max_sentence_length = cmd_args.max_sentence_length #int(sys.argv[3])
dataset_type = cmd_args.dataset_type #sys.argv[4]
vl_loss_weight = cmd_args.vl_loss_weight #float(sys.argv[5])
bound_weight = cmd_args.bound_weight #float(sys.argv[6])
use_symbolic_input = cmd_args.use_symbolic_input
should_train_visual = cmd_args.should_train_visual
use_random_model = cmd_args.use_random_model
rsa_sampling = cmd_args.rsa_sampling
noise_strength = cmd_args.noise_strength
use_distractors_in_sender = cmd_args.use_distractors_in_sender
pop_size = cmd_args.pop_size
use_bullet = cmd_args.use_bullet

if dataset_type == 0: # Even, same pos
	shapes_dataset = 'get_dataset_balanced_incomplete_noise_{}_3_3'.format(noise_strength)
	dataset_name = 'even-samepos'
elif dataset_type == 1: # Even, diff pos
	shapes_dataset = 'get_dataset_different_targets_incomplete_noise_{}_3_3'.format(noise_strength)
	dataset_name = 'even-diffpos'
elif dataset_type == 2: # Uneven, same pos
	shapes_dataset = 'get_dataset_uneven_incomplete_noise_{}_3_3'.format(noise_strength)
	dataset_name = 'uneven-samepos'
elif dataset_type == 3: # Uneven,  diff pos
	shapes_dataset = 'get_dataset_uneven_different_targets_row_incomplete_noise_{}_3_3'.format(noise_strength)
	dataset_name = 'uneven-diffpos'
elif dataset_type == 4: #
	print("Not Supported type")

# Symbolic input or mscoco never need to train visual features
# if not use_symbolic_input and not shapes_dataset is None:
# 	assert should_train_visual or cnn_model_file_name is not None, 'Need stored CNN weights if not training visual features'

# Get model id using a timestamp
if should_train_visual:
	repr = 'train'
else:
	if use_random_model:
		repr = 'random'
	else:
		repr = 'pre'

model_id = 'seed-{}_pop-{}_K-{}_repr-{}_distractor-aware-{}_bullet-{}_data-{}_noise-{}'.format(seed,pop_size, K, repr, use_distractors_in_sender, dataset_name, noise_strength)

dumps_dir = './dumps'
if should_dump and not os.path.exists(dumps_dir):
	os.mkdir(dumps_dir)

current_model_dir = '{}/{}'.format(dumps_dir, model_id)

if should_dump and not os.path.exists(current_model_dir):
	os.mkdir(current_model_dir)

if not should_train_visual:
	if use_random_model:
		cnn_model_file_name = './dumps/random/random_model'
	else:
		to_load_model_id = model_id.replace('pre', 'train')
		to_load_current_model_dir = current_model_dir.replace('pre', 'train')
		cnn_model_file_name = '{}/{}_{}_model'.format(to_load_current_model_dir, to_load_model_id, EPOCHS - 1)

starting_epoch = 0

wandb.init(project="referential-shapes", name=model_id)

wandb.config.K = K #int(sys.argv[1])
wandb.config.seed = seed #int(sys.argv[1])
wandb.config.vocab_size = vocab_size #int(sys.argv[2])
wandb.config.max_sentence_length = max_sentence_length #int(sys.argv[3])
wandb.config.dataset_name = dataset_name #sys.argv[4]
wandb.config.vl_loss_weight = vl_loss_weight #float(sys.argv[5])
wandb.config.bound_weight = bound_weight #float(sys.argv[6])
wandb.config.use_symbolic_input = use_symbolic_input
wandb.config.should_train_visual = should_train_visual
wandb.config.cnn_model_file_name = cnn_model_file_name
wandb.config.use_random_model = use_random_model
wandb.config.rsa_sampling = rsa_sampling
wandb.config.noise_strength = noise_strength
wandb.config.repr = repr
wandb.config.exp_id = model_id[6:]

################# Print info ####################
print('========================================')
print('Model id: {}'.format(model_id))
print('Seed: {}'.format(seed))
print('Training visual module: {}'.format(should_train_visual))
if not should_train_visual and not use_symbolic_input and not shapes_dataset is None:
	print('Loading pretrained CNN from: {}'.format(cnn_model_file_name))
print('|V|: {}'.format(vocab_size))
print('L: {}'.format(max_sentence_length))
print('Using gpu: {}'.format(use_gpu))
if not shapes_dataset is None:
	print('Dataset: {} ({})'.format(shapes_dataset, 'symbolic' if use_symbolic_input else 'pixels'))
else:
	print('Dataset: mscoco')
print('Lambda: {}'.format(vl_loss_weight))
print('Alpha: {}'.format(bound_weight))
if not use_symbolic_input and not shapes_dataset is None:
	print('N image features: {}'.format(n_image_features))
if rsa_sampling >= 0:
	print('N samples for RSA: {}'.format(rsa_sampling))
print()
#################################################

print("build vocab")
if not shapes_dataset is None:
	# Create vocab if there is not one for the desired size already
	if not does_vocab_exist(vocab_size):
		build_vocab(vocab_size)

print("loading vocab")
# Load vocab
word_to_idx, idx_to_word, bound_idx = load_dictionaries(
	'shapes' if not shapes_dataset is None else 'mscoco',
	vocab_size)
print("loading pretrained cnn")
# Load pretrained CNN if necessary
if not should_train_visual and not use_symbolic_input and not shapes_dataset is None:
	cnn_model_id = cnn_model_file_name.split('/')[-1]

	features_folder_name = 'data/shapes/{}_{}'.format(shapes_dataset, cnn_model_id)

	# Check if the features were already extracted with this CNN
	if not os.path.exists(features_folder_name):
		# Load CNN from dumped model
		state = torch.load(cnn_model_file_name, map_location= lambda storage, location: storage)
		cnn_state = {k[4:]:v for k,v in state.items() if 'cnn' in k}
		trained_cnn = CNN(n_image_features)
		trained_cnn.load_state_dict(cnn_state)

		if use_gpu:
			trained_cnn = trained_cnn.cuda()

		print("=CNN state loaded=")
		print("Extracting features...")

		# Dump the features to then load them
		features_folder_name = save_features(trained_cnn, shapes_dataset, cnn_model_id)

print("crating one hot metadata")
if not shapes_dataset is None:
	# Create onehot metadata if not created yet
	if not does_shapes_onehot_metadata_exist(shapes_dataset):
		create_shapes_onehot_metadata(shapes_dataset)

	# Load metadata
	train_metadata, valid_metadata, test_metadata, noise_metadata = load_shapes_onehot_metadata(shapes_dataset)
else:
	train_metadata = None
	valid_metadata = None
	test_metadata = None
	noise_metadata = None
print("loaded metadata")
print("loading data")
# Load data
if not shapes_dataset is None:
	if not use_symbolic_input:
		if should_train_visual:
			train_data, valid_data, test_data, noise_data = load_images('shapes/{}'.format(shapes_dataset), BATCH_SIZE, K)
		else:
			n_pretrained_image_features, train_data, valid_data, test_data, noise_data = load_pretrained_features(
				features_folder_name, BATCH_SIZE, K)
			assert n_pretrained_image_features == n_image_features
	else:
		n_image_features, train_data, valid_data, test_data, noise_data= load_pretrained_features(
			'shapes/{}'.format(shapes_dataset), BATCH_SIZE, K, use_symbolic=True)
else:
	n_image_features, train_data, valid_data, test_data, noise_data = load_pretrained_features(
			'data/mscoco', BATCH_SIZE, K)
	print('\nUsing {} image features\n'.format(n_image_features))

print("data loaded")
# Settings

print("creating model")
model = ModelPop(n_image_features, vocab_size,
	EMBEDDING_DIM, HIDDEN_SIZE,
	bound_idx, max_sentence_length,
	vl_loss_weight, bound_weight,
	should_train_visual, rsa_sampling,
	use_gpu, K, use_distractors_in_sender, pop_size)

wandb.watch(model)

print("model created")
if use_gpu:
	model = model.cuda()
print("model moved to gpu")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
es = EarlyStopping(mode="max", patience=10, threshold=0.005, threshold_mode="rel") # Not 30 patience

# Init metric trackers
losses_meters = []
eval_losses_meters = []

accuracy_meters = []
eval_accuracy_meters = []
noise_accuracy_meters = []

entropy_meters = []
eval_entropy_meters = []

distinctness_meters = []
eval_distinctness_meters = []

rsa_sr_meters = []
eval_rsa_sr_meters = []

rsa_si_meters = []
eval_rsa_si_meters = []

rsa_ri_meters = []
eval_rsa_ri_meters = []

topological_sim_meters = []
eval_topological_sim_meters = []

posdis_meters = []
eval_posdis_meters = []

bosdis_meters = []
eval_bosdis_meters = []

language_entropy_meters = []
eval_language_entropy_meters = []


word_counts = torch.zeros([vocab_size])
if use_gpu:
	word_counts = word_counts.cuda()

eval_word_counts = torch.zeros([vocab_size])
if use_gpu:
	eval_word_counts = eval_word_counts.cuda()

is_loss_nan = False
should_evaluate_best = False

train_start_time = time.time()
print("init done, start epochs")
# Train
for epoch in range(EPOCHS):
	epoch_start_time = time.time()

	e = epoch + starting_epoch
	print("train one epoch start")
	(epoch_loss_meter,
	epoch_acc_meter,
	messages,
	indices,
	epoch_w_counts,
	epoch_entropy_meter,
	epoch_distinctness_meter,
	epoch_rsa_sr_meter,
	epoch_rsa_si_meter,
	epoch_rsa_ri_meter,
	epoch_topological_sim_meter,
	epoch_posdis_meter,
	epoch_bosdis_meter,
	epoch_lang_entropy_meter) = train_one_epoch(model, train_data, optimizer, word_counts, train_metadata, debugging)
	print("done one epoch")
	model.shuffle_pair()
	if math.isnan(epoch_loss_meter.avg):
		print("The train loss in NaN. Stop training")
		is_loss_nan = True
		break

	losses_meters.append(epoch_loss_meter)
	accuracy_meters.append(epoch_acc_meter)
	entropy_meters.append(epoch_entropy_meter)
	distinctness_meters.append(epoch_distinctness_meter)
	rsa_sr_meters.append(epoch_rsa_sr_meter)
	rsa_si_meters.append(epoch_rsa_si_meter)
	rsa_ri_meters.append(epoch_rsa_ri_meter)
	topological_sim_meters.append(epoch_topological_sim_meter)
	posdis_meters.append(epoch_posdis_meter)
	bosdis_meters.append(epoch_bosdis_meter)
	language_entropy_meters.append(epoch_lang_entropy_meter)
	word_counts += epoch_w_counts

	(eval_loss_meter,
	eval_acc_meter,
	eval_messages,
	eval_indices,
	_w_counts,
	eval_entropy_meter,
	eval_distinctness_meter,
	eval_rsa_sr_meter,
	eval_rsa_si_meter,
	eval_rsa_ri_meter,
	eval_topological_sim_meter,
	eval_posdis_meter,
	eval_bosdis_meter,
	eval_lang_entropy_meter) = evaluate(model, valid_data, eval_word_counts, valid_metadata, debugging)

	model.shuffle_pair()
	eval_losses_meters.append(eval_loss_meter)
	eval_accuracy_meters.append(eval_acc_meter)
	eval_entropy_meters.append(eval_entropy_meter)
	eval_distinctness_meters.append(eval_distinctness_meter)
	eval_rsa_sr_meters.append(eval_rsa_sr_meter)
	eval_rsa_si_meters.append(eval_rsa_si_meter)
	eval_rsa_ri_meters.append(eval_rsa_ri_meter)
	eval_topological_sim_meters.append(eval_topological_sim_meter)
	eval_posdis_meters.append(eval_posdis_meter)
	eval_bosdis_meters.append(eval_bosdis_meter)
	eval_language_entropy_meters.append(eval_lang_entropy_meter)

	(_,
	noise_acc_meter,
	_,
	_,
	_,
	_,
	_,
	_,
	_,
	_,
	_,
	_,
	_,
	_) = evaluate(model, noise_data, eval_word_counts, noise_metadata, debugging)
	noise_accuracy_meters.append(noise_acc_meter)
	model.shuffle_pair()
	print('Epoch {}, average train loss: {}, average val loss: {} \n average accuracy: {}, average val accuracy: {}, average noise accuracy: {} \n'.format(
		e, losses_meters[e].avg, eval_losses_meters[e].avg, accuracy_meters[e].avg, eval_accuracy_meters[e].avg, noise_accuracy_meters[e].avg))
	if rsa_sampling > 0:
		print('	RSA sender-receiver: {}, RSA sender-input: {}, RSA receiver-input: {} \n Topological sim: {} \n'.format(
			epoch_rsa_sr_meter.avg, epoch_rsa_si_meter.avg, epoch_rsa_ri_meter.avg, epoch_topological_sim_meter.avg))
		print(' Train posdis: {}, Train posdis: {}, Eval posdis: {}, Eval bosdis: {}'.format(
			epoch_bosdis_meter.avg, epoch_bosdis_meter.avg, eval_posdis_meter.avg, eval_bosdis_meter.avg))
		print('	Eval RSA sender-receiver: {}, Eval RSA sender-input: {}, Eval RSA receiver-input: {}\n Eval Topological sim: {}\n'.format(
			eval_rsa_sr_meter.avg, eval_rsa_si_meter.avg, eval_rsa_ri_meter.avg, eval_topological_sim_meter.avg))

	wandb.log({'Epoch':e, 'average train loss': losses_meters[e].avg, 'average val loss': eval_losses_meters[e].avg, 'average accuracy': accuracy_meters[e].avg, 'average val accuracy': eval_accuracy_meters[e].avg, 'average noise accuracy': noise_accuracy_meters[e].avg})
	wandb.log({'RSA sender-receiver': epoch_rsa_sr_meter.avg, 'RSA sender-input': epoch_rsa_si_meter.avg, 'RSA receiver-input':epoch_rsa_ri_meter.avg})
	wandb.log({'Topological sim':epoch_topological_sim_meter.avg, 'Posdis':epoch_bosdis_meter.avg, 'Bosdis':epoch_bosdis_meter.avg})
	wandb.log({'RSA sender-receiver': eval_rsa_sr_meter.avg, 'RSA sender-input': eval_rsa_si_meter.avg, 'RSA receiver-input':eval_rsa_ri_meter.avg})
	wandb.log({'Eval Topological sim':eval_topological_sim_meter.avg, 'Eval Posdis':eval_posdis_meter.avg, 'Eval Bosdis':eval_bosdis_meter.avg})

	seconds_current_epoch = time.time() - epoch_start_time
	print('    (Took {} seconds)'.format(seconds_current_epoch))

	# es.step(eval_acc_meter.avg)

	if should_dump:
		# Save model every epoch
		if e%5==0 or e>=EPOCHS-1:
			torch.save(model.state_dict(), '{}/{}_{}_model'.format(current_model_dir, model_id, e))
			torch.save(model.state_dict(), '{}/{}_{}_model'.format(wandb.run.dir, model_id, e))

		# Dump messages every epoch
		pickle.dump(messages, open('{}/{}_{}_messages.p'.format(current_model_dir, model_id, e), 'wb'))
		pickle.dump(eval_messages, open('{}/{}_{}_eval_messages.p'.format(current_model_dir, model_id, e), 'wb'))

		# Dump indices as often as messages
		if should_dump_indices:
			pickle.dump(indices, open('{}/{}_{}_imageIndices.p'.format(current_model_dir, model_id, e), 'wb'))
			pickle.dump(eval_indices, open('{}/{}_{}_eval_imageIndices.p'.format(current_model_dir, model_id, e), 'wb'))

		if should_covert_to_words:
			dump_words(current_model_dir, messages, idx_to_word, '{}_{}_messages'.format(model_id, e))
			dump_words(current_model_dir, eval_messages, idx_to_word, '{}_{}_eval_messages'.format(model_id, e))

	# if es.is_converged:
	# 	print("Converged in epoch {}".format(e))
	# 	break

	if seconds_current_epoch * (e+1) >= 75000:
		print("Stopping because wall time limit is close")
		break

print()
print('Training took {} seconds'.format(time.time() - train_start_time))

if is_loss_nan:
	should_dump = False
	should_evaluate_best = False

# if should_dump:
	# # Dump latest stats
	# pickle.dump(losses_meters, open('{}/{}_{}_losses_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	# pickle.dump(eval_losses_meters, open('{}/{}_{}_eval_losses_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	# pickle.dump(accuracy_meters, open('{}/{}_{}_accuracy_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	# pickle.dump(eval_accuracy_meters, open('{}/{}_{}_eval_accuracy_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	# pickle.dump(entropy_meters, open('{}/{}_{}_entropy_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	# pickle.dump(eval_entropy_meters, open('{}/{}_{}_eval_entropy_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	# pickle.dump(distinctness_meters, open('{}/{}_{}_distinctness_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	# pickle.dump(eval_distinctness_meters, open('{}/{}_{}_eval_distinctness_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	# pickle.dump(rsa_sr_meters, open('{}/{}_{}_rsa_sr_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	# pickle.dump(eval_rsa_sr_meters, open('{}/{}_{}_eval_rsa_sr_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	# pickle.dump(rsa_si_meters, open('{}/{}_{}_rsa_si_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	# pickle.dump(eval_rsa_si_meters, open('{}/{}_{}_eval_rsa_si_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	# pickle.dump(rsa_ri_meters, open('{}/{}_{}_rsa_ri_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	# pickle.dump(eval_rsa_ri_meters, open('{}/{}_{}_eval_rsa_ri_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	# pickle.dump(topological_sim_meters, open('{}/{}_{}_topological_sim_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	# pickle.dump(eval_topological_sim_meters, open('{}/{}_{}_eval_topological_sim_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	# pickle.dump(language_entropy_meters, open('{}/{}_{}_language_entropy_meters.p'.format(current_model_dir, model_id, e), 'wb'))
	# pickle.dump(eval_language_entropy_meters, open('{}/{}_{}_eval_language_entropy_meters.p'.format(current_model_dir, model_id, e), 'wb'))


# Evaluate best model on test data
if should_evaluate_best:

	if debugging:
		# Just pick the latest
		best_model = model
		best_epoch = e
	else:
		# Actually pick the best
		best_epoch = np.argmax([m.avg for m in eval_accuracy_meters])
		best_model = Model(n_image_features, vocab_size,
			EMBEDDING_DIM, HIDDEN_SIZE,
			bound_idx, max_sentence_length,
			vl_loss_weight, bound_weight,
			should_train_visual, rsa_sampling,
			use_gpu)
		best_model_name = '{}/{}_{}_model'.format(current_model_dir, model_id, best_epoch)
		state = torch.load(best_model_name, map_location= lambda storage, location: storage)
		best_model.load_state_dict(state)

		print()
		print('Best model is in file: {}'.format(best_model_name))

	if use_gpu:
		best_model = best_model.cuda()

	test_word_counts = torch.zeros([vocab_size])
	if use_gpu:
		test_word_counts = test_word_counts.cuda()

	(test_loss_meter,
	test_acc_meter,
	test_messages,
	test_indices,
	_w_counts,
	test_entropy_meter,
	test_distinctness_meter,
	test_rsa_sr_meter,
	test_rsa_si_meter,
	test_rsa_ri_meter,
	test_topological_sim_meter,
	test_language_entropy_meter) = evaluate(best_model, test_data, test_word_counts, test_metadata, debugging)

	print()
	print('Test accuracy: {}'.format(test_acc_meter.avg))

	if should_dump:
		pickle.dump(test_loss_meter, open('{}/{}_{}_test_losses_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_acc_meter, open('{}/{}_{}_test_accuracy_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_entropy_meter, open('{}/{}_{}_test_entropy_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_distinctness_meter, open('{}/{}_{}_test_distinctness_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_rsa_sr_meter, open('{}/{}_{}_test_rsa_sr_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_rsa_si_meter, open('{}/{}_{}_test_rsa_si_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_rsa_ri_meter, open('{}/{}_{}_test_rsa_ri_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_topological_sim_meter, open('{}/{}_{}_test_topological_sim_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_language_entropy_meter, open('{}/{}_{}_test_language_entropy_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_messages, open('{}/{}_{}_test_messages.p'.format(current_model_dir, model_id, best_epoch), 'wb'))

		if should_dump_indices:
			pickle.dump(test_indices, open('{}/{}_{}_test_imageIndices.p'.format(current_model_dir, model_id, best_epoch), 'wb'))

		if should_covert_to_words:
			dump_words(current_model_dir, test_messages, idx_to_word, '{}_{}_test_messages'.format(model_id, best_epoch))
