from utils import AverageMeter, discretize_messages
import torch
import torch.nn as nn
import time
import numpy as np

def run_epoch(model, data, word_counts, optimizer, onehot_metadata, debugging):
	is_training_mode = not optimizer is None

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	entropy_meter = AverageMeter()
	distinctness_meter = AverageMeter()
	rsa_sr_meter = AverageMeter()
	rsa_si_meter = AverageMeter()
	rsa_ri_meter = AverageMeter()
	topological_sim_meter = AverageMeter()
	language_entropy_meter = AverageMeter()
	posdis_meter = AverageMeter()
	bosdis_meter = AverageMeter()
	messages = []
	indices = []
	w_counts = word_counts.clone()

	debugging_counter = 0
	idx = 0
	a1 = time.time()
	all_data_len = len(data)
	log_freq = all_data_len//5
	for d in data:
		if is_training_mode:
			optimizer.zero_grad()
		a2 = time.time()
		# if idx%log_freq==0:
		# 	print("Data loading time", a2-a1)
		target, distractors, idxs = d

		(loss,
		acc,
		m,
		_,
		batch_w_counts,
		entropy,
		distinctness,
		rsa_sr,
		rsa_si,
		rsa_ri,
		topological_sim,
		posdis,
		bosdis,
		lang_entropy) = model(target,
								distractors,
								w_counts,
								onehot_metadata[idxs[:,0]] if onehot_metadata is not None else None)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())
		entropy_meter.update(entropy.item())
		distinctness_meter.update(distinctness)
		rsa_sr_meter.update(rsa_sr)
		rsa_si_meter.update(rsa_si)
		rsa_ri_meter.update(rsa_ri)
		topological_sim_meter.update(topological_sim)
		if not np.isnan(posdis):
			posdis_meter.update(posdis)
		if not np.isnan(bosdis):
			bosdis_meter.update(bosdis)
		language_entropy_meter.update(lang_entropy)

		messages.append(discretize_messages(m) if is_training_mode else m)
		indices.append(idxs)
		w_counts += batch_w_counts

		if is_training_mode:
			loss.backward()
			optimizer.step()

		debugging_counter += 1

		if debugging and debugging_counter == 5:
			break
		a1 = time.time()
		# if idx%log_freq==0:
		# 	print("Trainign data id ", idx, " / ", all_data_len)
		# 	print("training time ", a1-a2)
		idx+=1

	return (loss_meter,
		acc_meter,
		torch.cat(messages, 0),
		torch.cat(indices, 0),
		w_counts,
		entropy_meter,
		distinctness_meter,
		rsa_sr_meter,
		rsa_si_meter,
		rsa_ri_meter,
		topological_sim_meter,
		posdis_meter,
		bosdis_meter,
		language_entropy_meter)


def train_one_epoch(model, data, optimizer, word_counts, onehot_metadata, debugging=False):
	model.train()
	return run_epoch(model, data, word_counts, optimizer, onehot_metadata, debugging)

def evaluate(model, data, word_counts, onehot_metadata, debugging=False):
	model.eval()
	return run_epoch(model, data, word_counts, None, onehot_metadata, debugging)
