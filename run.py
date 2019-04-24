from utils import AverageMeter, discretize_messages
import torch
import torch.nn as nn

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
	messages = []
	indices = []
	w_counts = word_counts.clone()

	for d in data:
		if is_training_mode:
			optimizer.zero_grad()

		target, distractors, idxs = d

		(loss,
		acc, 
		m, 
		batch_w_counts, 
		entropy, 
		distinctness,
		rsa_sr,
		rsa_si,
		rsa_ri,
		topological_sim) = model(target, distractors, w_counts, onehot_metadata[idxs[:,0]])

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())
		entropy_meter.update(entropy.item())
		distinctness_meter.update(distinctness)
		rsa_sr_meter.update(rsa_sr)
		rsa_si_meter.update(rsa_si)
		rsa_ri_meter.update(rsa_ri)
		topological_sim_meter.update(topological_sim)

		messages.append(discretize_messages(m) if is_training_mode else m)
		indices.append(idxs)
		w_counts += batch_w_counts

		if is_training_mode:
			loss.backward()
			optimizer.step()

		if debugging:
			break

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
		topological_sim_meter)


def train_one_epoch(model, data, optimizer, word_counts, onehot_metadata, debugging=False):
	model.train()
	return run_epoch(model, data, word_counts, optimizer, onehot_metadata, debugging)

def evaluate(model, data, word_counts, onehot_metadata, debugging=False):
	model.eval()
	return run_epoch(model, data, word_counts, None, onehot_metadata, debugging)