from utils import AverageMeter
import torch
import torch.nn as nn


def discretize_messages(m):
	_, res = torch.max(m, dim=-1)
	return res

def run_epoch(model, data, word_counts, optimizer, debugging):
	is_training_mode = not optimizer is None

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	entropy_meter = AverageMeter()
	distinctness_meter = AverageMeter()
	messages = []
	indices = []
	w_counts = word_counts.clone()

	for d in data:
		if is_training_mode:
			optimizer.zero_grad()

		target, distractors, idxs = d

		loss, acc, m, batch_w_counts, entropy, distinctness = model(target, distractors, w_counts)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())
		entropy_meter.update(entropy.item())
		distinctness_meter.update(distinctness)
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
		distinctness_meter)


def train_one_epoch(model, data, optimizer, word_counts, debugging=False):
	model.train()
	return run_epoch(model, data, word_counts, optimizer, debugging)

def evaluate(model, data, word_counts, debugging=False):
	model.eval()
	return run_epoch(model, data, word_counts, None, debugging)
