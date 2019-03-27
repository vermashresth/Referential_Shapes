from utils import AverageMeter
import torch
import torch.nn as nn


def discretize_messages(m):
	_, res = torch.max(m, dim=-1)
	return res

def train_one_epoch(model, data, optimizer, word_counts, debugging=False):

	model.train()

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	entropy_meter = AverageMeter()
	distinctness_meter = AverageMeter()
	messages = []
	w_counts = word_counts.clone()

	for d in data:
		optimizer.zero_grad()

		target, distractors = d

		loss, acc, m, batch_w_counts, entropy, distinctness = model(target, distractors, w_counts)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())
		entropy_meter.update(entropy.item())
		distinctness_meter.update(distinctness)
		messages.append(discretize_messages(m))
		w_counts += batch_w_counts

		loss.backward()
		optimizer.step()

		if debugging:
			break

	return (loss_meter, 
		acc_meter, 
		torch.cat(messages, 0), 
		w_counts, 
		entropy_meter,
		distinctness_meter)

def evaluate(model, data, word_counts, debugging=False):
	
	model.eval()

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	entropy_meter = AverageMeter()
	distinctness_meter = AverageMeter()
	messages = []
	w_counts = word_counts.clone()

	count  = 0
	for d in data:
		target, distractors = d

		loss, acc, m, batch_w_counts, entropy, distinctness = model(target, distractors, w_counts)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())
		entropy_meter.update(entropy.item())
		distinctness_meter.update(distinctness)
		messages.append(m)
		w_counts += batch_w_counts
		
		if debugging:
			break
	
	return (loss_meter, 
		acc_meter, 
		torch.cat(messages, 0), 
		w_counts, 
		entropy_meter,
		distinctness_meter)

