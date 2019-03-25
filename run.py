from utils import AverageMeter
import torch
import torch.nn as nn


def train_one_epoch(model, data, optimizer, word_counts, debugging=False):

	model.train()

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	w_counts = word_counts

	for d in data:
		optimizer.zero_grad()

		target, distractors = d

		loss, acc, _m, batch_w_counts = model(target, distractors, w_counts)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())
		w_counts += batch_w_counts

		loss.backward()
		optimizer.step()

		if debugging:
			break

	return loss_meter, acc_meter, w_counts

def evaluate(model, data, word_counts, debugging=False):
	
	model.eval()

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	messages = []
	w_counts = word_counts

	count  = 0
	for d in data:
		count += 1
		target, distractors = d

		loss, acc, m, batch_w_counts = model(target, distractors, w_counts)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())
		messages.append(m)
		w_counts += batch_w_counts
		
		if debugging:
			break
	
	return loss_meter, acc_meter, torch.cat(messages, 0), w_counts

