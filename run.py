from utils import AverageMeter
import torch
import torch.nn as nn


def train_one_epoch(model, data, optimizer, word_counts, debugging=False):

	model.train()

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()

	for i, d in enumerate(data):
		optimizer.zero_grad()

		target, distractors = d

		loss, acc, _m, batch_w_counts = model(target, distractors, word_counts)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())

		if i == 0:
			w_counts = batch_w_counts
		else:
			w_counts += batch_w_counts

		loss.backward()
		optimizer.step()

		if debugging: #and i == 2:
			break

	return loss_meter, acc_meter, w_counts

def evaluate(model, data, word_counts, debugging=False):
	
	model.eval()

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	messages = []

	count  = 0
	for d in data:
		count += 1
		target, distractors = d

		loss, acc, m, _w_counts = model(target, distractors, word_counts)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())
		messages.append(m)
		
		if debugging:
			break
	
	return loss_meter, acc_meter, torch.cat(messages, 0)

