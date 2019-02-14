from utils import AverageMeter
import torch
import torch.nn as nn

debugging = not torch.cuda.is_available()


def train_one_epoch(model, data, optimizer, word_to_idx, start_token, max_sentence_length):

	model.train()

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()

	for d in data:
		optimizer.zero_grad()

		target, distractors = d

		loss, acc, _ = model(target, distractors, word_to_idx, start_token, max_sentence_length)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())

		loss.backward()
		optimizer.step()


		if debugging:
			break

	return loss_meter, acc_meter

def evaluate(model, data, word_to_idx, start_token, max_sentence_length):
	
	model.eval()

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	messages = []

	count  = 0
	for d in data:
		count += 1
		target, distractors = d

		loss, acc, m = model(target, distractors, word_to_idx, start_token, max_sentence_length)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())
		messages.append(m)
		
		if debugging:
			break
	
	return loss_meter, acc_meter, torch.cat(messages, 0)

