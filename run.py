from utils import AverageMeter
import torch

def train_one_epoch(model, data, optimizer, word_to_idx, start_token, max_sentence_length, use_gpu):
	model.train()

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	messages = []

	count = 0

	for d in data:
		optimizer.zero_grad()

		target, distractors = d
	
		if use_gpu:
			target = target.cuda()
			for d in distractors:
				d = d.cuda()

		loss, acc, m = model(target, distractors, word_to_idx, start_token, max_sentence_length)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())
		messages.append(m)

		loss.backward()
		
		optimizer.step()

		count += 1
		if count == 10:
			break

	return loss_meter, acc_meter, torch.cat(messages, 0)


def evaluate(model, data, word_to_idx, start_token, max_sentence_length, use_gpu):
	model.eval()

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	messages = []

	count = 0

	for d in data:
		target, distractors = d

		if use_gpu:
			target = target.cuda()
			for d in distractors:
				d = d.cuda()

		loss, acc, m = model(target, distractors, word_to_idx, start_token, max_sentence_length)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())
		messages.append(m)

		count += 1
		if count == 5:
			break

	return loss_meter, acc_meter, torch.cat(messages, 0)

