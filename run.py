from utils import AverageMeter
import torch
import torch.nn as nn

debugging = not torch.cuda.is_available()


def train_one_epoch(
	model, data, optimizer, word_to_idx, start_token, max_sentence_length,
	baseline, baseline_optimizer):

	model.train()
	baseline.train()

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	messages = []

	if debugging:
		count = 0

	for d in data:
		optimizer.zero_grad()
		baseline_optimizer.zero_grad()

		target, distractors = d

		baseline_value = baseline(target, distractors)

		loss, acc, m, reward = model(target, distractors, word_to_idx, start_token, max_sentence_length, baseline_value.data)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())
		messages.append(m)

		loss.backward()
		optimizer.step()


		# Baseline loss
		mse = nn.MSELoss()
		baseline_loss = mse(baseline_value, reward)

		baseline_loss.backward()		
		baseline_optimizer.step()


		if debugging:
			count += 1
			if count == 50:
				break

	return loss_meter, acc_meter, torch.cat(messages, 0)


def evaluate(
	model, data, word_to_idx, start_token, max_sentence_length,
	baseline):
	
	model.eval()
	baseline.eval()

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	messages = []

	for d in data:
		target, distractors = d

		baseline_value = baseline(target, distractors)

		loss, acc, m, _ = model(target, distractors, word_to_idx, start_token, max_sentence_length, baseline_value.data)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())
		messages.append(m)

		if debugging:
			break

	return loss_meter, acc_meter, torch.cat(messages, 0)

