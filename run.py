from utils import AverageMeter

def train_one_epoch(model, data, optimizer, word_to_idx, start_token, max_sentence_length, use_gpu):
	model.train()

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()

	for d in data:
		optimizer.zero_grad()

		target, distractors = d
	
		if use_gpu:
			target = target.cuda()
			for d in distractors:
				d = d.cuda()

		loss, acc = model(target, distractors, word_to_idx, start_token, max_sentence_length)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())

		loss.backward()
		
		optimizer.step()

	return loss_meter, acc_meter


def evaluate(model, data, word_to_idx, start_token, max_sentence_length, use_gpu):
	model.eval()

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()

	for d in data:
		target, distractors = d

		if use_gpu:
			target = target.cuda()
			for d in distractors:
				d = d.cuda()

		loss, acc = model(target, distractors, word_to_idx, start_token, max_sentence_length)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())

	return loss_meter, acc_meter
