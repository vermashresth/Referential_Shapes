from utils import AverageMeter

def train_one_epoch(model, data, optimizer, word_to_idx, start_token, max_sentence_length):
	model.train()

	### Remove #####
	counter = 0

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()

	for d in data:
		optimizer.zero_grad()

		target, distractors = d
	
		loss, acc = model(target, distractors, word_to_idx, start_token, max_sentence_length)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())

		loss.backward()
		
		optimizer.step()
		
		##### REMOVE ######
		counter += 1
		if counter == 10:
			break

	return loss_meter, acc_meter


def evaluate(model, data, word_to_idx, start_token, max_sentence_length):
	model.eval()

	### Remove #####
	counter = 0

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()

	for d in data:
		target, distractors = d

		loss, acc = model(target, distractors, word_to_idx, start_token, max_sentence_length)

		loss_meter.update(loss.item())
		acc_meter.update(acc.item())

		##### REMOVE ######
		counter +=1
		if counter == 10:
			break

	return loss_meter, acc_meter
