from utils import AverageMeter

def train_one_epoch(model, data, optimizer, word_to_idx, start_token, max_sentence_length):
	model.train()

	### Remove #####
	counter = 0

	epoch_loss_meter = AverageMeter()

	for d in data:
		optimizer.zero_grad()

		target, distractors = d
	
		loss = model(target, distractors, word_to_idx, start_token, max_sentence_length)

		epoch_loss_meter.update(loss.item())

		loss.backward()
		
		optimizer.step()
		
		##### REMOVE ######
		counter +=1
		if counter == 10:
			break

	return epoch_loss_meter


def evaluate(model, data, word_to_idx, start_token, max_sentence_length):
	model.eval()

	### Remove #####
	counter = 0

	epoch_loss_meter = AverageMeter()

	for d in data:
		target, distractors = d

		loss = model(target, distractors, word_to_idx, start_token, max_sentence_length)

		epoch_loss_meter.update(loss.item())

		##### REMOVE ######
		counter +=1
		if counter == 10:
			break

	return epoch_loss_meter
