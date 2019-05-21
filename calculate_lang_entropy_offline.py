import os
import sys
import pickle

from utils import discretize_messages, AverageMeter
from entropy import language_entropy

def get_epoch(filename):
	e = int(filename.split('_')[1])
	return e

def get_sorted_message_file_names(model_dir, file_name_id):
	file_names = [f for f in os.listdir(model_dir) if file_name_id in f]
	file_names.sort(key=get_epoch)
	return ['{}/{}'.format(model_dir, f) for f in file_names]



model_ids = sys.argv[1:]

for model_id in model_ids:
	print('Model: {}'.format(model_id))

	current_model_dir = 'dumps/{}'.format(model_id)
	file_names = get_sorted_message_file_names(current_model_dir, '_messages.p') # Train, test, val
	id_for_dump = file_names[0].split('/')[-1].split('_')[0]

	# Three dumped files (one per set)
	train_meters = []
	val_meters = []

	test_meter = AverageMeter()

	for f in file_names:
		messages = pickle.load(open(f, 'rb'))
		entropy = language_entropy(messages.cpu())

		# One AverageMeter per epoch
		if 'test' in f:
			test_meter.update(entropy)
			best_epoch = f.split('_')[-3]

		elif 'eval' in f:
			meter = AverageMeter()
			meter.update(entropy)
			val_meters.append(meter)
		else:
			meter = AverageMeter()
			meter.update(entropy)
			train_meters.append(meter)
			e = f.split('_')[-2]

	print('Done with model {}'.format(model_id))


	pickle.dump(test_meter, open('{}/{}_{}_test_language_entropy_meter.p'.format(current_model_dir, id_for_dump, best_epoch), 'wb'))
	pickle.dump(train_meters, open('{}/{}_{}_language_entropy_meters.p'.format(current_model_dir, id_for_dump, e), 'wb'))
	pickle.dump(val_meters, open('{}/{}_{}_eval_language_entropy_meters.p'.format(current_model_dir, id_for_dump, e), 'wb'))