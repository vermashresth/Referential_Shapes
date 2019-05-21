import sys
import numpy as np

from utils import *
# from stats_calculator import get_test_messages_stats


def compute_variance(messages, padding_idx, idx_to_word):
	lengths = []
	for m in messages:
		length = get_message_length(m, padding_idx)
		lengths.append(length)

	arr = np.array(lengths)

	return np.var(arr), np.std(arr)

def get_test_messages_variance(model_id, vocab_size):
	_, idx_to_word, padding_idx = load_dictionaries(vocab_size)

	model_dir = get_model_dir(model_id)

	# Load messages
	messages = get_pickle_file(model_dir, 'test_messages.p')

	var, std = compute_variance(messages, padding_idx, idx_to_word) # counter includes <S> (aka EOS)

	return var, std


model_ids = sys.argv[1:]
vocab_size = 25

for model_id in model_ids:
	variance, standard_deviation = get_test_messages_variance(model_id, vocab_size)
	print('Mode id: {}, Variance: {}, Std: {}'.format(model_id, variance, standard_deviation))



