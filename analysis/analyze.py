import sys
import os
import pickle
import numpy as np
import pandas as pd

from utils import *


def get_stats(model_id, vocab_size, data_folder, plots_dir):
	_, idx_to_word, padding_idx = load_dictionaries(vocab_size)


	dumps_dir = '../dumps'
	model_dir = '{}/{}'.format(dumps_dir, model_id)

	accuracy_file_name = ['{}/{}'.format(model_dir, f) for f in os.listdir(model_dir) if 'test_accuracy_meter.p' in f]
	assert len(accuracy_file_name) == 1
	accuracy_file_name = accuracy_file_name[0]
	acc_meter = pickle.load(open(accuracy_file_name, 'rb'))

	messages_file_name = ['{}/{}'.format(model_dir, f) for f in os.listdir(model_dir) if 'test_messages.p' in f]

	assert len(messages_file_name) == 1
	messages_file_name = messages_file_name[0]

	# Load messages
	messages = pickle.load(open(messages_file_name, 'rb'))

	# Grab stats
	min_len, max_len, avg_len, counter = compute_stats(messages, padding_idx, idx_to_word)

	n_utt = len(counter)

	# Plots
	top_common_percent = 0.3
	plot_token_frequency(counter, n_utt, top_common_percent, model_id, plots_dir)
	plot_token_distribution(counter, model_id, plots_dir)

	# Token correlation with features
	metadata = pickle.load(open('../shapes/{}/test.metadata.p'.format(data_folder), 'rb'))

	token_to_attr, attr_to_token = get_attributes_dicts(messages, padding_idx, metadata, idx_to_word)

	plot_attributes_per_token(counter, n_utt, top_common_percent, token_to_attr, model_id, plots_dir)
	n_top_tokens = 10
	plot_tokens_per_attribute(attr_to_token, n_top_tokens, model_id, plots_dir)


	return (len(messages[0]), 
			acc_meter.avg,
			min_len,
			max_len,
			avg_len,
			n_utt)




assert len(sys.argv) >= 3 and len(sys.argv) % 2 != 0, 'You need at least one model id and its vocabulary size'


plots_dir = 'plots'

if not os.path.exists(plots_dir):
	os.mkdir(plots_dir)

data_folder = 'balanced'
stats_dict = {
	'id': [],
	'|V|' : [],
	'L' : [],
	'Test acc': [],
	'Min message length': [],
	'Max message length' : [],
	'Avg message length': [],
	'N tokens used': []
}

for i in range(1, len(sys.argv), 2):
	model_id = sys.argv[i]
	vocab_size = sys.argv[i+1]

	L, acc, min_len, max_len, avg_len, n_utt = get_stats(model_id, vocab_size, data_folder, plots_dir)

	stats_dict['id'].append(model_id)
	stats_dict['|V|'].append(vocab_size)
	stats_dict['L'].append(L)
	stats_dict['Test acc'].append(acc)
	stats_dict['Min message length'].append(min_len)
	stats_dict['Max message length'].append(max_len)
	stats_dict['Avg message length'].append(avg_len)
	stats_dict['N tokens used'].append(n_utt)



df = pd.DataFrame(stats_dict)
df.to_csv('{}/stats_{}.csv'.format(plots_dir, data_folder), index=None, header=True)


