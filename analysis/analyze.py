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
	min_len, max_len, avg_len, counter = compute_stats(messages, padding_idx, idx_to_word) # counter includes <S> (aka EOS)

	n_utt = len(counter)

	# Plots
	top_common_percent = 1 if vocab_size < 50 else 0.3
	plot_token_frequency(counter, n_utt, top_common_percent, model_id, plots_dir)
	plot_token_distribution(counter, model_id, plots_dir)

	# Token correlation with features
	metadata = pickle.load(open('../shapes/{}/test.metadata.p'.format(data_folder), 'rb'))

	token_to_attr, attr_to_token = get_attributes_dicts(messages, metadata, idx_to_word)

	#plot_attributes_per_token(counter, n_utt, top_common_percent, token_to_attr, model_id, plots_dir)
	n_top_tokens = 10
	plot_tokens_per_attribute(attr_to_token, n_top_tokens, model_id, plots_dir)


	return (len(messages[0]), 
			acc_meter.avg,
			min_len,
			max_len,
			avg_len,
			n_utt)




# assert len(sys.argv) >= 3 and len(sys.argv) % 2 != 0, 'You need at least one model id and its vocabulary size'
# You can either input 'model_folder vocab_size' list or it will take the dumps folder #a whole folder that contains everything you want (e.g. dumps)

if len(sys.argv) == 1:
	all_folders_names = os.listdir('../dumps')
	inputs = ['{} {}'.format(folder_name, folder_name[folder_name.find('_')+1:folder_name.rfind('_')]) for folder_name in all_folders_names]
else:
	inputs = sys.argv[1:]


plots_dir = 'plots' # individual
stats_dir = 'tables' # across experiments

if not os.path.exists(plots_dir):
	os.mkdir(plots_dir)

if not os.path.exists(stats_dir):
	os.mkdir(stats_dir)

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

# Read in the settings we want to analyze
for i in range(len(inputs)):
	model_id, vocab_size = inputs[i].split()

	L, acc, min_len, max_len, avg_len, n_utt = get_stats(model_id, vocab_size, data_folder, plots_dir)

	# L = model_id
	# acc = np.random.random()
	# min_len = np.random.randint(1,10)
	# max_len = min_len + np.random.randint(2,5)
	# avg_len = (min_len + max_len) / 2
	# n_utt = np.random.randint(1, vocab_size)

	stats_dict['id'].append(model_id)
	stats_dict['|V|'].append(vocab_size)
	stats_dict['L'].append(L)
	stats_dict['Test acc'].append(acc)
	stats_dict['Min message length'].append(min_len)
	stats_dict['Max message length'].append(max_len)
	stats_dict['Avg message length'].append(avg_len)
	stats_dict['N tokens used'].append(n_utt)

	print('id: {}, |V|: {}, L: {}, Test acc: {}'.format(model_id, vocab_size, L, acc))

# Dump all stats
df = pd.DataFrame(stats_dict)
df.to_csv('{}/all_stats_{}.csv'.format(stats_dir, data_folder), index=None, header=True)


unique_VLs = {}
for idx, (v, l) in enumerate(zip(stats_dict['|V|'], stats_dict['L'])):
	if (v,l) not in unique_VLs:
		unique_VLs[(v,l)] = [idx]
	else:
		unique_VLs[(v,l)].append(idx)

avg_stats_dict = {k: [] for k in stats_dict.keys() if k != 'id'}

for (v,l), idxs in unique_VLs.items():
	avg_stats_dict['|V|'].append(v)
	avg_stats_dict['L'].append(l)

	for measure in list(avg_stats_dict.keys())[2:]:
		acc = sum([stats_dict[measure][idx] for idx in idxs]) / len(idxs)
		avg_stats_dict[measure].append(acc)


# Dump avg stats
df = pd.DataFrame(avg_stats_dict)
df.to_csv('{}/avg_stats_{}.csv'.format(stats_dir, data_folder), index=None, header=True)


plot_acc_per_setting(avg_stats_dict, stats_dir, data_folder)