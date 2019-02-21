import sys
import os
import pickle
import numpy as np

from utils import compute_stats, plot_token_distribution, get_attributes


assert len(sys.argv) == 2, 'You need a model id'

model_id = sys.argv[1]

data_folder = 'balanced'

dumps_dir = '../dumps'
model_dir = '{}/{}'.format(dumps_dir, model_id)

messages_file_name = ['{}/{}'.format(model_dir, f) for f in os.listdir(model_dir) if 'test_messages.p' in f]

assert len(messages_file_name) == 1
messages_file_name = messages_file_name[0]

# Load messages
x = pickle.load(open(messages_file_name, 'rb'))

# First token is always <S>
padding_token = x[0][0]

# Grab stats
min_len, max_len, avg_len, counter = compute_stats(x, padding_token)

print('Min message length:', min_len)
print('Max message length:', max_len)
print('Avg message length:', avg_len)
n_utt = len(counter)
print('Number of used tokens', n_utt)
common_percent = 0.3
print('Top 30 percent most used tokens', counter.most_common(int(n_utt * common_percent)))

plot_token_distribution(counter, model_id)

metadata = pickle.load(open('../shapes/{}/test.metadata.p'.format(data_folder), 'rb'))
# print(len(metadata))
# print(metadata[0])

dict = get_attributes(messages, padding_token, metadata)
