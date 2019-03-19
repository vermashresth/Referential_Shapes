import pickle
import torch
import os
import sys

from dataloader import load_dictionaries

use_gpu = torch.cuda.is_available()

assert len(sys.argv) == 2, 'Need dumped messages path'

messages_path = sys.argv[1]

model_id, vocab_size, l = messages_path.split('_')

# Load vocab
_word_to_idx, idx_to_word, _bound_idx = load_dictionaries('shapes', vocab_size)


# Settings
dumps_dir = './dumps'

current_model_dir = '{}/{}_{}_{}'.format(dumps_dir, model_id, vocab_size, l)

test_messages_filename = [f for f in os.listdir(current_model_dir) if 'test_messages.p' in f]
assert len(test_messages_filename) == 1, 'More than one file?'
test_messages_filename = test_messages_filename[0]

test_messages = pickle.load(open('{}/{}'.format(current_model_dir, test_messages_filename), 'rb'))

# Decode test messages
words = []

for m in test_messages:
	words.append([idx_to_word[t] for t in m])

res_filename = '{}/{}_w.p'.format(current_model_dir, test_messages_filename[:-2])

pickle.dump(words, open(res_filename, 'wb'))

print('File dumped to {}'.format(res_filename))
