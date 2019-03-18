import pickle
import os

START_TOKEN = '<SOS>'
END_TOKEN = '<EOS>'
shapes_vocab_path = 'data/shapes'

def does_vocab_exist(vocab_size):
	return os.path.exists('{}/dict_{}.pckl'.format(shapes_vocab_path, vocab_size))

def build_vocab(vocab_size):
	idx_to_word = []
	word_to_idx = {}

	for i in range(vocab_size - 2):
		word = str(i)
		idx_to_word.append(word)
		word_to_idx[word] = i

	idx_to_word.append(START_TOKEN)
	word_to_idx[START_TOKEN] = len(idx_to_word) - 1

	idx_to_word.append(END_TOKEN)
	word_to_idx[END_TOKEN] = len(idx_to_word) - 1


	with open('{}/dict_{}.pckl'.format(shapes_vocab_path, vocab_size), 'wb') as f:
	    pickle.dump({'word_to_idx': word_to_idx,
	                  'idx_to_word': idx_to_word}, f)